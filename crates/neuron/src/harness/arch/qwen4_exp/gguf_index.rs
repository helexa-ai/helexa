//! Where each tensor of a split GGUF lies.
//!
//! A deliberately small reader: the native expert path needs names,
//! shapes, types and absolute offsets, and must accept quantisation
//! types candle's GGUF reader has no enum value for.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

/// One tensor's location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub shard: usize,
    pub offset: u64,
    /// In GGUF order, innermost dimension first.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
}

/// Every tensor of a model, across its shards.
#[derive(Debug)]
pub struct GgufIndex {
    pub shards: Vec<PathBuf>,
    pub tensors: HashMap<String, GgufTensor>,
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64(r)?;
    ensure!(len <= 1 << 30, "implausible GGUF string length {len}");
    let mut b = vec![0; len as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).context("GGUF string is not UTF-8")
}

fn skip(r: &mut (impl Read + Seek), bytes: i64) -> Result<()> {
    r.seek(SeekFrom::Current(bytes))?;
    Ok(())
}

fn scalar_size(value_type: u32) -> Option<i64> {
    match value_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Skips one value, returning it when it is an unsigned 32-bit integer
/// (the type `general.alignment` is stored as).
fn skip_value(r: &mut (impl Read + Seek), value_type: u32) -> Result<Option<u32>> {
    if value_type == 4 {
        return Ok(Some(read_u32(r)?));
    }
    if let Some(size) = scalar_size(value_type) {
        skip(r, size)?;
        return Ok(None);
    }
    match value_type {
        8 => {
            let len = read_u64(r)?;
            skip(r, i64::try_from(len)?)?;
        }
        9 => {
            let element = read_u32(r)?;
            let count = read_u64(r)?;
            if let Some(size) = scalar_size(element) {
                skip(
                    r,
                    size.checked_mul(i64::try_from(count)?)
                        .context("GGUF array size")?,
                )?;
            } else if element == 8 {
                for _ in 0..count {
                    let len = read_u64(r)?;
                    skip(r, i64::try_from(len)?)?;
                }
            } else {
                bail!("unsupported GGUF array element type {element}");
            }
        }
        other => bail!("unknown GGUF value type {other}"),
    }
    Ok(None)
}

fn read_shard(path: &Path, shard: usize, out: &mut HashMap<String, GgufTensor>) -> Result<()> {
    let mut r =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut magic = [0; 4];
    r.read_exact(&mut magic)?;
    ensure!(&magic == b"GGUF", "{} is not a GGUF file", path.display());
    let version = read_u32(&mut r)?;
    ensure!(version == 3, "{}: GGUF version {version}", path.display());
    let n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;

    let mut alignment = 32_u64;
    for _ in 0..n_kv {
        let key = read_string(&mut r)?;
        let value_type = read_u32(&mut r)?;
        if let Some(v) = skip_value(&mut r, value_type)?
            && key == "general.alignment"
        {
            ensure!(v > 0 && v.is_power_of_two(), "bad GGUF alignment {v}");
            alignment = u64::from(v);
        }
    }

    let mut found = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        ensure!(n_dims <= 8, "{name}: {n_dims} dimensions");
        let dims = (0..n_dims)
            .map(|_| read_u64(&mut r))
            .collect::<Result<Vec<_>>>()?;
        let ggml_type = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        found.push((name, dims, ggml_type, offset));
    }
    let header_end = r.stream_position()?;
    let data_start = header_end.div_ceil(alignment) * alignment;
    for (name, dims, ggml_type, offset) in found {
        out.insert(
            name,
            GgufTensor {
                shard,
                offset: data_start + offset,
                dims,
                ggml_type,
            },
        );
    }
    Ok(())
}

/// The sibling paths of a `name-00001-of-0000N.gguf` shard, or the file
/// itself when it is not split.
fn shard_paths(first: &Path) -> Result<Vec<PathBuf>> {
    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .context("GGUF path has no file name")?;
    let Some(stem) = name.strip_suffix(".gguf") else {
        bail!("{name} is not a .gguf file");
    };
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if parts.len() == 4
        && parts[1] == "of"
        && parts[0].len() == 5
        && parts[2].len() == 5
        && let Ok(total) = parts[0].parse::<usize>()
    {
        let dir = first.parent().unwrap_or_else(|| Path::new("."));
        return Ok((1..=total)
            .map(|i| dir.join(format!("{}-{i:05}-of-{total:05}.gguf", parts[3])))
            .collect());
    }
    Ok(vec![first.to_path_buf()])
}

impl GgufIndex {
    /// Reads every shard of the model whose first shard is `first`.
    pub fn open(first: &Path) -> Result<Self> {
        let shards = shard_paths(first)?;
        let mut tensors = HashMap::new();
        for (i, path) in shards.iter().enumerate() {
            read_shard(path, i, &mut tensors)?;
        }
        Ok(Self { shards, tensors })
    }
}

/// The `(layer, expert)` pairs of a Strata expert profile, hottest first.
pub fn read_expert_profile(path: &Path, layers: usize, experts: usize) -> Result<Vec<(u16, u16)>> {
    let blob = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_expert_profile(&blob, layers, experts)
        .with_context(|| format!("parse {}", path.display()))
}

fn parse_expert_profile(blob: &[u8], layers: usize, experts: usize) -> Result<Vec<(u16, u16)>> {
    const HEADER: usize = 24;
    ensure!(
        blob.len() >= HEADER && &blob[..4] == b"STRP",
        "not an expert profile"
    );
    let field = |i: usize| {
        u32::from_le_bytes(blob[4 + i * 4..8 + i * 4].try_into().expect("four bytes")) as usize
    };
    ensure!(field(0) == 1, "profile version {}", field(0));
    ensure!(
        (field(1), field(2)) == (layers, experts),
        "profile is {}x{}, the model is {layers}x{experts}",
        field(1),
        field(2)
    );
    let count = field(4);
    ensure!(count <= layers * experts, "profile ranks {count} pairs");
    ensure!(blob.len() >= HEADER + count * 4, "profile is truncated");
    blob[HEADER..HEADER + count * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pair| {
            let layer = u16::from_le_bytes([pair[0], pair[1]]);
            let expert = u16::from_le_bytes([pair[2], pair[3]]);
            ensure!(
                (layer as usize) < layers && (expert as usize) < experts,
                "pair ({layer}, {expert}) is out of range"
            );
            Ok((layer, expert))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// A GGUF with one of each key shape the skipper must cross.
    fn shard(alignment: u32, tensors: &[(&str, &[u64], u32, u64)]) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3_u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(4_u64.to_le_bytes());
        string(&mut b, "a.string");
        b.extend(8_u32.to_le_bytes());
        string(&mut b, "value");
        string(&mut b, "a.array");
        b.extend(9_u32.to_le_bytes());
        b.extend(8_u32.to_le_bytes());
        b.extend(2_u64.to_le_bytes());
        string(&mut b, "x");
        string(&mut b, "yz");
        string(&mut b, "a.numbers");
        b.extend(9_u32.to_le_bytes());
        b.extend(5_u32.to_le_bytes());
        b.extend(3_u64.to_le_bytes());
        b.extend([0_u8; 12]);
        string(&mut b, "general.alignment");
        b.extend(4_u32.to_le_bytes());
        b.extend(alignment.to_le_bytes());
        for (name, dims, ty, offset) in tensors {
            string(&mut b, name);
            b.extend((dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend(d.to_le_bytes());
            }
            b.extend(ty.to_le_bytes());
            b.extend(offset.to_le_bytes());
        }
        b
    }

    #[test]
    fn offsets_are_absolute_and_follow_the_alignment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        let bytes = shard(64, &[("t", &[2560, 640, 512], 21, 128)]);
        let data_start = (bytes.len() as u64).div_ceil(64) * 64;
        File::create(&path).unwrap().write_all(&bytes).unwrap();
        let index = GgufIndex::open(&path).unwrap();
        assert_eq!(
            index.tensors["t"],
            GgufTensor {
                shard: 0,
                offset: data_start + 128,
                dims: vec![2560, 640, 512],
                ggml_type: 21,
            }
        );
    }

    #[test]
    fn split_models_are_read_from_every_shard() {
        let dir = tempfile::tempdir().unwrap();
        for (i, name) in ["a", "b"].iter().enumerate() {
            let path = dir.path().join(format!("m-Q-{:05}-of-00002.gguf", i + 1));
            let bytes = shard(32, &[(name, &[1], 8, 0)]);
            File::create(path).unwrap().write_all(&bytes).unwrap();
        }
        let index = GgufIndex::open(&dir.path().join("m-Q-00001-of-00002.gguf")).unwrap();
        assert_eq!(index.shards.len(), 2);
        assert_eq!((index.tensors["a"].shard, index.tensors["b"].shard), (0, 1));
    }

    fn profile(version: u32, layers: u32, experts: u32, pairs: &[(u16, u16)]) -> Vec<u8> {
        let mut b = b"STRP".to_vec();
        for v in [
            version,
            layers,
            experts,
            layers * experts,
            pairs.len() as u32,
        ] {
            b.extend(v.to_le_bytes());
        }
        for (l, e) in pairs {
            b.extend(l.to_le_bytes());
            b.extend(e.to_le_bytes());
        }
        b
    }

    #[test]
    fn a_profile_keeps_its_rank_order() {
        let blob = profile(1, 2, 4, &[(1, 3), (0, 2)]);
        assert_eq!(
            parse_expert_profile(&blob, 2, 4).unwrap(),
            vec![(1, 3), (0, 2)]
        );
    }

    #[test]
    fn a_profile_for_another_shape_or_out_of_range_is_refused() {
        assert!(parse_expert_profile(&profile(1, 2, 4, &[]), 3, 4).is_err());
        assert!(parse_expert_profile(&profile(2, 2, 4, &[]), 2, 4).is_err());
        assert!(parse_expert_profile(&profile(1, 2, 4, &[(2, 0)]), 2, 4).is_err());
        let truncated = profile(1, 2, 4, &[(1, 1)]);
        assert!(parse_expert_profile(&truncated[..truncated.len() - 1], 2, 4).is_err());
    }

    #[test]
    fn a_file_that_is_not_gguf_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        File::create(&path).unwrap().write_all(b"NOPE0000").unwrap();
        assert!(GgufIndex::open(&path).is_err());
    }
}
