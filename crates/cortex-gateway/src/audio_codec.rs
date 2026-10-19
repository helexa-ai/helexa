//! Transcoding stored audio for download.
//!
//! Renders are stored as 16-bit FLAC, the lossless master. MP3 is offered
//! as a download format because some players and users expect it; it is
//! encoded from the master on request and cached beside it.

use anyhow::{Context, bail};
use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, InterleavedPcm, Quality};

/// Frames handed to the MP3 encoder per call, which bounds its output
/// buffer for long tracks.
const CHUNK_FRAMES: usize = 1152 * 64;

/// Transcode 16-bit FLAC (mono or stereo) to MP3 at 320 kbps CBR, the
/// highest rate MP3 has: it is a copy of a lossless master, so it should
/// lose no more than the format forces it to.
pub fn flac_to_mp3(flac: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut reader =
        claxon::FlacReader::new(std::io::Cursor::new(flac)).context("reading FLAC header")?;
    let info = reader.streaminfo();
    if info.bits_per_sample != 16 {
        bail!("{}-bit FLAC is not supported", info.bits_per_sample);
    }
    let channels = info.channels as usize;
    if !(1..=2).contains(&channels) {
        bail!("{channels}-channel FLAC is not supported");
    }
    let mut pcm = Vec::with_capacity(info.samples.unwrap_or(0) as usize * channels);
    for s in reader.samples() {
        pcm.push(s.context("decoding FLAC")? as i16);
    }
    // LAME's interleaved input is stereo; duplicate a mono channel.
    if channels == 1 {
        pcm = pcm.iter().flat_map(|&s| [s, s]).collect();
    }

    let mut builder = Builder::new().context("creating MP3 encoder")?;
    builder
        .set_num_channels(2)
        .map_err(|e| anyhow::anyhow!("MP3 channels: {e}"))?;
    builder
        .set_sample_rate(info.sample_rate)
        .map_err(|e| anyhow::anyhow!("MP3 sample rate: {e}"))?;
    builder
        .set_brate(Bitrate::Kbps320)
        .map_err(|e| anyhow::anyhow!("MP3 bitrate: {e}"))?;
    builder
        .set_quality(Quality::NearBest)
        .map_err(|e| anyhow::anyhow!("MP3 quality: {e}"))?;
    let mut encoder = builder
        .build()
        .map_err(|e| anyhow::anyhow!("MP3 encoder: {e}"))?;

    let mut out = Vec::new();
    for chunk in pcm.chunks(CHUNK_FRAMES * 2) {
        out.reserve(mp3lame_encoder::max_required_buffer_size(chunk.len() / 2));
        encoder
            .encode_to_vec(InterleavedPcm(chunk), &mut out)
            .map_err(|e| anyhow::anyhow!("MP3 encode: {e}"))?;
    }
    out.reserve(7200);
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(|e| anyhow::anyhow!("MP3 flush: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flac(channels: usize, frames: usize) -> Vec<u8> {
        use flacenc::component::BitRepr;
        use flacenc::error::Verify;
        let samples: Vec<i32> = (0..frames * channels)
            .map(|i| ((i as f32 * 0.03).sin() * 12_000.0) as i32)
            .collect();
        let config = flacenc::config::Encoder::default().into_verified().unwrap();
        let source = flacenc::source::MemSource::from_samples(&samples, channels, 16, 48_000);
        let stream =
            flacenc::encode_with_fixed_block_size(&config, source, config.block_size).unwrap();
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink).unwrap();
        sink.into_inner()
    }

    /// The first MPEG audio frame header: sync word, then MPEG-1 Layer III
    /// at 320 kbps / 48 kHz.
    fn first_frame(mp3: &[u8]) -> [u8; 4] {
        let at = mp3
            .windows(2)
            .position(|w| w[0] == 0xFF && w[1] & 0xE0 == 0xE0)
            .expect("no MPEG frame sync");
        mp3[at..at + 4].try_into().unwrap()
    }

    #[test]
    fn stereo_flac_becomes_320k_mp3() {
        let mp3 = flac_to_mp3(&flac(2, 48_000 * 2)).unwrap();
        let h = first_frame(&mp3);
        assert_eq!(h[1] & 0x1E, 0x1A, "not MPEG-1 Layer III: {h:02x?}");
        assert_eq!(h[2] >> 4, 0xE, "not 320 kbps: {h:02x?}");
        assert_eq!((h[2] >> 2) & 0x3, 0x1, "not 48 kHz: {h:02x?}");
        // Two seconds at 320 kbps is 80 kB, give or take a frame.
        assert!((70_000..90_000).contains(&mp3.len()), "{} bytes", mp3.len());
    }

    #[test]
    fn mono_flac_is_accepted() {
        let mp3 = flac_to_mp3(&flac(1, 48_000)).unwrap();
        assert!(mp3.len() > 30_000, "{} bytes", mp3.len());
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(flac_to_mp3(b"RIFF....WAVEfmt ").is_err());
        assert!(flac_to_mp3(&[]).is_err());
    }
}
