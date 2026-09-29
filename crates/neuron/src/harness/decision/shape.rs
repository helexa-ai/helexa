//! Per-option logits → calibrated, typed answers.
//!
//! The reference computes this in numpy on the float32 logits the model
//! returns, and publishes each probability rounded to 4 decimal places.
//! Reproducing the rounded values exactly means reproducing the float32
//! arithmetic, in numpy's order: division by a float32 temperature, a
//! max-subtracted `exp`, and numpy's *pairwise* summation (not a running
//! sum) for every reduction. The expected score is the exception: numpy
//! promotes `arange(k) * p` to float64.

use super::question::QType;
use super::sequence::Prepared;
use super::{DECISION_MODEL_NAME, DecisionConfig};
use cortex_core::decisions::{
    CollapsedOptions, DecisionAction, DecisionAnswer, DecisionUsage, OrderedMap, SystemOneResponse,
};

/// The model's output for one [`super::Row`].
#[derive(Debug, Clone, PartialEq)]
pub struct RowOutput {
    /// The scorer's raw logit at each option marker, before calibration.
    /// Entries past the row's marker count (batch padding) are ignored.
    pub logits: Vec<f32>,
    /// The act head's probability of acting: `softmax(act_logits)[0]`.
    pub act_probability: f32,
}

/// Build the answers and usage for a prepared request from the model's
/// per-row outputs (`outputs[i]` belongs to `prepared.rows[i]`).
///
/// `lang` selects per-language calibration where the checkpoint has it:
/// the language the request was declared or detected as.
pub fn shape(
    prepared: &Prepared,
    outputs: &[RowOutput],
    cfg: &DecisionConfig,
    lang: Option<&str>,
) -> (OrderedMap<DecisionAnswer>, DecisionUsage) {
    assert_eq!(
        outputs.len(),
        prepared.rows.len(),
        "one model output per prepared row"
    );
    let temps = cfg.temperatures_for(lang);
    let mut answers = Vec::with_capacity(outputs.len());
    let mut collapsed = Vec::new();
    for ((question, row), (out, stats)) in prepared
        .questions
        .iter()
        .zip(&prepared.rows)
        .zip(outputs.iter().zip(&prepared.stats))
    {
        let k = row.markers.len();
        let p = calibrate(
            &out.logits[..k],
            temps.for_question(question.kind as usize, k),
        );
        let answer_confidence = round4(
            p.iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .clamp(0.0, 1.0) as f64,
        );
        let action = DecisionAction {
            act_probability: round4(out.act_probability as f64),
        };
        let probabilities = |key: &dyn Fn(usize) -> String| {
            OrderedMap(
                p.iter()
                    .enumerate()
                    .map(|(i, &v)| (key(i), round4(v as f64)))
                    .collect(),
            )
        };
        let answer = match question.kind {
            QType::Choice => DecisionAnswer::Choice {
                choice: question.labels[argmax(&p)].clone(),
                probabilities: probabilities(&|i| {
                    question.labels[i]
                        .json_key()
                        .expect("choice labels are scalars")
                }),
                confidence: round4(entropy_confidence(&p) as f64),
                answer_confidence,
                action,
            },
            QType::Score => {
                // int64 * float32 promotes to float64 in numpy.
                let expected: Vec<f64> = p
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| i as f64 * v as f64)
                    .collect();
                DecisionAnswer::Score {
                    score: round4(pairwise_sum(&expected)),
                    legend: OrderedMap(
                        question
                            .labels
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i.to_string(), c.clone()))
                            .collect(),
                    ),
                    probabilities: probabilities(&|i| i.to_string()),
                    confidence: round4(entropy_confidence(&p) as f64),
                    answer_confidence,
                    action,
                }
            }
            QType::Noul => {
                let yes = p[1] as f64;
                DecisionAnswer::Noul {
                    noul: round4(yes),
                    confidence: round4(yes.max(1.0 - yes)),
                    answer_confidence,
                    action,
                }
            }
        };
        answers.push((question.id.clone(), answer));
        if stats.options_distinct < stats.options {
            collapsed.push((
                question.id.clone(),
                CollapsedOptions {
                    total: stats.options,
                    distinct: stats.options_distinct,
                    tokens_per_option: stats.tokens_per_option,
                },
            ));
        }
    }
    let usage = DecisionUsage {
        input_tokens: prepared.input_tokens,
        output_tokens: 0,
        options: (!collapsed.is_empty()).then_some(OrderedMap(collapsed)),
    };
    (OrderedMap(answers), usage)
}

/// Assemble the full response once routing is known.
pub fn response(
    answers: OrderedMap<DecisionAnswer>,
    usage: DecisionUsage,
    routing: Option<serde_json::Value>,
) -> SystemOneResponse {
    SystemOneResponse {
        model: DECISION_MODEL_NAME.to_string(),
        answers,
        usage,
        routing,
    }
}

/// Temperature-scaled softmax, as the reference evaluates it in numpy:
/// `z = logits / t`, `e = exp(z - z.max())`, `p = e / e.sum()`.
///
/// numpy's SIMD `expf` and libm's disagree by one ulp on about a third of
/// inputs. That is invisible after the 4-decimal rounding on every
/// reference fixture, and far below the engine's own distance from the
/// reference's logits, so it is not reproduced; the steps either side of
/// the `exp` are.
fn calibrate(logits: &[f32], t: f64) -> Vec<f32> {
    let z = scale(logits, t);
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
    normalize(&e)
}

/// `logits / t` in float32: numpy casts the Python-float temperature to the
/// array's dtype rather than promoting the array.
fn scale(logits: &[f32], t: f64) -> Vec<f32> {
    let t = t as f32;
    logits.iter().map(|&l| l / t).collect()
}

/// `e / e.sum()` with numpy's pairwise float32 sum.
fn normalize(e: &[f32]) -> Vec<f32> {
    let sum = pairwise_sum(e);
    e.iter().map(|&v| v / sum).collect()
}

/// `1 - H(p) / log(k)`, clipped to [0, 1]: how concentrated the whole
/// distribution is. Not calibrated; see `answer_confidence` for that.
fn entropy_confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let terms: Vec<f32> = p.iter().map(|&v| v * v.clamp(1e-12, 1.0).ln()).collect();
    let ent = -pairwise_sum(&terms);
    (1.0 - ent / (k as f64).ln() as f32).clamp(0.0, 1.0)
}

/// The first index of the largest value, as `argmax` picks it.
fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// Python's `round(x, 4)`: correctly rounded, ties to even on the exact
/// binary value. Rust's fixed-precision formatting rounds the same way.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("formatted float parses")
}

/// numpy's pairwise summation over a contiguous array, which the rounded
/// outputs depend on: a plain loop below 8 elements, eight interleaved
/// accumulators up to 128, and recursive halving (at a multiple of 8)
/// beyond.
fn pairwise_sum<T>(a: &[T]) -> T
where
    T: Copy + std::ops::Add<Output = T> + Default,
{
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = T::default();
        for &v in a {
            res = res + v;
        }
        res
    } else if n <= BLOCK {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] = r[j] + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_differs_from_a_running_sum_where_numpy_does() {
        // 1e8 swamps every 1.0 added to it one at a time in f32; pairwise
        // grouping keeps them. `int(np.float32 array.sum())` of this array is 100000008.
        let mut a = vec![1.0f32; 16];
        a[0] = 1e8;
        let running = a.iter().fold(0.0f32, |s, &v| s + v);
        assert_eq!(running, 1e8);
        assert_eq!(pairwise_sum(&a), 100_000_008.0);
    }

    /// Rows of `testdata/laya/numpy_softmax.txt`: the reference's
    /// calibration run in numpy on random float32 logits, with the bits of
    /// every intermediate. Bit equality pins the evaluation order — a
    /// running sum instead of numpy's pairwise one, or dividing by the
    /// temperature in float64, changes low bits that the 4-decimal rounding
    /// usually hides.
    fn numpy_rows() -> Vec<(f64, [Vec<f32>; 4])> {
        include_str!("../testdata/laya/numpy_softmax.txt")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let mut f = l.split('|');
                let t: f64 = f.next().unwrap().parse().unwrap();
                let bits = |s: &str| -> Vec<f32> {
                    s.split(',')
                        .map(|h| f32::from_bits(u32::from_str_radix(&h[2..], 16).unwrap()))
                        .collect()
                };
                let v: Vec<Vec<f32>> = f.map(bits).collect();
                (t, [v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()])
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn calibration_matches_numpy_either_side_of_exp() {
        let rows = numpy_rows();
        assert_eq!(rows.len(), 4);
        for (t, [logits, z, e, p]) in rows {
            assert_eq!(
                bits(&scale(&logits, t)),
                bits(&z),
                "k={} t={t}: z",
                logits.len()
            );
            assert_eq!(
                bits(&normalize(&e)),
                bits(&p),
                "k={} t={t}: p",
                logits.len()
            );
        }
    }

    #[test]
    fn round4_ties_to_even_on_the_binary_value() {
        assert_eq!(round4(0.03125), 0.0312);
        assert_eq!(round4(1.00005), 1.0001); // 1.00005 is just above the tie in binary
        assert_eq!(round4(0.99995), 1.0);
    }
}
