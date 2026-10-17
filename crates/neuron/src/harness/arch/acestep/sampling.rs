//! Flow matching sampler for ACE-Step diffusion model.
//!
//! Implements the Euler ODE solver for rectified flow / flow matching,
//! with Adaptive Prompt Guidance (APG) support.

use candle_core::{Result, Tensor};

/// All unique timesteps from shift=1,2,3 with fix_nfe=8.
/// Turbo models were distilled on these exact values — custom timesteps are
/// mapped to the nearest entry in this table.
const VALID_TIMESTEPS: [f64; 20] = [
    1.0,
    0.9545454545454546,
    0.9333333333333333,
    0.9,
    0.875,
    0.8571428571428571,
    0.8333333333333334,
    0.7692307692307693,
    0.75,
    0.6666666666666666,
    0.6428571428571429,
    0.625,
    0.5454545454545454,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.2222222222222222,
    0.125,
];

/// Generate a linear timestep schedule from 1.0 to 0.0.
///
/// Optionally applies a time shift transformation:
/// `t_shifted = shift * t / (1 + (shift - 1) * t)`
pub fn get_schedule(num_steps: usize, shift: f64) -> Vec<f64> {
    if num_steps == 0 {
        return vec![0.0];
    }
    let mut schedule = Vec::with_capacity(num_steps + 1);
    for i in 0..=num_steps {
        let t = 1.0 - (i as f64 / num_steps as f64);
        let t = if (shift - 1.0).abs() > 1e-6 {
            shift * t / (1.0 + (shift - 1.0) * t)
        } else {
            t
        };
        schedule.push(t);
    }
    schedule
}

/// Pre-defined discrete timestep schedules for turbo models.
/// These are the only valid schedules — turbo models were distilled on these
/// exact timesteps and will produce poor results with continuous schedules.
///
/// Each schedule has `fix_nfe` steps (default 8) plus an implicit final t=0.
pub fn get_turbo_schedule(shift: f64) -> Vec<f64> {
    if (shift - 1.0).abs() < 0.5 {
        // shift ≈ 1
        vec![1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    } else if (shift - 2.0).abs() < 0.5 {
        // shift ≈ 2
        vec![
            1.0,
            0.9333333333333333,
            0.8571428571428571,
            0.7692307692307693,
            0.6666666666666666,
            0.5454545454545454,
            0.4,
            0.2222222222222222,
        ]
    } else {
        // shift ≈ 3 (default for turbo)
        vec![
            1.0,
            0.9545454545454546,
            0.9,
            0.8333333333333334,
            0.75,
            0.6428571428571429,
            0.5,
            0.3,
        ]
    }
}

/// Build a custom turbo schedule from user-provided timestep values.
///
/// Each value is mapped to the nearest entry in `VALID_TIMESTEPS`. Trailing
/// zeros are stripped, and the resulting schedule must have 1–20 entries.
pub fn get_custom_turbo_schedule(timesteps: &[f64]) -> Result<Vec<f64>> {
    let end = timesteps
        .iter()
        .rposition(|&t| t != 0.0)
        .map(|i| i + 1)
        .unwrap_or(0);
    let timesteps = &timesteps[..end];
    if timesteps.is_empty() || timesteps.len() > 20 {
        candle_core::bail!(
            "custom timesteps must have 1-20 non-zero values, got {}",
            timesteps.len()
        );
    }
    for &t in timesteps {
        if !t.is_finite() {
            candle_core::bail!("custom timesteps contain non-finite value: {t}");
        }
    }
    Ok(timesteps
        .iter()
        .map(|&t| {
            *VALID_TIMESTEPS
                .iter()
                .min_by(|&&a, &&b| {
                    (a - t)
                        .abs()
                        .partial_cmp(&(b - t).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        })
        .collect())
}

/// Get x0 (clean sample) from noisy sample and velocity prediction.
/// Flow matching: x_t = t * noise + (1 - t) * x_0, v = noise - x_0
/// Therefore: x_0 = x_t - v * t
pub fn get_x0_from_noise(xt: &Tensor, vt: &Tensor, t: f64) -> Result<Tensor> {
    xt - (vt * t)?
}

/// Euler ODE step for flow matching.
///
/// Given current sample `x_t` and velocity prediction `v_t`:
/// `x_{t-dt} = x_t - v_t * dt`
pub fn euler_step(x: &Tensor, v: &Tensor, dt: f64) -> Result<Tensor> {
    x - (v * dt)?
}

/// SDE step: predict clean sample, then re-add noise at the next timestep.
///
/// `x_0 = get_x0_from_noise(x_t, v_t, t_curr)`
/// `x_{t_next} = (1 - t_next) * x_0 + t_next * noise`
pub fn sde_step(xt: &Tensor, vt: &Tensor, t_curr: f64, t_next: f64) -> Result<Tensor> {
    let x0 = get_x0_from_noise(xt, vt, t_curr)?;
    let noise = Tensor::randn_like(&x0, 0., 1.)?;
    let clean_part = (&x0 * (1.0 - t_next))?;
    let noise_part = (noise * t_next)?;
    clean_part + noise_part
}

/// Inference method for the denoising loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferMethod {
    /// Ordinary Differential Equation — deterministic Euler integration.
    Ode,
    /// Stochastic Differential Equation — predicts x0 then re-noises.
    Sde,
}

/// Momentum buffer for APG (Adaptive Prompt Guidance).
pub struct MomentumBuffer {
    momentum: f64,
    running_average: Option<Tensor>,
}

impl MomentumBuffer {
    pub fn new(momentum: f64) -> Self {
        Self {
            momentum,
            running_average: None,
        }
    }

    /// Update running average: `running_avg = update_value + momentum * running_avg`
    pub fn update(&mut self, update_value: &Tensor) -> Result<()> {
        let new_average = match &self.running_average {
            Some(ra) => (ra * self.momentum)?,
            None => Tensor::zeros_like(update_value)?,
        };
        self.running_average = Some((update_value + new_average)?);
        Ok(())
    }

    pub fn get(&self) -> Option<&Tensor> {
        self.running_average.as_ref()
    }
}

/// Project v0 onto v1 and decompose into parallel and orthogonal components.
///
/// Operates along `dim` (typically dim=1 for sequence dimension).
/// Uses F64 precision internally for numerical stability (matching Python).
/// On Metal, computation is moved to CPU to avoid MPS numeric issues.
fn project(v0: &Tensor, v1: &Tensor, dim: usize) -> Result<(Tensor, Tensor)> {
    let orig_device = v0.device().clone();
    let orig_dtype = v0.dtype();

    // Use CPU for Metal (MPS numeric issues); stay on device for CUDA.
    let compute_device = if orig_device.is_metal() {
        candle_core::Device::Cpu
    } else {
        orig_device.clone()
    };
    let v0 = v0
        .to_device(&compute_device)?
        .to_dtype(candle_core::DType::F64)?;
    let v1 = v1
        .to_device(&compute_device)?
        .to_dtype(candle_core::DType::F64)?;

    // L2 normalize v1 along dim
    let v1_norm = v1.sqr()?.sum_keepdim(dim)?.sqrt()?;
    let v1_normalized = v1.broadcast_div(&(v1_norm + 1e-8)?)?;

    // Project: v0_parallel = (v0 · v1_hat) * v1_hat
    let dot = (&v0 * &v1_normalized)?.sum_keepdim(dim)?;
    let v0_parallel = dot.broadcast_mul(&v1_normalized)?;
    let v0_orthogonal = (&v0 - &v0_parallel)?;

    Ok((
        v0_parallel.to_dtype(orig_dtype)?.to_device(&orig_device)?,
        v0_orthogonal
            .to_dtype(orig_dtype)?
            .to_device(&orig_device)?,
    ))
}

/// Adaptive Prompt Guidance (APG).
///
/// Decomposes the cond-uncond difference into parallel/orthogonal components
/// relative to the conditional prediction, applies norm thresholding, and
/// uses momentum for temporal smoothing.
///
/// Reference: ACE-Step's `apg_forward` in `apg_guidance.py`.
pub fn apg_forward(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    guidance_scale: f64,
    momentum_buffer: &mut MomentumBuffer,
    norm_threshold: f64,
) -> Result<Tensor> {
    // dims=[1] in Python — operates along the sequence dimension
    let dim = 1usize;
    let orig_device = pred_cond.device().clone();
    let orig_dtype = pred_cond.dtype();

    // On Metal, move to CPU for numerical stability (Python does this for MPS).
    // On CUDA/CPU, stay on the original device to avoid transfer overhead.
    let compute_device = if orig_device.is_metal() {
        candle_core::Device::Cpu
    } else {
        orig_device.clone()
    };
    let pred_cond_cpu = pred_cond.to_device(&compute_device)?;
    let pred_uncond_cpu = pred_uncond.to_device(&compute_device)?;

    let diff = (&pred_cond_cpu - &pred_uncond_cpu)?;

    // Apply momentum smoothing
    momentum_buffer.update(&diff)?;
    let diff = momentum_buffer.get().unwrap().clone();

    // Norm thresholding: clamp L2 norm along dim to prevent oversaturation
    let diff = if norm_threshold > 0.0 {
        let diff_norm = diff.sqr()?.sum_keepdim(dim)?.sqrt()?;
        let ones = Tensor::ones_like(&diff_norm)?;
        let scale = (ones * norm_threshold)?.broadcast_div(&(&diff_norm + 1e-8)?)?;
        let scale = scale.minimum(&Tensor::ones_like(&scale)?)?;
        diff.broadcast_mul(&scale)?
    } else {
        diff
    };

    // Decompose into parallel and orthogonal components
    let (_diff_parallel, diff_orthogonal) = project(&diff, &pred_cond_cpu, dim)?;

    // eta=0.0 by default: use only orthogonal component
    let result = (pred_cond_cpu + diff_orthogonal * (guidance_scale - 1.0))?;
    result.to_dtype(orig_dtype)?.to_device(&orig_device)
}

/// Differential Correction in Wavelet domain ("double" mode, Haar), which
/// upstream applies by default after every turbo step: split the latent and
/// the predicted clean sample into low and high bands along time, and push
/// each band of the latent away from the clean sample's. The low band's
/// strength decays with `t`, the high band's grows as `t → 0`.
#[derive(Debug, Clone, Copy)]
pub struct Dcw {
    pub low_scaler: f64,
    pub high_scaler: f64,
}

impl Default for Dcw {
    /// Upstream's defaults for turbo (`dcw_scaler`, `dcw_high_scaler`).
    fn default() -> Self {
        Self {
            low_scaler: 0.05,
            high_scaler: 0.02,
        }
    }
}

impl Dcw {
    /// Correct `x_next` (after the sampler step at `t_curr`) against
    /// `denoised = x_t - v·t_curr`. Shapes are `(B, T, C)`.
    pub fn apply(&self, x_next: &Tensor, denoised: &Tensor, t_curr: f64) -> Result<Tensor> {
        let low = t_curr * self.low_scaler;
        let high = (1.0 - t_curr) * self.high_scaler;
        if low == 0.0 && high == 0.0 {
            return Ok(x_next.clone());
        }
        let t_out = x_next.dim(1)?;
        let (xl, xh) = haar_dwt_t(x_next)?;
        let (yl, yh) = haar_dwt_t(denoised)?;
        let xl = (&xl + ((&xl - &yl)? * low)?)?;
        let xh = (&xh + ((&xh - &yh)? * high)?)?;
        haar_idwt_t(&xl, &xh, t_out)
    }
}

/// One-level Haar DWT along dim 1 of `(B, T, C)`, zero-padding an odd `T`
/// as `pytorch_wavelets` does in `mode="zero"`. Returns `(low, high)`, each
/// `(B, ceil(T/2), C)`.
fn haar_dwt_t(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, t, c) = x.dims3()?;
    let x = if t % 2 == 1 {
        let pad = Tensor::zeros((b, 1, c), x.dtype(), x.device())?;
        Tensor::cat(&[x, &pad], 1)?
    } else {
        x.clone()
    };
    let half = x.dim(1)? / 2;
    let pairs = x.reshape((b, half, 2, c))?;
    let even = pairs.narrow(2, 0, 1)?.squeeze(2)?;
    let odd = pairs.narrow(2, 1, 1)?.squeeze(2)?;
    let k = std::f64::consts::FRAC_1_SQRT_2;
    Ok((((&even + &odd)? * k)?, ((&even - &odd)? * k)?))
}

/// Inverse of [`haar_dwt_t`], trimmed back to `t_out` frames.
fn haar_idwt_t(low: &Tensor, high: &Tensor, t_out: usize) -> Result<Tensor> {
    let (b, half, c) = low.dims3()?;
    let k = std::f64::consts::FRAC_1_SQRT_2;
    let even = ((low + high)? * k)?;
    let odd = ((low - high)? * k)?;
    let x = Tensor::stack(&[&even, &odd], 2)?.reshape((b, half * 2, c))?;
    x.narrow(1, 0, t_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_custom_turbo_schedule_basic() -> Result<()> {
        // Exact valid timesteps should pass through unchanged
        let ts = vec![1.0, 0.875, 0.75, 0.5, 0.25];
        let result = get_custom_turbo_schedule(&ts)?;
        assert_eq!(result, ts);
        Ok(())
    }

    #[test]
    fn test_get_custom_turbo_schedule_nearest_mapping() -> Result<()> {
        // Values should be mapped to nearest valid timestep
        let ts = vec![0.99, 0.51, 0.13];
        let result = get_custom_turbo_schedule(&ts)?;
        assert_eq!(result, vec![1.0, 0.5, 0.125]);
        Ok(())
    }

    #[test]
    fn test_get_custom_turbo_schedule_strips_trailing_zeros() -> Result<()> {
        let ts = vec![1.0, 0.5, 0.0, 0.0];
        let result = get_custom_turbo_schedule(&ts)?;
        assert_eq!(result.len(), 2);
        Ok(())
    }

    #[test]
    fn test_get_custom_turbo_schedule_empty_after_strip() {
        let ts = vec![0.0, 0.0];
        assert!(get_custom_turbo_schedule(&ts).is_err());
    }

    #[test]
    fn test_get_custom_turbo_schedule_too_long() {
        let ts = vec![1.0; 21];
        assert!(get_custom_turbo_schedule(&ts).is_err());
    }

    #[test]
    fn test_get_custom_turbo_schedule_single() -> Result<()> {
        let ts = vec![0.87];
        let result = get_custom_turbo_schedule(&ts)?;
        assert_eq!(result, vec![0.875]);
        Ok(())
    }

    #[test]
    fn test_turbo_schedule_shift1_matches_custom() -> Result<()> {
        // The shift=1 hardcoded schedule should match custom input of same values
        let hardcoded = get_turbo_schedule(1.0);
        let custom = get_custom_turbo_schedule(&hardcoded)?;
        assert_eq!(hardcoded, custom);
        Ok(())
    }

    #[test]
    fn test_turbo_schedule_shift3_matches_custom() -> Result<()> {
        let hardcoded = get_turbo_schedule(3.0);
        let custom = get_custom_turbo_schedule(&hardcoded)?;
        assert_eq!(hardcoded, custom);
        Ok(())
    }

    #[test]
    fn haar_round_trips_and_dcw_matches_the_reference_formula() {
        use candle_core::Device;
        let d = Device::Cpu;
        // Odd T exercises the zero pad and the trim.
        let x = Tensor::arange(0f32, 21.0, &d)
            .unwrap()
            .reshape((1, 7, 3))
            .unwrap();
        let (l, h) = super::haar_dwt_t(&x).unwrap();
        assert_eq!(l.dims3().unwrap(), (1, 4, 3));
        let back = super::haar_idwt_t(&l, &h, 7).unwrap();
        let diff = (back - &x)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "{diff}");

        // Against the band formula directly, for a 2-frame signal:
        // L = (a+b)/√2, H = (a-b)/√2; corrected bands, then inverted.
        let xn = Tensor::new(&[[[3f32], [1.0]]], &d).unwrap();
        let dn = Tensor::new(&[[[1f32], [1.0]]], &d).unwrap();
        let t = 0.5;
        let out = Dcw::default()
            .apply(&xn, &dn, t)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let (xl, xh, yl, yh) = (4.0 * r, 2.0 * r, 2.0 * r, 0.0);
        let (s_lo, s_hi) = (t * 0.05, (1.0 - t) * 0.02);
        let (cl, ch) = (xl + s_lo * (xl - yl), xh + s_hi * (xh - yh));
        let want = [((cl + ch) * r) as f32, ((cl - ch) * r) as f32];
        assert!(
            (out[0] - want[0]).abs() < 1e-5 && (out[1] - want[1]).abs() < 1e-5,
            "{out:?} vs {want:?}"
        );
    }
}
