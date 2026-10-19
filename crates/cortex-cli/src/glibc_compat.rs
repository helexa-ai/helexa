//! Replacements for libm symbols whose current version is newer than the
//! glibc cortex is packaged for. `build.rs` points the linker at these.

/// `log10f` computed through `log10`, which keeps its original symbol
/// version. Rounding the f64 result to f32 is at least as accurate as
/// libm's single-precision version.
#[unsafe(no_mangle)]
extern "C" fn __wrap_log10f(x: f32) -> f32 {
    (x as f64).log10() as f32
}

#[cfg(test)]
mod tests {
    use super::__wrap_log10f;

    #[test]
    fn matches_log10_including_edge_cases() {
        for x in [1e-30_f32, 0.5, 1.0, 2.0, 10.0, 1234.5, f32::MAX] {
            assert!(
                (__wrap_log10f(x) - x.log10()).abs() <= 1e-6 * x.log10().abs().max(1.0),
                "{x}"
            );
        }
        assert_eq!(__wrap_log10f(0.0), f32::NEG_INFINITY);
        assert!(__wrap_log10f(-1.0).is_nan());
        assert!(__wrap_log10f(f32::NAN).is_nan());
        assert_eq!(__wrap_log10f(f32::INFINITY), f32::INFINITY);
    }
}
