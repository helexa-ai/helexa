//! Keep the cortex binary installable on the oldest glibc we package for.
//!
//! glibc 2.43 gave `log10f` a new symbol version, so a binary linked on a
//! newer host references `log10f@GLIBC_2.43` and will not install on a
//! system with an older glibc. The only caller is LAME (the MP3 encoder
//! behind `/content?format=mp3`). Routing its calls through
//! `__wrap_log10f` (see `glibc_compat.rs`) removes the reference.
fn main() {
    println!("cargo:rustc-link-arg-bins=-Wl,--wrap=log10f");
}
