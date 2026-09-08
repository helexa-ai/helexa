//! Opt-in phase timing inside a model's forward pass.
//!
//! The request-level split (`PhaseTiming` on the wire) says whether a
//! decode step's time went into the forward, the sampler or the emit.
//! When the answer is "the forward", this says *where* in the forward —
//! embedding, a gather, the layer stack, the head.
//!
//! # Why this needs a device synchronise, and why it is off by default
//!
//! candle's CUDA ops are asynchronous on the stream. An `Instant` taken
//! around GPU work therefore measures how fast kernels were *queued*,
//! not how long they ran, and whichever phase happens to contain the
//! next real synchronisation point silently absorbs everyone else's
//! cost. The numbers look plausible and attribute the time to the wrong
//! place — a worse failure than no numbers at all.
//!
//! So each mark synchronises the device first. That is only affordable
//! when the phases being measured are much larger than a sync: at a few
//! tens of microseconds per sync, a handful of marks is noise against a
//! step measured in hundreds of milliseconds, and material against one
//! measured in single milliseconds. Since that ratio is a property of
//! how slow the model currently is rather than something safe to assume,
//! the probe stays off unless `NEURON_FORWARD_PHASES` is set.

use candle_core::Device;
use std::sync::OnceLock;
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NEURON_FORWARD_PHASES").is_ok())
}

/// Accumulates named phases within one forward pass.
///
/// Construct at the top of the forward, call [`mark`] at each boundary,
/// and [`emit`] at the end. Disabled, every method is a predictable
/// branch and nothing is allocated.
///
/// [`mark`]: PhaseProbe::mark
/// [`emit`]: PhaseProbe::emit
pub struct PhaseProbe {
    on: bool,
    device: Device,
    last: Instant,
    marks: Vec<(&'static str, f64)>,
}

impl PhaseProbe {
    pub fn new(device: &Device) -> Self {
        Self::with_enabled(device, enabled())
    }

    /// Constructor with the gate supplied explicitly.
    ///
    /// [`new`] reads the environment through a `OnceLock`, which a test
    /// cannot toggle — so without this the enabled path would be
    /// unreachable from a test, and a probe that only ever runs
    /// disabled is not evidence that it records anything.
    ///
    /// [`new`]: PhaseProbe::new
    pub fn with_enabled(device: &Device, on: bool) -> Self {
        // Start from a synchronised device, or the first phase inherits
        // whatever was still in flight when the forward was entered.
        if on {
            let _ = device.synchronize();
        }
        Self {
            on,
            device: device.clone(),
            last: Instant::now(),
            marks: Vec::new(),
        }
    }

    /// Close the phase that ended here and open the next one.
    pub fn mark(&mut self, name: &'static str) {
        if !self.on {
            return;
        }
        // A failed synchronise means the elapsed time below is not the
        // phase's own. Record it as such rather than reporting a number
        // that looks like a measurement.
        let synced = self.device.synchronize().is_ok();
        let ms = self.last.elapsed().as_secs_f64() * 1e3;
        self.marks
            .push((if synced { name } else { "UNSYNCED" }, ms));
        self.last = Instant::now();
    }

    /// The phases recorded so far, in order.
    #[cfg(test)]
    pub fn marks(&self) -> &[(&'static str, f64)] {
        &self.marks
    }

    /// Emit one line carrying every phase of this forward.
    pub fn emit(self, rows: usize) {
        if !self.on || self.marks.is_empty() {
            return;
        }
        let total: f64 = self.marks.iter().map(|(_, ms)| ms).sum();
        let detail = self
            .marks
            .iter()
            .map(|(n, ms)| format!("{n}={ms:.3}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::info!(rows, total_ms = total, %detail, "forward phases");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disabled probe must record nothing at all — not zero-valued
    /// phases. Zeros in a log read as "this phase was free", which is
    /// the same failure the wire-side `Option` exists to prevent.
    #[test]
    fn a_disabled_probe_records_nothing() {
        let mut p = PhaseProbe::with_enabled(&Device::Cpu, false);
        p.mark("embed");
        p.mark("layers");
        assert!(
            p.marks().is_empty(),
            "disabled probe recorded {:?}",
            p.marks()
        );
    }

    /// An enabled probe records one phase per mark, in call order, and
    /// each phase measures only the span since the previous mark rather
    /// than since the start.
    #[test]
    fn phases_are_recorded_in_order_and_do_not_accumulate() {
        let mut p = PhaseProbe::with_enabled(&Device::Cpu, true);
        p.mark("embed");
        std::thread::sleep(std::time::Duration::from_millis(20));
        p.mark("layers");
        p.mark("head");

        let names: Vec<_> = p.marks().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["embed", "layers", "head"]);

        let ms = |i: usize| p.marks()[i].1;
        // "layers" contains the sleep; the marks either side must not.
        // A probe that timed from the start instead of from the previous
        // mark would make "head" >= "layers", which is the mistake this
        // pins.
        assert!(
            ms(1) >= 15.0,
            "layers should carry the sleep, got {:?}",
            p.marks()
        );
        assert!(
            ms(2) < 10.0,
            "head must not inherit it, got {:?}",
            p.marks()
        );
        assert!(
            ms(0) < 10.0,
            "embed must not inherit it, got {:?}",
            p.marks()
        );
    }
}
