//! A host's power schedule (#397).
//!
//! Solar-powered hosts leave the mesh on a daily schedule. The operator
//! lists the windows in which the host is powered; `/health` announces the
//! next window in which it is not, so cortex can stop sending new work
//! before the host goes and treat its absence as planned.
//!
//! ```toml
//! [availability]
//! # UTC. A window may wrap past midnight ("22:00-06:00").
//! online = ["06:30-18:45"]
//! # Take no new work this long before each dark window, so running work
//! # can finish. Longer than this host's longest job.
//! drain_secs = 900
//! ```
//!
//! With no `online` windows the host is always on and announces nothing.

use cortex_core::discovery::DarkWindow;
use serde::{Deserialize, Serialize};

const DAY: u64 = 24 * 60 * 60;

/// `[availability]` in neuron.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityConfig {
    /// Daily windows, in UTC, in which this host is powered:
    /// `"HH:MM-HH:MM"`. Empty means always on.
    #[serde(default)]
    pub online: Vec<String>,
    /// How long before each dark window the host stops taking new work,
    /// in seconds. It should exceed the host's longest job.
    #[serde(default = "default_drain_secs")]
    pub drain_secs: u64,
}

impl Default for AvailabilityConfig {
    fn default() -> Self {
        Self {
            online: Vec::new(),
            drain_secs: default_drain_secs(),
        }
    }
}

/// Fifteen minutes: a 600 s song on the slowest card that serves one.
fn default_drain_secs() -> u64 {
    900
}

/// Parsed online windows, as `[start, end)` seconds past UTC midnight.
/// A window that wraps past midnight is split in two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    online: Vec<(u64, u64)>,
    drain_secs: u64,
}

impl Schedule {
    /// `None` when the config lists no windows: an always-on host.
    pub fn from_config(cfg: &AvailabilityConfig) -> anyhow::Result<Option<Self>> {
        if cfg.online.is_empty() {
            return Ok(None);
        }
        let mut online = Vec::new();
        for w in &cfg.online {
            let (a, b) = w
                .split_once('-')
                .ok_or_else(|| anyhow::anyhow!("availability window '{w}' is not HH:MM-HH:MM"))?;
            let (start, end) = (time_of_day(a.trim(), w)?, time_of_day(b.trim(), w)?);
            anyhow::ensure!(start != end, "availability window '{w}' is empty");
            if start < end {
                online.push((start, end));
            } else {
                online.push((start, DAY));
                online.push((0, end));
            }
        }
        Ok(Some(Self {
            online,
            drain_secs: cfg.drain_secs,
        }))
    }

    /// The dark window in progress at `now`, or else the next one; `None`
    /// when the windows cover the whole day.
    pub fn next_dark(&self, now: u64) -> Option<DarkWindow> {
        // Lay the online windows out over yesterday, today and the next
        // two days; the gaps between them are the dark windows. Three days
        // ahead is enough because a dark window is shorter than a day.
        let midnight = now - now % DAY;
        let mut on: Vec<(u64, u64)> = (0..4)
            .flat_map(|d| {
                let base = midnight - DAY + d * DAY;
                self.online.iter().map(move |&(s, e)| (base + s, base + e))
            })
            .collect();
        on.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (s, e) in on {
            match merged.last_mut() {
                Some(last) if s <= last.1 => last.1 = last.1.max(e),
                _ => merged.push((s, e)),
            }
        }
        merged
            .windows(2)
            .map(|w| DarkWindow {
                drain_at: w[0].1.saturating_sub(self.drain_secs),
                starts_at: w[0].1,
                ends_at: w[1].0,
            })
            .find(|d| d.ends_at > now)
    }
}

fn time_of_day(hhmm: &str, window: &str) -> anyhow::Result<u64> {
    let bad = || anyhow::anyhow!("availability window '{window}' has a bad time '{hhmm}'");
    let (h, m) = hhmm.split_once(':').ok_or_else(bad)?;
    let (h, m): (u64, u64) = (h.parse().map_err(|_| bad())?, m.parse().map_err(|_| bad())?);
    // 24:00 is the end of the day, so a window can run to midnight.
    anyhow::ensure!(m < 60 && (h < 24 || (h == 24 && m == 0)), bad());
    Ok(h * 3600 + m * 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(windows: &[&str]) -> Schedule {
        Schedule::from_config(&AvailabilityConfig {
            online: windows.iter().map(|w| w.to_string()).collect(),
            drain_secs: 600,
        })
        .unwrap()
        .unwrap()
    }

    /// 2026-10-21 00:00 UTC.
    const MIDNIGHT: u64 = 1_792_540_800;
    const fn at(h: u64, m: u64) -> u64 {
        MIDNIGHT + h * 3600 + m * 60
    }

    #[test]
    fn a_day_host_announces_tonight_while_it_is_up() {
        let s = schedule(&["06:30-18:45"]);
        let dark = s.next_dark(at(12, 0)).unwrap();
        assert_eq!(dark.starts_at, at(18, 45));
        assert_eq!(dark.ends_at, at(24 + 6, 30));
        assert_eq!(dark.drain_at, at(18, 35));
        assert!(dark.takes_work_at(at(18, 34)));
        assert!(!dark.takes_work_at(at(18, 35)));
        assert!(dark.takes_work_at(at(24 + 6, 30)));
    }

    #[test]
    fn inside_the_dark_window_it_reports_the_one_in_progress() {
        let s = schedule(&["06:30-18:45"]);
        // Before dawn: the window that began yesterday evening.
        let dark = s.next_dark(at(3, 0)).unwrap();
        assert_eq!(dark.starts_at, at(18, 45) - DAY);
        assert_eq!(dark.ends_at, at(6, 30));
    }

    #[test]
    fn a_window_may_wrap_past_midnight() {
        let s = schedule(&["22:00-06:00"]);
        let dark = s.next_dark(at(23, 0)).unwrap();
        assert_eq!(
            (dark.starts_at, dark.ends_at),
            (at(24 + 6, 0), at(24 + 22, 0))
        );
    }

    #[test]
    fn overlapping_and_adjacent_windows_merge() {
        let s = schedule(&["06:00-12:00", "11:00-14:00", "14:00-18:00"]);
        let dark = s.next_dark(at(9, 0)).unwrap();
        assert_eq!((dark.starts_at, dark.ends_at), (at(18, 0), at(24 + 6, 0)));
    }

    #[test]
    fn a_whole_day_is_never_dark() {
        assert_eq!(schedule(&["00:00-24:00"]).next_dark(at(5, 0)), None);
    }

    #[test]
    fn no_windows_is_an_always_on_host() {
        assert_eq!(
            Schedule::from_config(&AvailabilityConfig::default()).unwrap(),
            None
        );
    }

    #[test]
    fn malformed_windows_are_config_errors() {
        for w in [
            "0630-1845",
            "06:30",
            "25:00-26:00",
            "06:60-07:00",
            "08:00-08:00",
        ] {
            let cfg = AvailabilityConfig {
                online: vec![w.into()],
                ..Default::default()
            };
            assert!(Schedule::from_config(&cfg).is_err(), "{w} accepted");
        }
    }
}
