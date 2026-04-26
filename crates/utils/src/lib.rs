#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//! Utilities for benchmarking and profiling using [Criterion].

pub mod measure;
pub mod profiler;

pub use criterion;
use criterion::Criterion;

pub type CriterionConfig = Criterion<measure::CPUTime>;

fn parse_env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
}

#[must_use]
pub fn criterion_config() -> CriterionConfig {
    if let pinned_core = parse_env::<usize>("PINNED_CORE").unwrap_or_default()
        && let Some(core_id) =
            core_affinity::get_core_ids().and_then(|ids| ids.get(pinned_core).copied())
        && core_affinity::set_for_current(core_id)
    {
        eprintln!("Benchmark has been pinned to core {}", core_id.id);
    }

    let mut config = Criterion::default().with_measurement(measure::CPUTime);
    if let Some(freq) = parse_env("PPROF_FREQUENCY").filter(|&f| f > 0) {
        config = config.with_profiler(profiler::PProfProfiler::new(freq));
    }
    config
}

#[cfg(test)]
mod tests {
    #[test]
    fn criterion_config_builds() {
        let _ = super::criterion_config();
    }
}
