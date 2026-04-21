#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//! Utilities for benchmarking and profiling using [Criterion].

pub mod measure;
pub mod profiler;

pub use criterion;
use criterion::Criterion;

pub type CriterionConfig = Criterion<measure::CPUTime>;

#[must_use]
pub fn criterion_config() -> CriterionConfig {
    if let Some(core_ids) = core_affinity::get_core_ids()
        && let Some(first_core) = core_ids.first()
        && core_affinity::set_for_current(*first_core)
    {
        eprintln!("Benchmark has been pinned to core {}", first_core.id);
    }

    let mut config = Criterion::default().with_measurement(measure::CPUTime);
    if let Some(freq) = std::env::var("PPROF_FREQUENCY")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&f| f > 0)
    {
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
