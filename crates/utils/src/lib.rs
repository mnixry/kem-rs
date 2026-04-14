#![cfg_attr(coverage_nightly, feature(coverage_attribute))]
//! Utilities for benchmarking and profiling using [Criterion].

pub mod measure;
pub mod profiler;
pub use criterion;
use criterion::Criterion;

pub type CriterionConfig = Criterion<measure::UserTime>;

#[must_use]
pub fn criterion_config() -> CriterionConfig {
    criterion::Criterion::default()
        .with_profiler(profiler::PProfProfiler::new(1997))
        .with_measurement(measure::UserTime)
}

#[cfg(test)]
mod tests {
    #[test]
    fn criterion_config_builds() {
        let _ = super::criterion_config();
    }
}
