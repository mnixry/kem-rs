use std::time::Duration;

use criterion::{
    Criterion,
    measurement::{Measurement, WallTime},
};
use nix::sys::{
    resource::{UsageWho, getrusage},
    time::TimeValLike,
};

pub struct CPUTime;

impl CPUTime {
    fn get_time() -> Duration {
        let rc = getrusage(UsageWho::RUSAGE_SELF).expect("failed to get process CPU time");
        Duration::from_micros(rc.user_time().num_microseconds().cast_unsigned())
    }
}

impl Measurement for CPUTime {
    type Intermediate = Duration;
    type Value = Duration;

    fn start(&self) -> Self::Intermediate {
        Self::get_time()
    }

    fn end(&self, i: Self::Intermediate) -> Self::Value {
        let now = Self::get_time();
        now.saturating_sub(i)
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1.saturating_add(*v2)
    }

    fn zero(&self) -> Self::Value {
        Duration::ZERO
    }

    #[allow(clippy::cast_precision_loss)]
    fn to_f64(&self, value: &Self::Value) -> f64 {
        value.as_nanos() as f64
    }

    fn formatter(&self) -> &dyn criterion::measurement::ValueFormatter {
        WallTime.formatter()
    }
}

pub type CPUTimeConfig = Criterion<CPUTime>;
