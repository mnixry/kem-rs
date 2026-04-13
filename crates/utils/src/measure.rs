use std::time::Duration;

use criterion::measurement::{Measurement, WallTime};
use nix::sys::{
    resource::{UsageWho, getrusage},
    time::{TimeVal, TimeValLike},
};

pub struct UserTime;

impl UserTime {
    fn get_time() -> TimeVal {
        let rc = getrusage(UsageWho::RUSAGE_SELF).expect("failed to get process CPU time");
        rc.user_time()
    }
}

impl Measurement for UserTime {
    type Intermediate = TimeVal;
    type Value = Duration;

    fn start(&self) -> Self::Intermediate {
        Self::get_time()
    }

    #[allow(clippy::cast_sign_loss)]
    fn end(&self, i: Self::Intermediate) -> Self::Value {
        let elapsed = Self::get_time() - i;
        Duration::from_nanos(
            elapsed
                .num_nanoseconds()
                .try_into()
                .expect("time gone backwards"),
        )
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
