use std::{fs::File, io::Write, os::raw::c_int, path::Path, process::Command};

use criterion::profiler::Profiler;
use pprof::{ProfilerGuard, protos::Message};

pub struct PProfProfiler<'a> {
    frequency: c_int,
    active_profiler: Option<ProfilerGuard<'a>>,
}

impl PProfProfiler<'_> {
    #[must_use]
    pub const fn new(frequency: c_int) -> Self {
        Self {
            frequency,
            active_profiler: None,
        }
    }
}

impl Profiler for PProfProfiler<'_> {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        self.active_profiler =
            Some(ProfilerGuard::new(self.frequency).expect("failed to start profiler"));
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        std::fs::create_dir_all(benchmark_dir).expect("failed to create benchmark directory");
        let report = self.active_profiler.take().map_or_else(
            || unreachable!("profiler not started"),
            |profiler| profiler.report().build().expect("failed to build report"),
        );
        File::create(benchmark_dir.join("flamegraph.svg")).map_or_else(
            |e| panic!("failed to create flamegraph file: {e}"),
            |file| report.flamegraph(file).expect("failed to write flamegraph"),
        );
        File::create(benchmark_dir.join("profile.pb")).map_or_else(
            |e| panic!("failed to create profile file: {e}"),
            |mut file| {
                let mut content = Vec::new();
                report
                    .pprof()
                    .expect("failed to get pprof profile")
                    .encode(&mut content)
                    .expect("failed to encode pprof profile");
                file.write_all(&content)
                    .expect("failed to write pprof profile");
            },
        );
        let Ok(current_exe) = std::env::current_exe() else {
            return;
        };
        match Command::new("pprof")
            .arg("-dot")
            .arg("-output")
            .arg(benchmark_dir.join("pprof.dot"))
            .arg(current_exe)
            .arg(benchmark_dir.join("profile.pb"))
            .output()
        {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                eprintln!("pprof failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("pprof not found, skipping pprof.svg generation");
            }
            Err(e) => panic!("failed to run pprof: {e}"),
        }
    }
}
