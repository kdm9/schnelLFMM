use std::time::Instant;

pub struct Timer {
    #[allow(dead_code)]
    label: &'static str,
    #[allow(dead_code)]
    start: Instant,
}

impl Timer {
    pub fn new(label: &'static str) -> Self {
        Self { label, start: Instant::now() }
    }

    pub fn finish(self) {
        {
            #![cfg(feature = "profiling")]
            let elapsed = self.start.elapsed();
            eprintln!("  [timer] {}: {:.3}s", self.label, elapsed.as_secs_f64());
        }
    }
}


