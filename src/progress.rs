use indicatif::{ProgressBar, ProgressStyle};

/// Create a progress bar for a streaming pass over SNP chunks.
///
/// Returns a visible bar when `show` is true, or a hidden no-op bar otherwise.
/// The bar is written to stderr and uses a compact format:
///   `  label [=====>     ] 42/100 chunks (12.3 chunks/s, ETA 5s)`
pub fn make_progress_bar(n_chunks: u64, label: &str, show: bool) -> ProgressBar {
    if !show || n_chunks == 0 {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(n_chunks);
    pb.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:30}] {pos}/{len} chunks ({per_sec}, ETA {eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(label.to_string());
    pb
}
