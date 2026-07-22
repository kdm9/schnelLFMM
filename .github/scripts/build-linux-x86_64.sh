#!/usr/bin/env bash
set -euo pipefail

export PATH="/root/.cargo/bin:${PATH}"

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi

rustup set profile minimal
rustup toolchain install "${RUST_TOOLCHAIN}"
rustup default "${RUST_TOOLCHAIN}"

cargo build --verbose --release
cargo test --verbose --release
