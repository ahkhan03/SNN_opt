#!/usr/bin/env bash
# Build the lean CPU baseline. Usage: build_cpu_baseline.sh [output-dir]
# Same script for x86 (tests) and aarch64 (KV260, Ubuntu 22.04, gcc 11.4).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${1:-${script_dir}/work}"
mkdir -p "${output_dir}"

g++ -O3 -march=native -mtune=native -funroll-loops -fopenmp -std=c++17 \
  "${script_dir}/cpu_baseline.cpp" -o "${output_dir}/cpu_baseline"

sha256sum "${output_dir}/cpu_baseline"
