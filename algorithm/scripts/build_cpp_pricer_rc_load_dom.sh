#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/pricing/native/cpp_pricing_core_rc_load_dom.cpp"
OUT="$ROOT/src/pricing/native/libcpp_pricing_core_rc_load_dom.so"

g++ -O3 -std=c++17 -march=native -funroll-loops -shared -fPIC "$SRC" -o "$OUT"
echo "built: $OUT"
