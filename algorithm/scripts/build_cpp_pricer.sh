#!/usr/bin/env zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/pricing/native/cpp_pricing_core.cpp"
OUT="$ROOT/src/pricing/native/libcpp_pricing_core.so"

g++ -O3 -std=c++17 -shared -fPIC "$SRC" -o "$OUT"
echo "built: $OUT"
