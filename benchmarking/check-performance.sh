#!/bin/bash

# Performance Regression Check Script
# Runs benchmarks and fails if any are >5% slower than previous run

set -e

REGRESSION_THRESHOLD=10.0
BENCHMARK_JSON="benchmark.json"

echo "🚀 Running benchmarks with JSON output..."
CRITERION_HOME=./.prior_bench cargo criterion config_from_env --message-format=json --output-format=quiet > "$BENCHMARK_JSON"

echo "🔍 Checking for performance regressions (threshold: ${REGRESSION_THRESHOLD}%)..."

# Use jq to find regressions >5% slower
regressions=$(jq -r 'select(.reason == "benchmark-complete" and has("change") and .change.mean.estimate > 0.05) | "\(.id): +\((.change.mean.estimate * 100) | round)%"' "$BENCHMARK_JSON" || echo "")

if [[ -n "$regressions" ]]; then
    echo ""
    echo "❌ Performance regressions detected:"
    echo "$regressions"
    echo ""
    echo "::error::Performance regression detected! Benchmarks are >5% slower than previous run."
    exit 1
else
    echo ""
    echo "✅ No performance regressions detected"
    echo ""
    echo "::notice::All benchmarks passed performance regression check."
    exit 0
fi
