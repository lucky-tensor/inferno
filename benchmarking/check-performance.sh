#!/bin/bash

# Performance Regression Check Script
#
# This script runs Rust criterion benchmarks and compares them against previous runs
# to detect performance regressions. If any benchmark is slower than the configured
# threshold, the script fails with an error.
#
# Usage: ./check-performance.sh
#
# The script will:
# 1. Run cargo criterion with JSON output
# 2. Parse the results to find any performance regressions
# 3. Exit with error code 1 if regressions are found, 0 otherwise

# Exit on any error
set -e

# Configuration
# CI should automatically reject any one benchmark is more than 15% slower
REGRESSION_THRESHOLD_PERCENT=15.0  # Fail if benchmarks are this % slower
BENCHMARK_OUTPUT_FILE="benchmark.json"
PREVIOUS_BENCHMARKS_DIR="./.prior_bench"

# Convert percentage threshold to decimal for calculations
# Example: 10.0% becomes 0.10
convert_percent_to_decimal() {
    local percent=$1
    echo "scale=4; $percent / 100" | bc
}

# Run the benchmarks and save results to JSON file
run_benchmarks() {
    echo "🚀 Running benchmarks with JSON output..."
    echo "   Output file: $BENCHMARK_OUTPUT_FILE"
    echo "   Previous benchmarks: $PREVIOUS_BENCHMARKS_DIR"

    CRITERION_HOME="$PREVIOUS_BENCHMARKS_DIR" \
        cargo criterion \
        --message-format=json \
        --output-format=quiet \
        > "$BENCHMARK_OUTPUT_FILE"
}

# Parse benchmark results and find any performance regressions
find_regressions() {
    local threshold_decimal
    threshold_decimal=$(convert_percent_to_decimal "$REGRESSION_THRESHOLD_PERCENT")

    echo "🔍 Checking for performance regressions..."
    echo "   Threshold: ${REGRESSION_THRESHOLD_PERCENT}% (${threshold_decimal} decimal)"

    # Use jq to filter benchmark results:
    # - Only look at completed benchmarks with change data
    # - Find ones where the mean estimate is greater than our threshold
    # - Format as "benchmark_name: +XX%"
    jq -r \
        --argjson threshold "$threshold_decimal" \
        'select(.reason == "benchmark-complete" and has("change") and .change.mean.estimate > $threshold) |
         "\(.id): +\((.change.mean.estimate * 100) | round)%"' \
        "$BENCHMARK_OUTPUT_FILE" || echo ""
}

# Report results and exit with appropriate code
report_results() {
    local regressions="$1"

    if [[ -n "$regressions" ]]; then
        echo ""
        echo "❌ Performance regressions detected:"
        echo "$regressions"
        echo ""
        echo "::error::Performance regression detected! Some benchmarks are more than ${REGRESSION_THRESHOLD_PERCENT}% slower."
        exit 1
    else
        echo ""
        echo "✅ No performance regressions detected"
        echo "   All benchmarks are within the ${REGRESSION_THRESHOLD_PERCENT}% threshold."
        echo ""
        echo "::notice::All benchmarks passed performance regression check."
        exit 0
    fi
}

# Main execution
main() {
    run_benchmarks

    local regressions
    regressions=$(find_regressions)

    report_results "$regressions"
}

# Run the script
main "$@"
