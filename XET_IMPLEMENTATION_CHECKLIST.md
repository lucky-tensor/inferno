# Xet Integration Implementation Checklist

## Phase 1: Dependencies and Setup

- [x] Research xet-core API and usage patterns - **DISCOVERY: xet-core is Python-only**
- [x] Evaluate implementation approach - **DECISION: Use subprocess to call Python huggingface_hub**
- [ ] Add subprocess and Python integration dependencies
- [ ] Create feature flag for xet integration (optional, for conditional compilation)
- [ ] Update root `Cargo.toml` if needed for workspace dependencies

### Implementation Strategy Update
After researching xet-core, discovered that:
- xet-core is primarily Python-based, not available as a direct Rust crate
- Intended to be used through the `huggingface_hub` Python package
- The Rust components are internal to HuggingFace Hub's Python implementation

**New Approach**: Implement xet downloads by calling the Python `huggingface_hub` library as a subprocess, which automatically uses xet when available (huggingface_hub >= 0.32.0).

## Phase 2: Command Interface

- [x] Extend download subcommand with `--use-xet` flag in CLI argument parsing
- [x] Update help text and documentation for the new flag
- [x] Ensure backward compatibility - default behavior unchanged
- [ ] Add validation for xet-specific parameters if any

## Phase 3: Core Implementation

- [x] ~~Create new module `xet_downloader.rs` in inference crate~~ **Implemented in model_downloader.rs**
- [x] Implement `XetDownloader` functionality using Python subprocess calls
- [x] Add Python environment detection and validation
- [x] Implement `huggingface_hub` Python script for downloading
- [x] Add error handling specific to subprocess and Python operations
- [x] Add home directory and cache setup for xet/huggingface_hub
- [ ] Implement progress reporting for xet downloads (parse Python output)
- [ ] Add timeout and retry logic for robustness
- [ ] Create abstraction layer for download methods (trait-based approach)

## Phase 4: Integration

- [x] Modify existing download logic to support multiple backends
- [x] Implement fallback mechanism (xet -> git lfs if xet fails)
- [x] Handle authentication for xet (HF token support)
- [ ] Update model discovery and metadata handling for xet
- [ ] Ensure file integrity verification works with both methods

## Phase 5: Testing

- [ ] Write unit tests for `XetDownloader`
- [ ] Create integration tests comparing xet vs git lfs downloads
- [ ] Add tests for fallback behavior
- [ ] Test error handling and edge cases
- [ ] Performance benchmark tests (optional but recommended)
- [ ] Test with various model sizes and types

## Phase 6: Quality Assurance

- [ ] Run `cargo clippy` and fix all warnings
- [ ] Run `cargo fmt` to ensure consistent formatting
- [ ] Ensure all tests pass with `cargo test`
- [ ] Check test coverage meets project standards
- [ ] Run any existing CI/CD checks locally

## Phase 7: Documentation

- [ ] Update inline code documentation
- [ ] Update CLI help text and man pages if applicable
- [ ] Add examples of using `--use-xet` flag
- [ ] Document performance differences and when to use each method

## Phase 8: Validation

- [ ] Test with real Hugging Face models
- [ ] Verify download integrity with both methods
- [ ] Performance comparison testing
- [ ] Test on different platforms (if applicable)
- [ ] Verify no regressions in existing functionality

## Definition of Done

- [ ] All tests pass
- [ ] Code passes linting (clippy) with no warnings
- [ ] Code is properly formatted (rustfmt)
- [ ] Feature works as specified in rationale document
- [ ] Backward compatibility maintained
- [ ] Documentation is complete and accurate
- [ ] Code review completed (when ready)

## Notes

- Start with a minimal viable implementation focusing on core functionality
- Prioritize error handling and user experience
- Consider logging/telemetry for debugging download issues
- Keep performance metrics to help users choose the best method