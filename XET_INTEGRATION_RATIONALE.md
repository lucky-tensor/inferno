# Xet Integration for Model Downloads

## Why We Need Xet Integration

Currently, our `download` subcommand uses Git LFS (Large File Storage) to download AI model files from Hugging Face. While this works, Hugging Face now recommends using **xet** as the preferred method for downloading models due to several advantages:

### Benefits of Xet over Git LFS

1. **Performance**: Xet provides faster download speeds and better handling of large files
2. **Reliability**: More robust handling of network interruptions and resume capabilities
3. **Efficiency**: Better deduplication and compression for model files
4. **Official Support**: Recommended by Hugging Face as the modern approach
5. **Future-proof**: Git LFS may eventually be deprecated for model downloads

### Current State

- We use Git LFS protocol for model downloads
- Downloads work but may be slower and less reliable
- No option to use alternative download methods

### Proposed Solution

Use the native Rust `hf-hub` crate to provide xet-enabled downloads as an optional method:

- **Default behavior**: Continue using Git LFS (no breaking changes)
- **Opt-in xet**: Users can explicitly request native HuggingFace Hub downloads with automatic xet backend
- **No Python dependencies**: Fully self-contained Rust implementation
- **Graceful fallback**: If hf-hub fails, fall back to Git LFS
- **Automatic optimization**: hf-hub automatically uses xet backend when available

## Implementation Overview

1. Add `hf-hub` dependency to the CLI crate with tokio features
2. Extend the download subcommand with a `--use-xet` flag
3. Implement native Rust HuggingFace Hub downloads with automatic xet backend
4. Maintain backward compatibility with existing Git LFS flow
5. Provide graceful fallback from hf-hub to Git LFS on any failures
6. Add comprehensive error handling and user feedback
7. Update documentation and help text

This approach ensures we can leverage xet's benefits through a fully self-contained Rust implementation, requiring no Python dependencies while maintaining stability for existing users.