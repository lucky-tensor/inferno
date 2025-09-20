//! Path utilities for consistent file and directory management across Inferno
//!
//! This module provides standardized paths for models, cache, and other
//! directories used throughout the Inferno application stack.

use std::env;
use std::path::{Path, PathBuf};

/// Default models directory name relative to user home
const MODELS_DIR_NAME: &str = "models";

/// Default cache directory name relative to user home
const CACHE_DIR_NAME: &str = ".cache/inferno";

/// Get the default models directory path
///
/// Returns `$HOME/models` on Unix systems, or equivalent on other platforms.
/// This is the standard location where Inferno looks for and stores AI models.
///
/// # Examples
///
/// ```
/// use inferno_shared::default_models_dir;
///
/// let models_path = default_models_dir();
/// println!("Models directory: {}", models_path.display());
/// ```
pub fn default_models_dir() -> PathBuf {
    get_home_dir().join(MODELS_DIR_NAME)
}

/// Get the default cache directory path
///
/// Returns `$HOME/.cache/inferno` on Unix systems, or equivalent on other platforms.
/// This is where Inferno stores temporary files, downloaded models cache, etc.
pub fn default_cache_dir() -> PathBuf {
    get_home_dir().join(CACHE_DIR_NAME)
}

/// Get the user's home directory
///
/// Falls back to current directory if HOME cannot be determined.
fn get_home_dir() -> PathBuf {
    if let Ok(home) = env::var("HOME") {
        PathBuf::from(home)
    } else if let Some(home_dir) = dirs::home_dir() {
        home_dir
    } else {
        // Fallback to current directory if we can't determine home
        PathBuf::from(".")
    }
}

/// Expand a path that starts with `~` to use the user's home directory
///
/// # Examples
///
/// ```
/// use inferno_shared::expand_home_dir;
///
/// let path = expand_home_dir("~/models/my-model");
/// // Returns: /home/user/models/my-model
/// ```
pub fn expand_home_dir<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if let Some(path_str) = path.to_str() {
        if let Some(stripped) = path_str.strip_prefix("~/") {
            let home = get_home_dir();
            return home.join(stripped);
        } else if path_str == "~" {
            return get_home_dir();
        }
    }
    path.to_path_buf()
}

/// Resolve a models directory path, expanding ~ if needed
///
/// This is the canonical way to resolve model paths throughout Inferno.
/// It handles:
/// - Relative paths (returned as-is)
/// - Absolute paths (returned as-is)
/// - Home directory expansion for paths starting with `~`
///
/// # Examples
///
/// ```
/// use inferno_shared::resolve_models_path;
///
/// let path1 = resolve_models_path("~/models");          // -> /home/user/models
/// let path2 = resolve_models_path("/opt/models");       // -> /opt/models
/// let path3 = resolve_models_path("./models");          // -> ./models
/// ```
pub fn resolve_models_path<P: AsRef<Path>>(path: P) -> PathBuf {
    expand_home_dir(path)
}

/// Get the default models directory as a string
///
/// Convenience function that returns the default models directory path as a String.
/// Useful for CLI default values and configuration.
pub fn default_models_dir_string() -> String {
    default_models_dir().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_models_dir() {
        let models_dir = default_models_dir();
        assert!(models_dir.to_string_lossy().ends_with("models"));
    }

    #[test]
    fn test_expand_home_dir() {
        // Test tilde expansion
        let expanded = expand_home_dir("~/test");
        assert!(!expanded.to_string_lossy().starts_with("~"));
        assert!(expanded.to_string_lossy().ends_with("test"));

        // Test absolute path (should be unchanged)
        let absolute = expand_home_dir("/absolute/path");
        assert_eq!(absolute, PathBuf::from("/absolute/path"));

        // Test relative path (should be unchanged)
        let relative = expand_home_dir("relative/path");
        assert_eq!(relative, PathBuf::from("relative/path"));
    }

    #[test]
    fn test_resolve_models_path() {
        let path1 = resolve_models_path("~/models");
        assert!(!path1.to_string_lossy().starts_with("~"));
        assert!(path1.to_string_lossy().ends_with("models"));

        let path2 = resolve_models_path("/opt/models");
        assert_eq!(path2, PathBuf::from("/opt/models"));
    }
}
