use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-changed=build.rs");

    // Only build C++/CUDA components if CUDA feature is enabled
    if !cfg!(feature = "cuda") {
        println!("cargo:warning=Skipping C++/CUDA build - CUDA feature not enabled");
        return;
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Detect CUDA installation
    let cuda_root = detect_cuda_installation();
    if cuda_root.is_none() {
        println!("cargo:warning=CUDA not found - falling back to CPU-only mode");
        return;
    }
    let cuda_root = cuda_root.unwrap();

    println!(
        "cargo:rustc-link-search=native={}/lib64",
        cuda_root.display()
    );
    println!("cargo:rustc-link-search=native={}/lib", cuda_root.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");
    println!("cargo:rustc-link-lib=cusparse");

    // Set up CMake build
    let cpp_dir = Path::new(&manifest_dir).join("cpp");
    let build_dir = Path::new(&out_dir).join("cpp_build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Configure CMake
    let mut cmake_config = cmake::Config::new(&cpp_dir);
    cmake_config
        .out_dir(&build_dir)
        .profile("Release")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CUDA_TOOLKIT_ROOT_DIR", &cuda_root)
        .define(
            "CMAKE_CUDA_COMPILER",
            find_nvcc(&cuda_root).unwrap_or_else(|| {
                panic!("nvcc not found in CUDA installation");
            }),
        );

    // Build with CMake
    let cmake_target = cmake_config.build();

    // Link the built library
    println!(
        "cargo:rustc-link-search=native={}/lib",
        cmake_target.display()
    );
    println!("cargo:rustc-link-lib=inferno_vllm_cpp");

    // Generate bindings if bindgen is available
    generate_bindings(&cpp_dir, &out_dir, &cuda_root);

    println!("cargo:rustc-env=VLLM_CPP_BUILD_DIR={}", build_dir.display());
}

fn detect_cuda_installation() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(cuda_path);
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let path = PathBuf::from(cuda_home);
        if path.exists() {
            return Some(path);
        }
    }

    // Check common installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/nvidia-cuda-toolkit",
    ];

    for path in &common_paths {
        let cuda_path = PathBuf::from(path);
        if cuda_path.exists() {
            return Some(cuda_path);
        }
    }

    // Try to find via nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi").output() {
        if output.status.success() {
            // nvidia-smi exists, likely CUDA is installed
            // Try default path
            let default_path = PathBuf::from("/usr/local/cuda");
            if default_path.exists() {
                return Some(default_path);
            }
        }
    }

    None
}

fn find_nvcc(cuda_root: &Path) -> Option<String> {
    let nvcc_path = cuda_root.join("bin").join("nvcc");
    if nvcc_path.exists() {
        nvcc_path.to_str().map(String::from)
    } else {
        None
    }
}

fn generate_bindings(_cpp_dir: &Path, _out_dir: &str, _cuda_root: &Path) {
    #[cfg(feature = "cuda")]
    {
        let header_path = _cpp_dir.join("include").join("vllm_wrapper.hpp");

        if !header_path.exists() {
            println!(
                "cargo:warning=Header file not found: {}",
                header_path.display()
            );
            return;
        }

        let bindings = bindgen::Builder::default()
            .header(header_path.to_str().unwrap())
            .clang_arg(format!("-I{}/include", _cpp_dir.display()))
            .clang_arg(format!("-I{}/include", _cuda_root.display()))
            .clang_arg("-std=c++17")
            .allowlist_function("vllm_.*")
            .allowlist_type("VLLM.*")
            .allowlist_var("VLLM_.*")
            .rustified_enum(".*")
            .derive_default(true)
            .derive_debug(true)
            .derive_copy(true)
            .derive_clone(true)
            .generate_comments(true)
            .layout_tests(false) // Disable layout tests for faster builds
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(_out_dir).join("vllm_bindings.rs");
        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");
    }
}

fn _check_cuda_version(cuda_root: &Path) -> Option<(u32, u32)> {
    let nvcc_path = cuda_root.join("bin").join("nvcc");
    if !nvcc_path.exists() {
        return None;
    }

    let output = Command::new(&nvcc_path).arg("--version").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let version_output = String::from_utf8(output.stdout).ok()?;

    // Parse version from nvcc output
    // Example: "Cuda compilation tools, release 11.8, V11.8.89"
    for line in version_output.lines() {
        if line.contains("release") {
            let parts: Vec<&str> = line.split(',').collect();
            for part in parts {
                if part.trim().starts_with("release") {
                    let version_str = part.trim().strip_prefix("release ")?;
                    let version_parts: Vec<&str> = version_str.split('.').collect();
                    if version_parts.len() >= 2 {
                        let major: u32 = version_parts[0].parse().ok()?;
                        let minor: u32 = version_parts[1].parse().ok()?;
                        return Some((major, minor));
                    }
                }
            }
        }
    }

    None
}
