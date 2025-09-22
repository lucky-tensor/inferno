#!/bin/bash

# Inferno Development Environment Setup Script
# This script installs all system dependencies needed for compiling and optimizing Inferno
# including CUDA drivers, LLVM tools, and performance analysis tools

set -e

# Configuration
NVIDIA_DRIVER_VERSION="535"
CUDA_VERSION_MAJOR="13"
CUDA_VERSION_MINOR="0"
CUDA_VERSION="${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"
CUDA_VERSION_FULL="${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
UBUNTU_VERSION="2204"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo
    echo -e "${BLUE}🔧 $1${NC}"
    echo "================================"
}

# OS detection
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS_NAME="$NAME"
            OS_VERSION="$VERSION_ID"
            log_info "Detected OS: $OS_NAME $OS_VERSION"
        else
            log_error "Cannot detect Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_error "macOS is not currently supported for full development setup"
        exit 1
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if running as root
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi

    # Check if user has sudo privileges
    if ! sudo -n true 2>/dev/null; then
        log_error "This script requires sudo privileges. Please ensure you can run sudo commands."
        exit 1
    fi
}

# Update system packages
update_system() {
    log_section "Updating System Packages"

    log_info "Updating package lists..."
    sudo apt-get update

    log_info "Upgrading existing packages..."
    sudo apt-get upgrade -y

    log_info "Installing essential build tools..."
    sudo apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        curl \
        wget \
        git \
        unzip \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common
}

# Install Rust toolchain
install_rust() {
    log_section "Installing Rust Toolchain"

    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        log_info "Rust is already installed: version $RUST_VERSION"
    else
        log_info "Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

        # Source the cargo environment
        source "$HOME/.cargo/env"
    fi

    # Ensure we have the latest stable toolchain
    log_info "Updating Rust toolchain..."
    rustup update stable
    rustup default stable

    # Install useful components
    log_info "Installing Rust components..."
    rustup component add clippy rustfmt

    # Verify installation
    log_info "Rust installation complete:"
    rustc --version
    cargo --version
}

# Install LLVM tools for PGO and BOLT optimization
install_llvm_tools() {
    log_section "Installing LLVM Tools for Optimization"

    log_info "Installing LLVM tools..."
    sudo apt-get install -y \
        llvm \
        llvm-dev \
        clang \
        clang-dev \
        libc++1 \
        libc++-dev

    # Verify LLVM tools
    if command -v llvm-profdata &> /dev/null; then
        LLVM_VERSION=$(llvm-profdata --version | head -1)
        log_info "LLVM profdata installed: $LLVM_VERSION"
    else
        log_error "llvm-profdata not found after installation"
    fi

    if command -v llvm-bolt &> /dev/null; then
        BOLT_VERSION=$(llvm-bolt --version | head -1)
        log_info "LLVM BOLT installed: $BOLT_VERSION"
    else
        log_warn "llvm-bolt not available (may need newer LLVM version)"
        log_info "BOLT optimization will not be available"
    fi
}

# Install performance analysis tools
install_perf_tools() {
    log_section "Installing Performance Analysis Tools"

    log_info "Installing perf and related tools..."
    sudo apt-get install -y \
        linux-perf \
        linux-tools-common \
        linux-tools-generic \
        linux-tools-$(uname -r) \
        perf-tools-unstable \
        sysstat \
        htop \
        iotop \
        nethogs

    # Install hyperfine for benchmarking
    if ! command -v hyperfine &> /dev/null; then
        log_info "Installing hyperfine..."
        wget -q https://github.com/sharkdp/hyperfine/releases/download/v1.18.0/hyperfine_1.18.0_amd64.deb
        sudo dpkg -i hyperfine_1.18.0_amd64.deb
        rm hyperfine_1.18.0_amd64.deb
    fi

    # Verify perf installation
    if command -v perf &> /dev/null; then
        log_info "Perf tools installed successfully"
        perf --version || log_warn "Perf installed but may need kernel module configuration"
    else
        log_error "Perf tools installation failed"
    fi

    if command -v hyperfine &> /dev/null; then
        log_info "Hyperfine benchmarking tool installed"
    else
        log_warn "Hyperfine installation failed, benchmarking may not work optimally"
    fi
}

# Install ML and inference dependencies
install_ml_dependencies() {
    log_section "Installing ML and Inference Dependencies"

    log_info "Installing Python and ML tools..."
    sudo apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        libssl-dev \
        libffi-dev \
        libbz2-dev \
        liblzma-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev

    log_info "Installing linear algebra libraries..."
    sudo apt-get install -y \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libopenblas-dev \
        gfortran

    log_info "Installing additional development libraries..."
    sudo apt-get install -y \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libgtk2.0-dev \
        libcanberra-gtk-module \
        libcanberra-gtk3-module

    # Install safetensors tools if available
    pip3 install --user safetensors huggingface-hub tokenizers || log_warn "Python ML packages installation failed"
}

# Install NVIDIA drivers and CUDA
install_cuda() {
    log_section "Installing NVIDIA CUDA Support"

    # Check if user wants to skip CUDA installation
    if [ "${SKIP_CUDA:-}" = "true" ]; then
        log_info "Skipping CUDA installation (SKIP_CUDA=true)"
        return 0
    fi

    # Check if CUDA is already installed
    if command -v nvcc &> /dev/null && nvidia-smi &> /dev/null; then
        INSTALLED_CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        log_info "CUDA $INSTALLED_CUDA_VERSION is already installed"
        if [ "$INSTALLED_CUDA_VERSION" == "$CUDA_VERSION_FULL" ]; then
            log_info "CUDA version matches expected version $CUDA_VERSION_FULL"
            return 0
        else
            log_warn "CUDA version mismatch. Expected: $CUDA_VERSION_FULL, Found: $INSTALLED_CUDA_VERSION"
            read -p "Do you want to reinstall CUDA $CUDA_VERSION_FULL? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Keeping existing CUDA installation"
                return 0
            fi
        fi
    fi

    log_info "Installing NVIDIA drivers and CUDA $CUDA_VERSION_FULL..."
    log_info "This will use the existing setup-nvidia.sh script"

    # Check if setup-nvidia.sh exists
    if [ -f "./scripts/setup-nvidia.sh" ]; then
        log_info "Running NVIDIA setup script..."
        bash ./scripts/setup-nvidia.sh
    else
        log_error "setup-nvidia.sh not found. Please run this script from the project root directory."
        log_info "You can manually install CUDA using: sudo apt install nvidia-cuda-toolkit"
        return 1
    fi

    log_info "CUDA installation completed"
}

# Configure system for performance optimization
configure_system() {
    log_section "Configuring System for Performance Optimization"

    # Set up perf permissions
    log_info "Configuring perf permissions..."
    echo 'kernel.perf_event_paranoid = 1' | sudo tee -a /etc/sysctl.conf > /dev/null
    echo 'kernel.kptr_restrict = 0' | sudo tee -a /etc/sysctl.conf > /dev/null
    sudo sysctl -p || log_warn "Could not reload sysctl configuration"

    # Add user to performance group if it exists
    if getent group perf-users > /dev/null 2>&1; then
        sudo usermod -a -G perf-users $USER
        log_info "Added $USER to perf-users group"
    fi

    # Set CPU governor to performance for benchmarking (optional)
    log_info "CPU governor configuration (for optimal benchmarking):"
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        current_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        log_info "Current CPU governor: $current_governor"
        log_info "For best benchmark results, consider setting to 'performance':"
        log_info "  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
    else
        log_info "CPU frequency scaling not available"
    fi
}

# Install additional development tools
install_dev_tools() {
    log_section "Installing Additional Development Tools"

    log_info "Installing debugging and profiling tools..."
    sudo apt-get install -y \
        gdb \
        valgrind \
        strace \
        ltrace \
        lsof \
        tcpdump \
        wireshark-common \
        tree \
        jq \
        tmux \
        screen

    log_info "Installing container tools..."
    sudo apt-get install -y \
        docker.io \
        docker-compose

    # Add user to docker group
    sudo usermod -a -G docker $USER
    log_info "Added $USER to docker group (logout/login required for effect)"

    log_info "Installing additional utilities..."
    sudo apt-get install -y \
        fd-find \
        ripgrep \
        bat \
        exa || log_warn "Some modern CLI tools unavailable in this Ubuntu version"
}

# Verify installation
verify_installation() {
    log_section "Verifying Installation"

    # Check Rust
    if command -v rustc &> /dev/null && command -v cargo &> /dev/null; then
        log_info "✅ Rust: $(rustc --version)"
    else
        log_error "❌ Rust installation verification failed"
    fi

    # Check LLVM tools
    if command -v llvm-profdata &> /dev/null; then
        log_info "✅ LLVM PGO tools available"
    else
        log_error "❌ LLVM PGO tools not available"
    fi

    if command -v llvm-bolt &> /dev/null; then
        log_info "✅ LLVM BOLT available"
    else
        log_warn "⚠️  LLVM BOLT not available (BOLT optimizations disabled)"
    fi

    # Check performance tools
    if command -v perf &> /dev/null; then
        log_info "✅ Perf tools available"
    else
        log_error "❌ Perf tools not available"
    fi

    if command -v hyperfine &> /dev/null; then
        log_info "✅ Hyperfine benchmarking available"
    else
        log_warn "⚠️  Hyperfine not available"
    fi

    # Check CUDA (optional)
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        log_info "✅ CUDA $CUDA_VER available"
    else
        log_warn "⚠️  CUDA not available (GPU acceleration disabled)"
    fi

    # Check if nvidia-smi works
    if nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        log_info "✅ NVIDIA GPU(s): $GPU_COUNT detected"
    else
        log_warn "⚠️  No NVIDIA GPUs detected or driver not working"
    fi

    log_info "Development environment setup verification complete!"
}

# Show post-installation instructions
show_post_install() {
    log_section "Post-Installation Instructions"

    echo -e "${GREEN}🎉 Development environment setup completed!${NC}"
    echo
    echo "📋 Next steps:"
    echo "  1. Restart your terminal or run: source ~/.cargo/env"
    echo "  2. Test the build: cargo build --release"
    echo "  3. Run PGO optimization: ./scripts/build-pgo.sh"
    echo "  4. Run PGO+BOLT optimization: ./scripts/build-pgo.sh --bolt"
    echo
    echo "🔧 Environment variables (add to ~/.bashrc or ~/.zshrc):"
    if command -v nvcc &> /dev/null; then
        echo "  export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION_FULL}"
        echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
    fi
    echo "  export PATH=~/.cargo/bin:\$PATH"
    echo
    echo "⚡ Performance optimization:"
    echo "  - For benchmarking, set CPU governor to performance mode"
    echo "  - You may need to logout/login for group changes to take effect"
    echo
    echo "📚 Documentation:"
    echo "  - Compilation optimization: ./docs/compilation-optimization.md"
    echo "  - PGO+BOLT script usage: ./scripts/build-pgo.sh --help"
    echo
}

# Main execution
main() {
    log_section "Inferno Development Environment Setup"

    log_info "This script will install all dependencies needed for:"
    log_info "  • Rust development and compilation"
    log_info "  • Profile-guided optimization (PGO)"
    log_info "  • BOLT binary layout optimization"
    log_info "  • NVIDIA CUDA support (optional)"
    log_info "  • Performance analysis and benchmarking tools"
    echo

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-cuda)
                export SKIP_CUDA=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-cuda    Skip NVIDIA CUDA installation"
                echo "  --help, -h     Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Confirmation prompt
    read -p "Do you want to proceed with the installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled by user"
        exit 0
    fi

    detect_os
    check_privileges
    update_system
    install_rust
    install_llvm_tools
    install_perf_tools
    install_ml_dependencies
    install_cuda
    configure_system
    install_dev_tools
    verify_installation
    show_post_install

    log_info "🚀 Setup complete! You're ready to build and optimize Inferno."
}

# Show help
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo "Inferno Development Environment Setup Script"
    echo
    echo "This script installs all system dependencies needed for compiling"
    echo "and optimizing Inferno, including:"
    echo "  • Rust toolchain"
    echo "  • LLVM tools (PGO, BOLT)"
    echo "  • Performance analysis tools (perf, hyperfine)"
    echo "  • NVIDIA CUDA drivers and toolkit"
    echo "  • ML and development dependencies"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --skip-cuda    Skip NVIDIA CUDA installation"
    echo "  --help, -h     Show this help message"
    echo
    exit 0
fi

# Run main function
main "$@"