#!/bin/bash

# Setup script for dgx-spark-gemms project
# This script sets up dependencies using uv for Python environment management

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default versions
CUTLASS_VERSION="4.4.1"
CUDA_VERSION="13.0"

# Parse command-line arguments
usage() {
    echo -e "${BOLD}${CYAN}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo -e "  ${GREEN}-t, --cutlass VERSION${NC}    CUTLASS version (default: 4.4.1)"
    echo -e "  ${GREEN}-c, --cuda VERSION${NC}       CUDA version (default: 13.0)"
    echo -e "  ${GREEN}-h, --help${NC}              Show this help message"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ${YELLOW}$0${NC}                                    # Use defaults (CUTLASS 4.4.1, CUDA 13.0)"
    echo -e "  ${YELLOW}$0 -t 4.5.0 -c 13.0${NC}                  # Use CUTLASS 4.5.0 and CUDA 13.0"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--cutlass)
            CUTLASS_VERSION="$2"
            shift 2
            ;;
        -c|--cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Setup environment
echo -e "${BOLD}${MAGENTA}🚀 Setting up dgx-spark-gemms environment...${NC}"
echo ""

# Check if uv is installed
echo -e "${BLUE}🔍 Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠️  uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ Error: Failed to install uv${NC}"
        echo -e "${YELLOW}Please install uv manually:${NC}"
        echo -e "${CYAN}   curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ uv installed successfully${NC}"
else
    echo -e "${GREEN}✅ uv is already installed${NC}"
fi

# Determine Python version
PYTHON_VERSION="3.12"
echo -e "${BLUE}🐍 Using Python ${PYTHON_VERSION}${NC}"

# Create or sync uv environment
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists at ./${VENV_DIR}${NC}"
    read -p "$(echo -e ${CYAN}Remove and recreate? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo -e "${BLUE}🔧 Creating uv environment...${NC}"
        uv venv --python $PYTHON_VERSION
        echo -e "${GREEN}✅ uv environment created${NC}"
    else
        echo -e "${GREEN}✅ Using existing uv environment${NC}"
    fi
else
    echo -e "${BLUE}🔧 Creating uv environment...${NC}"
    uv venv --python $PYTHON_VERSION
    echo -e "${GREEN}✅ uv environment created${NC}"
fi

# Activate the environment
source "$VENV_DIR/bin/activate"

# Determine CUDA index URL based on version
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_SHORT}"

echo ""
echo -e "${BLUE}📦 Installing PyTorch with CUDA ${CUDA_VERSION} support...${NC}"
echo -e "${CYAN}   Index URL: ${PYTORCH_INDEX_URL}${NC}"
uv pip install torch --index-url $PYTORCH_INDEX_URL

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PyTorch installed successfully${NC}"
else
    echo -e "${RED}❌ Error: Failed to install PyTorch${NC}"
    echo -e "${YELLOW}Please try manually:${NC}"
    echo -e "${CYAN}   uv pip install torch --index-url ${PYTORCH_INDEX_URL}${NC}"
    exit 1
fi

# Get PyTorch info
PYTORCH_INFO=$(python -c "
import torch
import os
print(f'{torch.__version__}')
print(f'{os.path.dirname(torch.__file__)}')
print(f'{torch.version.cuda if torch.version.cuda else \"cpu\"}')
")

# Parse the output
PYTORCH_VERSION=$(echo "$PYTORCH_INFO" | sed -n '1p')
PYTORCH_PATH=$(echo "$PYTORCH_INFO" | sed -n '2p')
DETECTED_CUDA_VERSION=$(echo "$PYTORCH_INFO" | sed -n '3p')

echo -e "${GREEN}✅ Found PyTorch ${PYTORCH_VERSION}${NC}"
echo -e "${GREEN}   Location: ${PYTORCH_PATH}${NC}"
echo -e "${GREEN}   CUDA version: ${DETECTED_CUDA_VERSION}${NC}"

# Verify PyTorch C++ libraries are available
echo ""
echo -e "${BLUE}🔍 Verifying PyTorch C++ components...${NC}"
MISSING_LIBS=()

# Check for essential libtorch components
if [ ! -d "$PYTORCH_PATH/lib" ]; then
    MISSING_LIBS+=("lib directory")
fi
if [ ! -d "$PYTORCH_PATH/include" ]; then
    MISSING_LIBS+=("include directory")
fi
if [ ! -f "$PYTORCH_PATH/lib/libtorch.so" ] && [ ! -f "$PYTORCH_PATH/lib/libtorch.dylib" ]; then
    MISSING_LIBS+=("libtorch shared library")
fi
if [ ! -d "$PYTORCH_PATH/share/cmake" ]; then
    MISSING_LIBS+=("CMake configuration files")
fi

if [ ${#MISSING_LIBS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Error: PyTorch C++ components not found!${NC}"
    echo -e "${YELLOW}   Missing components:${NC}"
    for lib in "${MISSING_LIBS[@]}"; do
        echo -e "${YELLOW}   - ${lib}${NC}"
    done
    exit 1
fi

echo -e "${GREEN}✅ PyTorch C++ components verified${NC}"
echo -e "${GREEN}   Headers: ${PYTORCH_PATH}/include${NC}"
echo -e "${GREEN}   Libraries: ${PYTORCH_PATH}/lib${NC}"
echo -e "${GREEN}   CMake config: ${PYTORCH_PATH}/share/cmake${NC}"

# Create symlink to PyTorch installation for CMake
LIBTORCH_DIR="third-party/libtorch"
mkdir -p third-party

if [ -L "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}⚠️  Removing existing libtorch symlink...${NC}"
    rm "$LIBTORCH_DIR"
fi

if [ -d "$LIBTORCH_DIR" ] && [ ! -L "$LIBTORCH_DIR" ]; then
    echo -e "${YELLOW}⚠️  Found existing libtorch directory (not a symlink)${NC}"
    read -p "$(echo -e ${CYAN}Do you want to remove it and create a symlink? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}🗑️  Removing existing libtorch...${NC}"
        rm -rf "$LIBTORCH_DIR"
    else
        echo -e "${YELLOW}⚠️  Skipping libtorch symlink creation.${NC}"
        LIBTORCH_DIR=""
    fi
fi

if [ -n "$LIBTORCH_DIR" ]; then
    echo -e "${BLUE}🔗 Creating symlink: ${LIBTORCH_DIR} -> ${PYTORCH_PATH}${NC}"
    ln -sf "$PYTORCH_PATH" "$LIBTORCH_DIR"
    echo -e "${GREEN}✅ libtorch symlink created${NC}"
fi

# CUTLASS configuration
CUTLASS_DIR="third-party/cutlass"
CUTLASS_URL="https://github.com/NVIDIA/cutlass/archive/refs/tags/v${CUTLASS_VERSION}.zip"

echo ""
echo -e "${CYAN}✂️  CUTLASS version:${NC} ${BOLD}${CUTLASS_VERSION}${NC}"
echo ""

# Install unzip and cmake if not available
PACKAGES_NEEDED=()
if ! command -v unzip &> /dev/null; then
    PACKAGES_NEEDED+=(unzip)
fi
if ! command -v cmake &> /dev/null; then
    PACKAGES_NEEDED+=(cmake)
fi
if ! command -v wget &> /dev/null; then
    PACKAGES_NEEDED+=(wget)
fi

if [ ${#PACKAGES_NEEDED[@]} -gt 0 ]; then
    echo -e "${YELLOW}📦 Missing packages: ${PACKAGES_NEEDED[*]}. Installing...${NC}"
    sudo apt install -y "${PACKAGES_NEEDED[@]}"
    echo -e "${GREEN}✅ Packages installed: ${PACKAGES_NEEDED[*]}${NC}"
fi

# Check if CUTLASS already exists
if [ -d "$CUTLASS_DIR" ]; then
    echo -e "${YELLOW}⚠️  CUTLASS already exists in $CUTLASS_DIR${NC}"
    read -p "$(echo -e ${CYAN}Do you want to re-download? \(y/N\): ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ Skipping CUTLASS download.${NC}"
    else
        echo -e "${RED}🗑️  Removing existing CUTLASS...${NC}"
        rm -rf "$CUTLASS_DIR"

        # Download CUTLASS
        echo -e "${BLUE}⬇️  Downloading CUTLASS...${NC}"
        cd third-party
        wget -q --show-progress "$CUTLASS_URL" -O cutlass.zip

        # Extract
        echo -e "${BLUE}📂 Extracting CUTLASS...${NC}"
        unzip -q cutlass.zip

        # Rename to cutlass
        mv cutlass-${CUTLASS_VERSION} cutlass
        rm cutlass.zip
        cd ..
        echo -e "${GREEN}✅ CUTLASS installed${NC}"
    fi
else
    # Download CUTLASS
    echo -e "${BLUE}⬇️  Downloading CUTLASS...${NC}"
    cd third-party
    wget -q --show-progress "$CUTLASS_URL" -O cutlass.zip

    # Extract
    echo -e "${BLUE}📂 Extracting CUTLASS...${NC}"
    unzip -q cutlass.zip

    # Rename to cutlass
    mv cutlass-${CUTLASS_VERSION} cutlass
    rm cutlass.zip
    cd ..
    echo -e "${GREEN}✅ CUTLASS installed${NC}"
fi

# Install additional Python packages
echo ""
echo -e "${BLUE}📦 Installing additional Python packages...${NC}"
uv pip install loguru pandas plotly pytest click ninja

echo -e "${GREEN}✅ Python packages installed${NC}"

echo ""
echo -e "${BOLD}${GREEN}✨ Setup complete!${NC}"
if [ -n "$LIBTORCH_DIR" ]; then
    echo -e "${CYAN}📍 libtorch symlink: ${BOLD}$LIBTORCH_DIR${NC} -> ${PYTORCH_PATH}"
fi
echo -e "${CYAN}📍 PyTorch version: ${BOLD}${PYTORCH_VERSION}${NC} (CUDA ${DETECTED_CUDA_VERSION})"
echo -e "${CYAN}📍 CUTLASS: ${BOLD}$CUTLASS_DIR${NC} (v${CUTLASS_VERSION})"
echo ""
echo -e "${BOLD}${YELLOW}💡 Activate the environment:${NC}"
echo -e "${MAGENTA}  source .venv/bin/activate${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 Build the project:${NC}"
echo -e "${MAGENTA}  cmake -B build${NC}"
echo -e "${MAGENTA}  cmake --build build${NC}"
echo ""
echo -e "${BOLD}${YELLOW}💡 Run tests:${NC}"
echo -e "${MAGENTA}  ctest --test-dir build${NC}"
echo ""
