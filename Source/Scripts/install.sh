#!/bin/bash
# ============================================================================
# 洋流模拟系统Python引擎安装脚本 (Linux/Mac)
# 放置位置: Scripts/install.sh
# 使用方法: chmod +x install.sh && ./install.sh
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查系统
check_system() {
    log_step "检查系统环境..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "检测到Linux系统"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "检测到macOS系统"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ $PYTHON_MAJOR -eq 3 ] && [ $PYTHON_MINOR -ge 8 ]; then
            log_info "Python版本: $PYTHON_VERSION ✓"
        else
            log_error "需要Python 3.8或更高版本，当前版本: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "未找到Python3，请先安装Python 3.8+"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        log_error "未找到pip3，请先安装pip"
        exit 1
    fi
    
    log_info "系统检查完成 ✓"
}

# 安装系统依赖
install_system_dependencies() {
    log_step "安装系统依赖..."
    
    if [ "$OS" = "linux" ]; then
        # 检测Linux发行版
        if command -v apt-get &> /dev/null; then
            # Ubuntu/Debian
            log_info "检测到Ubuntu/Debian系统"
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                cmake \
                libnetcdf-dev \
                libhdf5-dev \
                libproj-dev \
                libgeos-dev \
                libffi-dev \
                libssl-dev \
                pkg-config \
                git \
                curl \
                wget
                
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            log_info "检测到CentOS/RHEL系统"
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                netcdf-devel \
                hdf5-devel \
                proj-devel \
                geos-devel \
                libffi-devel \
                openssl-devel \
                pkgconfig \
                git \
                curl \
                wget
                
        elif command -v pacman &> /dev/null; then
            # Arch Linux
            log_info "检测到Arch Linux系统"
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                base-devel \
                cmake \
                netcdf \
                hdf5 \
                proj \
                geos \
                libffi \
                openssl \
                pkgconf \
                git \
                curl \
                wget
        else
            log_warn "未识别的Linux发行版，请手动安装依赖"
        fi
        
    elif [ "$OS" = "macos" ]; then
        # macOS
        log_info "在macOS上安装依赖"
        
        # 检查Homebrew
        if ! command -v brew &> /dev/null; then
            log_info "安装Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # 安装依赖
        brew update
        brew install \
            cmake \
            netcdf \
            hdf5 \
            proj \
            geos \
            libffi \
            openssl \
            pkg-config \
            git \
            curl \
            wget
    fi
    
    log_info "系统依赖安装完成 ✓"
}

# 创建项目目录结构
create_directories() {
    log_step "创建项目目录结构..."
    
    # 定义目录列表
    directories=(
        "Data/NetCDF/Historical"
        "Data/NetCDF/RealTime"
        "Data/NetCDF/Forecast"
        "Data/Models/LSTM"
        "Data/Models/PINN"
        "Data/Models/TrainingData"
        "Data/Results/Simulations"
        "Data/Results/Predictions"
        "Data/Results/Analysis"
        "Data/Cache/PythonCache"
        "Data/Export"
        "Logs"
        "Build/Release/Cpp"
        "Configuration"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "创建目录: $dir"
    done
    
    log_info "目录结构创建完成 ✓"
}

# 设置Python虚拟环境
setup_virtual_environment() {
    log_step "设置Python虚拟环境..."
    
    # 检查venv模块
    if ! python3 -m venv --help &> /dev/null; then
        log_error "Python venv模块不可用，请安装python3-venv"
        exit 1
    fi
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
    else
        log_info "虚拟环境已存在"
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    log_info "升级pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    log_info "虚拟环境设置完成 ✓"
}

# 安装Python依赖
install_python_dependencies() {
    log_step "安装Python依赖包..."
    
    # 确保虚拟环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi
    
    # 安装requirements.txt中的依赖
    if [ -f "Source/PythonEngine/requirements.txt" ]; then
        log_info "从requirements.txt安装依赖..."
        pip install -r Source/PythonEngine/requirements.txt
    else
        log_warn "未找到requirements.txt，使用基础依赖包..."
        pip install \
            numpy scipy pandas xarray \
            fastapi uvicorn pydantic \
            netcdf4 h5py psutil \
            pyyaml matplotlib
    fi
    
    # 安装开发工具（可选）
    read -p "是否安装开发工具? (y/N): " install_dev
    if [[ $install_dev == [Yy]* ]]; then
        log_info "安装开发工具..."
        pip install \
            pytest pytest-asyncio pytest-cov \
            black flake8 mypy \
            jupyter notebook
    fi
    
    log_info "Python依赖安装完成 ✓"
}

# 配置环境变量
setup_environment_variables() {
    log_step "配置环境变量..."
    
    # 创建环境变量脚本
    cat > setup_env.sh << 'EOF'
#!/bin/bash
# 洋流模拟系统环境变量设置

export OCEAN_SIM_ROOT="$(pwd)"
export OCEAN_SIM_DATA="$OCEAN_SIM_ROOT/Data"
export OCEAN_SIM_CONFIG="$OCEAN_SIM_ROOT/Configuration"
export PYTHONPATH="$OCEAN_SIM_ROOT/Source/PythonEngine:$PYTHONPATH"

echo "洋流模拟系统环境变量已设置"
echo "OCEAN_SIM_ROOT: $OCEAN_SIM_ROOT"
echo "OCEAN_SIM_DATA: $OCEAN_SIM_DATA"
echo "OCEAN_SIM_CONFIG: $OCEAN_SIM_CONFIG"
EOF
    
    chmod +x setup_env.sh
    
    # 提示用户添加到shell配置
    SHELL_CONFIG=""
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    
    if [ -n "$SHELL_CONFIG" ]; then
        echo "
# 洋流模拟系统环境变量
source $(pwd)/setup_env.sh" >> $SHELL_CONFIG
        
        log_info "环境变量已添加到 $SHELL_CONFIG"
        log_warn "请运行 'source $SHELL_CONFIG' 或重启终端以生效"
    fi
    
    log_info "环境变量配置完成 ✓"
}

# 配置文件设置
setup_configuration() {
    log_step "设置配置文件..."
    
    # 复制配置文件模板
    if [ -f "Source/PythonEngine/config.yaml" ]; then
        cp Source/PythonEngine/config.yaml Configuration/
        log_info "配置文件已复制到 Configuration/"
    fi
    
    # 创建日志配置
    mkdir -p Configuration
    cat > Configuration/logging.yaml << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  detailed:
    format: '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: Logs/ocean_simulation.log
    maxBytes: 10485760
    backupCount: 5

loggers:
  OceanSimulation:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
EOF
    
    log_info "配置文件设置完成 ✓"
}

# 运行测试
run_tests() {
    log_step "运行系统测试..."
    
    # 确保虚拟环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        source venv/bin/activate
    fi
    
    # 基础导入测试
    log_info "测试Python包导入..."
    python3 -c "
import numpy as np
import scipy
import pandas as pd
import xarray as xr
import fastapi
print('✓ 基础包导入成功')
"
    
    # 测试配置文件
    if [ -f "Configuration/config.yaml" ]; then
        python3 -c "
import yaml
with open('Configuration/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✓ 配置文件读取成功')
"
    fi
    
    log_info "系统测试完成 ✓"
}

# 主安装流程
main() {
    echo "🌊 洋流模拟系统Python引擎安装程序"
    echo "========================================"
    
    # 检查是否在正确的目录
    if [ ! -f "Source/PythonEngine/main.py" ] && [ ! -f "main.py" ]; then
        log_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 执行安装步骤
    check_system
    install_system_dependencies
    create_directories
    setup_virtual_environment
    install_python_dependencies
    setup_environment_variables
    setup_configuration
    run_tests
    
    echo
    log_info "🎉 安装完成！"
    echo
    echo "下一步："
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 设置环境变量: source setup_env.sh"
    echo "3. 启动服务: cd Source/PythonEngine && python main.py"
    echo "4. 访问API文档: http://localhost:8000/docs"
}

# 执行主函数
main "$@"