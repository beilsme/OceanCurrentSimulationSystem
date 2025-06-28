#!/bin/bash
# Scripts/build_cpp.sh - C++核心模块构建脚本

set -e  # 出错时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查构建依赖..."
    
    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake未安装，请先安装CMake 3.16或更高版本"
        exit 1
    fi
    
    # 检查编译器
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        log_error "未找到C++编译器，请安装GCC或Clang"
        exit 1
    fi
    
    # 检查Eigen
    if [ ! -d "CppCore/dependencies/eigen" ]; then
        log_warning "Eigen库未找到，将自动下载"
        download_eigen
    fi
    
    # 检查Intel TBB
    if ! pkg-config --exists tbb; then
        log_warning "Intel TBB未找到，将尝试系统安装"
        install_tbb
    fi
    
    log_success "依赖检查完成"
}

# 下载Eigen库
download_eigen() {
    log_info "下载Eigen库..."
    cd CppCore/dependencies
    
    if [ ! -d "eigen" ]; then
        git clone --depth 1 --branch 3.4 https://gitlab.com/libeigen/eigen.git
        log_success "Eigen库下载完成"
    fi
    
    cd ../..
}

# 安装Intel TBB
install_tbb() {
    log_info "安装Intel TBB..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y libtbb-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install tbb
    else
        log_warning "未知操作系统，请手动安装Intel TBB"
    fi
}

# 设置构建参数
setup_build_config() {
    # 构建类型
    BUILD_TYPE=${BUILD_TYPE:-Release}
    
    # 线程数
    NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # 编译器选择
    if [ -n "$CXX" ]; then
        COMPILER_OPTION="-DCMAKE_CXX_COMPILER=$CXX"
    else
        COMPILER_OPTION=""
    fi
    
    # Python绑定
    BUILD_PYTHON=${BUILD_PYTHON:-ON}
    
    log_info "构建配置:"
    log_info "  构建类型: $BUILD_TYPE"
    log_info "  并行线程: $NPROC"
    log_info "  Python绑定: $BUILD_PYTHON"
}

# 清理构建目录
clean_build() {
    log_info "清理构建目录..."
    
    cd CppCore
    
    if [ -d "build" ]; then
        rm -rf build
    fi
    
    if [ -d "bin" ]; then
        rm -rf bin
    fi
    
    if [ -d "lib" ]; then
        rm -rf lib
    fi
    
    log_success "构建目录清理完成"
    cd ..
}

# 配置CMake
configure_cmake() {
    log_info "配置CMake..."
    
    cd CppCore
    mkdir -p build
    cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DBUILD_PYTHON_BINDINGS=$BUILD_PYTHON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INSTALL_PREFIX=../install \
        $COMPILER_OPTION
    
    if [ $? -eq 0 ]; then
        log_success "CMake配置完成"
    else
        log_error "CMake配置失败"
        exit 1
    fi
    
    cd ../..
}

# 编译C++核心
build_cpp_core() {
    log_info "编译C++核心模块..."
    
    cd CppCore/build
    
    make -j$NPROC
    
    if [ $? -eq 0 ]; then
        log_success "C++核心模块编译完成"
    else
        log_error "C++核心模块编译失败"
        exit 1
    fi
    
    cd ../..
}

# 运行测试
run_tests() {
    log_info "运行单元测试..."
    
    cd CppCore/build
    
    if [ -f "tests/unit_tests" ]; then
        ./tests/unit_tests
        
        if [ $? -eq 0 ]; then
            log_success "单元测试通过"
        else
            log_error "单元测试失败"
            exit 1
        fi
    else
        log_warning "未找到单元测试可执行文件"
    fi
    
    cd ../..
}

# 安装库文件
install_libraries() {
    log_info "安装库文件..."
    
    cd CppCore/build
    
    make install
    
    if [ $? -eq 0 ]; then
        log_success "库文件安装完成"
    else
        log_error "库文件安装失败"
        exit 1
    fi
    
    cd ../..
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 检查静态库
    if [ -f "CppCore/install/lib/libOceanSimCore.a" ]; then
        log_success "静态库安装成功"
    else
        log_error "静态库安装失败"
        exit 1
    fi
    
    # 检查动态库
    if [ -f "CppCore/install/lib/libOceanSimCSharp.so" ] || [ -f "CppCore/install/lib/libOceanSimCSharp.dylib" ]; then
        log_success "C#绑定库安装成功"
    else
        log_error "C#绑定库安装失败"
        exit 1
    fi
    
    # 检查头文件
    if [ -d "CppCore/install/include" ]; then
        log_success "头文件安装成功"
    else
        log_error "头文件安装失败"
        exit 1
    fi
    
    log_success "安装验证完成"
}

# 性能测试
run_performance_tests() {
    log_info "运行性能测试..."
    
    cd CppCore/build
    
    if [ -f "tests/performance_tests" ]; then
        ./tests/performance_tests
        
        if [ $? -eq 0 ]; then
            log_success "性能测试完成"
        else
            log_warning "性能测试出现问题"
        fi
    else
        log_warning "未找到性能测试可执行文件"
    fi
    
    cd ../..
}

# 生成文档
generate_documentation() {
    log_info "生成C++文档..."
    
    if command -v doxygen &> /dev/null; then
        cd CppCore
        
        if [ -f "Doxyfile" ]; then
            doxygen Doxyfile
            log_success "C++文档生成完成"
        else
            log_warning "未找到Doxygen配置文件"
        fi
        
        cd ..
    else
        log_warning "Doxygen未安装，跳过文档生成"
    fi
}

# 打包发布
package_release() {
    log_info "打包发布版本..."
    
    VERSION=$(grep "VERSION" CppCore/CMakeLists.txt | head -1 | sed 's/.*VERSION \([0-9.]*\).*/\1/')
    PACKAGE_NAME="OceanSimCore-${VERSION}-$(uname -s)-$(uname -m)"
    
    mkdir -p Build/Packages
    cd Build/Packages
    
    # 创建发布目录
    mkdir -p $PACKAGE_NAME
    
    # 复制文件
    cp -r ../../CppCore/install/* $PACKAGE_NAME/
    cp ../../README.md $PACKAGE_NAME/
    cp ../../LICENSE $PACKAGE_NAME/ 2>/dev/null || true
    
    # 创建安装脚本
    cat > $PACKAGE_NAME/install.sh << 'EOF'
#!/bin/bash
echo "安装OceanSim C++核心库..."
sudo cp -r lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig
echo "安装完成"
EOF
    chmod +x $PACKAGE_NAME/install.sh
    
    # 打包
    tar -czf ${PACKAGE_NAME}.tar.gz $PACKAGE_NAME
    
    log_success "发布包创建完成: Build/Packages/${PACKAGE_NAME}.tar.gz"
    
    cd ../..
}

# 主函数
main() {
    echo "========================================"
    echo "OceanSim C++核心模块构建脚本"
    echo "========================================"
    
    # 解析命令行参数
    CLEAN=false
    RUN_TESTS=true
    RUN_PERF_TESTS=false
    GENERATE_DOCS=false
    PACKAGE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                CLEAN=true
                shift
                ;;
            --no-tests)
                RUN_TESTS=false
                shift
                ;;
            --perf-tests)
                RUN_PERF_TESTS=true
                shift
                ;;
            --docs)
                GENERATE_DOCS=true
                shift
                ;;
            --package)
                PACKAGE=true
                shift
                ;;
            --build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            --no-python)
                BUILD_PYTHON=OFF
                shift
                ;;
            -h|--help)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --clean          清理构建目录"
                echo "  --no-tests       跳过单元测试"
                echo "  --perf-tests     运行性能测试"
                echo "  --docs           生成文档"
                echo "  --package        创建发布包"
                echo "  --build-type     构建类型 (Debug/Release)"
                echo "  --no-python      禁用Python绑定"
                echo "  -h, --help       显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行构建步骤
    check_dependencies
    setup_build_config
    
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    configure_cmake
    build_cpp_core
    
    if [ "$RUN_TESTS" = true ]; then
        run_tests
    fi
    
    if [ "$RUN_PERF_TESTS" = true ]; then
        run_performance_tests
    fi
    
    install_libraries
    verify_installation
    
    if [ "$GENERATE_DOCS" = true ]; then
        generate_documentation
    fi
    
    if [ "$PACKAGE" = true ]; then
        package_release
    fi
    
    echo "========================================"
    log_success "C++核心模块构建完成！"
    echo "========================================"
    
    # 显示构建摘要
    log_info "构建摘要:"
    log_info "  静态库: CppCore/install/lib/libOceanSimCore.a"
    log_info "  C#绑定: CppCore/install/lib/libOceanSimCSharp.*"
    log_info "  头文件: CppCore/install/include/"
    
    if [ "$BUILD_PYTHON" = "ON" ]; then
        log_info "  Python模块: CppCore/build/python_modules/"
    fi
}

# Windows批处理脚本
create_windows_script() {
    cat > Scripts/build_cpp.bat << 'EOF'
@echo off
REM Windows构建脚本

echo ========================================
echo OceanSim C++核心模块构建脚本 (Windows)
echo ========================================

REM 检查Visual Studio
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Visual Studio编译器
    echo 请在Visual Studio开发者命令提示符中运行此脚本
    pause
    exit /b 1
)

REM 检查CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到CMake
    echo 请先安装CMake并添加到PATH
    pause
    exit /b 1
)

REM 创建构建目录
cd CppCore
if not exist build mkdir build
cd build

REM 配置CMake
echo 配置CMake...
cmake .. -G "Visual Studio 16 2019" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_PYTHON_BINDINGS=ON

if %errorlevel% neq 0 (
    echo CMake配置失败
    pause
    exit /b 1
)

REM 编译
echo 编译C++核心模块...
cmake --build . --config Release --parallel

if %errorlevel% neq 0 (
    echo 编译失败
    pause
    exit /b 1
)

REM 安装
echo 安装库文件...
cmake --install . --prefix ../install

if %errorlevel% neq 0 (
    echo 安装失败
    pause
    exit /b 1
)

cd ..\..

echo ========================================
echo C++核心模块构建完成！
echo ========================================
echo 静态库: CppCore\install\lib\OceanSimCore.lib
echo C#绑定: CppCore\install\bin\OceanSimCSharp.dll
echo 头文件: CppCore\install\include\

pause
EOF
}

# Python环境设置脚本
create_python_setup_script() {
    cat > Scripts/setup_environment.py << 'EOF'
#!/usr/bin/env python3
"""
OceanSim开发环境设置脚本
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令并处理错误"""
    print(f"执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return False
    print(f"Python版本: {sys.version}")
    return True

def install_python_dependencies():
    """安装Python依赖"""
    print("安装Python依赖包...")
    
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "netCDF4>=1.5.8",
        "h5py>=3.6.0",
        "pybind11>=2.8.0",
        "pytest>=6.2.0",
        "jupyter>=1.0.0",
        "ipython>=7.30.0"
    ]
    
    for req in requirements:
        if not run_command(f"pip install {req}"):
            print(f"警告: 无法安装 {req}")
    
    print("Python依赖安装完成")

def setup_conda_environment():
    """设置Conda环境"""
    env_name = "oceansim"
    
    print(f"创建Conda环境: {env_name}")
    
    # 检查环境是否存在
    result = subprocess.run(f"conda env list | grep {env_name}", 
                          shell=True, capture_output=True)
    
    if result.returncode == 0:
        print(f"环境 {env_name} 已存在")
        return
    
    # 创建环境
    conda_cmd = f"""
    conda create -n {env_name} python=3.9 -y
    conda activate {env_name}
    conda install numpy scipy matplotlib netcdf4 h5py -y
    conda install -c conda-forge pybind11 -y
    pip install pytest jupyter ipython
    """
    
    run_command(conda_cmd, check=False)

def create_ide_configs():
    """创建IDE配置文件"""
    print("创建IDE配置文件...")
    
    # VS Code配置
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # settings.json
    settings = {
        "C_Cpp.default.cppStandard": "c++17",
        "C_Cpp.default.compilerPath": "/usr/bin/g++",
        "C_Cpp.default.includePath": [
            "${workspaceFolder}/CppCore/include",
            "${workspaceFolder}/CppCore/dependencies/eigen",
            "/usr/include/eigen3"
        ],
        "python.defaultInterpreterPath": "./venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "cmake.configureOnOpen": True
    }
    
    with open(vscode_dir / "settings.json", "w") as f:
        import json
        json.dump(settings, f, indent=2)
    
    # launch.json
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "C++ Debug",
                "type": "cppdbg",
                "request": "launch",
                "program": "${workspaceFolder}/CppCore/build/tests/unit_tests",
                "args": [],
                "stopAtEntry": False,
                "cwd": "${workspaceFolder}",
                "environment": [],
                "externalConsole": False,
                "MIMode": "gdb"
            }
        ]
    }
    
    with open(vscode_dir / "launch.json", "w") as f:
        import json
        json.dump(launch_config, f, indent=2)
    
    print("VS Code配置文件创建完成")

def main():
    """主函数"""
    print("OceanSim开发环境设置")
    print("=" * 40)
    
    if not check_python_version():
        return 1
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("检测到虚拟环境，继续安装...")
        install_python_dependencies()
    else:
        print("建议在虚拟环境中运行")
        choice = input("是否创建Conda环境? (y/n): ")
        if choice.lower() == 'y':
            setup_conda_environment()
        else:
            install_python_dependencies()
    
    create_ide_configs()
    
    print("=" * 40)
    print("开发环境设置完成！")
    print("=" * 40)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    chmod +x Scripts/setup_environment.py
}

# 创建所有辅助脚本
create_all_scripts() {
    log_info "创建辅助脚本..."
    
    mkdir -p Scripts
    
    create_windows_script
    create_python_setup_script
    
    # 性能测试脚本
    cat > Scripts/run_performance_tests.py << 'EOF'
#!/usr/bin/env python3
"""
性能测试和基准测试脚本
"""

import subprocess
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

def run_cpp_benchmarks():
    """运行C++性能测试"""
    print("运行C++性能基准测试...")
    
    cmd = "./CppCore/build/tests/performance_tests --json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        print(f"C++性能测试失败: {result.stderr}")
        return None

def generate_performance_report(benchmark_data):
    """生成性能报告"""
    if not benchmark_data:
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 粒子模拟性能
    particle_times = benchmark_data.get('particle_simulation', {})
    if particle_times:
        particle_counts = list(particle_times.keys())
        times = list(particle_times.values())
        
        axes[0, 0].plot(particle_counts, times, 'b-o')
        axes[0, 0].set_title('Particle Simulation Performance')
        axes[0, 0].set_xlabel('Particle Count')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True)
    
    # 内存使用
    memory_data = benchmark_data.get('memory_usage', {})
    if memory_data:
        operations = list(memory_data.keys())
        memory = list(memory_data.values())
        
        axes[0, 1].bar(operations, memory)
        axes[0, 1].set_title('Memory Usage by Operation')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 并行效率
    parallel_data = benchmark_data.get('parallel_efficiency', {})
    if parallel_data:
        thread_counts = list(parallel_data.keys())
        efficiency = list(parallel_data.values())
        
        axes[1, 0].plot(thread_counts, efficiency, 'g-s')
        axes[1, 0].set_title('Parallel Efficiency')
        axes[1, 0].set_xlabel('Thread Count')
        axes[1, 0].set_ylabel('Efficiency (%)')
        axes[1, 0].grid(True)
    
    # 算法比较
    algorithm_data = benchmark_data.get('algorithm_comparison', {})
    if algorithm_data:
        algorithms = list(algorithm_data.keys())
        performance = list(algorithm_data.values())
        
        axes[1, 1].bar(algorithms, performance)
        axes[1, 1].set_title('Algorithm Performance Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('performance_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("性能报告已生成: performance_report.png")

if __name__ == "__main__":
    benchmark_data = run_cpp_benchmarks()
    generate_performance_report(benchmark_data)
EOF
    chmod +x Scripts/run_performance_tests.py
    
    log_success "辅助脚本创建完成"
}

# 如果直接运行此脚本
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    create_all_scripts
    main "$@"
fi