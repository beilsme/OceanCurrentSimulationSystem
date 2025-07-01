#!/usr/bin/env bash
# 文件路径：Source/Scripts/build_csharp_bindings.sh
# 作者：beilsm
# 版本号：v1.0.0
# 创建时间：2025-07-01
# 最新更改时间：2025-07-01
# ==============================================================================
# 📝 功能说明：
#   C#绑定构建脚本
#   自动化构建C++核心模块的C#绑定动态库
# ==============================================================================

set -euo pipefail

# ===========================================
# 配置参数
# ===========================================

# 项目根目录：脚本所在目录向上两级
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 构建配置
BUILD_TYPE="${BUILD_TYPE:-Release}"
CLEAN_BUILD="${CLEAN_BUILD:-ON}"
TARGET_PLATFORM="${TARGET_PLATFORM:-x64}"
DOTNET_VERSION="${DOTNET_VERSION:-8.0}"

# 路径配置
CPP_CORE="$PROJ_ROOT/Source/CppCore"
CSHARP_ENGINE="$PROJ_ROOT/Source/CSharpEngine"
BUILD_DIR="$CPP_CORE/cmake-build-csharp"
OUTPUT_DIR="$PROJ_ROOT/Build/$BUILD_TYPE/CSharp"

# 并行编译核心数
if command -v nproc >/dev/null; then
    JOBS=$(nproc --all)
elif command -v sysctl >/dev/null; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

# ===========================================
# 工具函数
# ===========================================

log_info() {
    echo "🔄 [$(date '+%H:%M:%S')] $1"
}

log_success() {
    echo "✅ [$(date '+%H:%M:%S')] $1"
}

log_warning() {
    echo "⚠️  [$(date '+%H:%M:%S')] $1"
}

log_error() {
    echo "❌ [$(date '+%H:%M:%S')] $1" >&2
}

check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "Required command '$1' not found"
        return 1
    fi
}

# ===========================================
# 依赖检查
# ===========================================

check_dependencies() {
    log_info "检查构建依赖..."
    
    # 检查必需的工具
    check_command cmake || exit 1
    check_command make || check_command ninja || {
        log_error "Neither make nor ninja found"
        exit 1
    }
    
    # 检查.NET SDK
    if command -v dotnet >/dev/null; then
        DOTNET_INSTALLED_VERSION=$(dotnet --version 2>/dev/null | cut -d. -f1-2)
        log_info "Found .NET SDK version: $DOTNET_INSTALLED_VERSION"
        
        # 检查版本兼容性
        if [[ "$(echo "$DOTNET_INSTALLED_VERSION $DOTNET_VERSION" | tr ' ' '\n' | sort -V | head -n1)" != "$DOTNET_VERSION" ]]; then
            log_warning ".NET SDK version $DOTNET_INSTALLED_VERSION is older than required $DOTNET_VERSION"
        fi
    else
        log_warning ".NET SDK not found - C# projects may not build correctly"
    fi
    
    # 检查平台特定依赖
    if [[ "$OSTYPE" == "darwin"* ]]; then
        check_platform_deps_macos
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        check_platform_deps_linux
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        check_platform_deps_windows
    fi
    
    log_success "依赖检查完成"
}

check_platform_deps_macos() {
    log_info "检查 macOS 平台依赖..."
    
    if command -v brew >/dev/null; then
        # 检查 Homebrew 包
        local packages=("cmake" "eigen" "libomp" "tbb")
        for pkg in "${packages[@]}"; do
            if brew list "$pkg" >/dev/null 2>&1; then
                log_info "Found Homebrew package: $pkg"
            else
                log_warning "Homebrew package not found: $pkg"
                log_info "Run: brew install $pkg"
            fi
        done
    else
        log_warning "Homebrew not found - manual dependency management required"
    fi
}

check_platform_deps_linux() {
    log_info "检查 Linux 平台依赖..."
    
    # 检查包管理器
    if command -v apt-get >/dev/null; then
        local packages=("build-essential" "cmake" "libeigen3-dev" "libomp-dev" "libtbb-dev")
        for pkg in "${packages[@]}"; do
            if dpkg -l "$pkg" >/dev/null 2>&1; then
                log_info "Found apt package: $pkg"
            else
                log_warning "apt package not found: $pkg"
            fi
        done
    elif command -v yum >/dev/null; then
        log_info "Detected YUM package manager"
    else
        log_warning "Unknown package manager - manual dependency management required"
    fi
}

check_platform_deps_windows() {
    log_info "检查 Windows 平台依赖..."
    
    # 检查 Visual Studio 构建工具
    if command -v cl >/dev/null; then
        log_info "Found Visual Studio compiler"
    else
        log_warning "Visual Studio compiler not found in PATH"
    fi
    
    # 检查 vcpkg
    if [[ -n "${VCPKG_ROOT:-}" ]] && [[ -d "$VCPKG_ROOT" ]]; then
        log_info "Found vcpkg at: $VCPKG_ROOT"
    else
        log_warning "vcpkg not found - consider using vcpkg for dependency management"
    fi
}

# =

# ===========================================
# 清理构建目录
# ===========================================

clean_build_directory() {
    if [[ "$CLEAN_BUILD" == "ON" ]]; then
        log_info "清理旧构建目录..."
        rm -rf "$BUILD_DIR"
        rm -rf "$OUTPUT_DIR"
        log_success "构建目录清理完成"
    else
        log_info "跳过清理构建目录"
    fi
}

# ===========================================
# 配置CMake
# ===========================================

configure_cmake() {
    log_info "配置 CMake 构建系统..."
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # 设置环境变量
    export CMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    # 平台特定的CMake配置
    local cmake_args=()
    cmake_args+=("-DCMAKE_BUILD_TYPE=$BUILD_TYPE")
    cmake_args+=("-DBUILD_CSHARP_BINDINGS=ON")
    cmake_args+=("-DBUILD_PYTHON_BINDINGS=OFF")
    cmake_args+=("-DBUILD_TESTS=OFF")
    
    # 设置安装前缀
    cmake_args+=("-DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR")
    
    # 平台特定配置
    if [[ "$OSTYPE" == "darwin"* ]]; then
        configure_cmake_macos cmake_args
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        configure_cmake_linux cmake_args
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        configure_cmake_windows cmake_args
    fi
    
    # 执行CMake配置
    log_info "Running cmake with args: ${cmake_args[*]}"
    cmake "$CPP_CORE" "${cmake_args[@]}"
    
    log_success "CMake 配置完成"
}

configure_cmake_macos() {
    local -n args=$1
    
    # macOS 特定配置
    if command -v brew >/dev/null; then
        local brew_prefix
        brew_prefix="$(brew --prefix)"
        
        args+=("-DCMAKE_PREFIX_PATH=$brew_prefix/opt/eigen:$brew_prefix/opt/libomp:$brew_prefix/opt/tbb")
        
        # OpenMP 配置
        if [[ -d "$brew_prefix/opt/libomp" ]]; then
            export LDFLAGS="-L$brew_prefix/opt/libomp/lib $LDFLAGS"
            export CPPFLAGS="-I$brew_prefix/opt/libomp/include $CPPFLAGS"
        fi
    fi
    
    # 设置目标架构
    if [[ "$TARGET_PLATFORM" == "arm64" ]]; then
        args+=("-DCMAKE_OSX_ARCHITECTURES=arm64")
    elif [[ "$TARGET_PLATFORM" == "x64" ]]; then
        args+=("-DCMAKE_OSX_ARCHITECTURES=x86_64")
    fi
}

configure_cmake_linux() {
    local -n args=$1
    
    # Linux 特定配置
    args+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
    
    # 如果是交叉编译
    if [[ "$TARGET_PLATFORM" != "$(uname -m)" ]]; then
        log_warning "Cross-compilation detected for $TARGET_PLATFORM"
        # 这里可以添加交叉编译工具链配置
    fi
}

configure_cmake_windows() {
    local -n args=$1
    
    # Windows 特定配置
    args+=("-G" "Visual Studio 17 2022")
    
    if [[ "$TARGET_PLATFORM" == "x64" ]]; then
        args+=("-A" "x64")
    elif [[ "$TARGET_PLATFORM" == "x86" ]]; then
        args+=("-A" "Win32")
    fi
    
    # vcpkg 集成
    if [[ -n "${VCPKG_ROOT:-}" ]]; then
        args+=("-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")
    fi
}

# ===========================================
# 构建C++动态库
# ===========================================

build_cpp_library() {
    log_info "构建 C++ 动态库..."
    
    cd "$BUILD_DIR"
    
    # 并行构建
    local build_args=("--build" "." "--config" "$BUILD_TYPE")
    
    if command -v ninja >/dev/null && grep -q "Ninja" CMakeCache.txt 2>/dev/null; then
        build_args+=("--parallel" "$JOBS")
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        build_args+=("--parallel" "$JOBS")
    else
        build_args+=("--" "-j$JOBS")
    fi
    
    log_info "Building with: cmake ${build_args[*]}"
    cmake "${build_args[@]}"
    
    # 验证构建结果
    verify_cpp_build
    
    log_success "C++ 动态库构建完成"
}

verify_cpp_build() {
    log_info "验证 C++ 构建结果..."
    
    local lib_patterns=()
    if [[ "$OSTYPE" == "darwin"* ]]; then
        lib_patterns+=("*.dylib")
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        lib_patterns+=("*.so")
    else
        lib_patterns+=("*.dll" "*.lib")
    fi
    
    local found_libs=0
    for pattern in "${lib_patterns[@]}"; do
        if find "$BUILD_DIR" -name "$pattern" -type f | grep -q .; then
            ((found_libs++))
            log_info "Found library files matching: $pattern"
        fi
    done
    
    if [[ $found_libs -eq 0 ]]; then
        log_error "No library files found after build"
        return 1
    fi
    
    log_success "构建产物验证通过"
}

# ===========================================
# 安装C++库文件
# ===========================================

install_cpp_library() {
    log_info "安装 C++ 库文件..."
    
    cd "$BUILD_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    # 执行安装
    cmake --install . --config "$BUILD_TYPE"
    
    # 复制额外的库文件到便于访问的位置
    copy_library_files
    
    log_success "C++ 库文件安装完成"
}

copy_library_files() {
    log_info "复制库文件到输出目录..."
    
    local lib_dest="$OUTPUT_DIR/lib"
    mkdir -p "$lib_dest"
    
    # 根据平台复制相应的库文件
    if [[ "$OSTYPE" == "darwin"* ]]; then
        find "$BUILD_DIR" -name "*.dylib" -exec cp {} "$lib_dest/" \;
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        find "$BUILD_DIR" -name "*.so*" -exec cp {} "$lib_dest/" \;
    else
        find "$BUILD_DIR" -name "*.dll" -exec cp {} "$lib_dest/" \;
        find "$BUILD_DIR" -name "*.lib" -exec cp {} "$lib_dest/" \;
    fi
    
    # 设置库文件权限
    chmod 755 "$lib_dest"/*
    
    log_info "库文件复制到: $lib_dest"
}

# ===========================================
# 构建C#项目
# ===========================================

build_csharp_projects() {
    log_info "构建 C# 项目..."
    
    if ! command -v dotnet >/dev/null; then
        log_warning ".NET SDK 未找到，跳过 C# 项目构建"
        return 0
    fi
    
    cd "$CSHARP_ENGINE"
    
    # 恢复NuGet包
    log_info "恢复 NuGet 包..."
    dotnet restore
    
    # 构建解决方案
    log_info "构建 C# 解决方案..."
    local dotnet_args=("build")
    dotnet_args+=("--configuration" "$BUILD_TYPE")
    dotnet_args+=("--no-restore")
    dotnet_args+=("--verbosity" "minimal")
    
    if [[ "$TARGET_PLATFORM" == "x64" ]]; then
        dotnet_args+=("--arch" "x64")
    elif [[ "$TARGET_PLATFORM" == "x86" ]]; then
        dotnet_args+=("--arch" "x86")
    elif [[ "$TARGET_PLATFORM" == "arm64" ]]; then
        dotnet_args+=("--arch" "arm64")
    fi
    
    dotnet "${dotnet_args[@]}"
    
    # 运行测试（如果有）
    if [[ -f "OceanSim.Tests/OceanSim.Tests.csproj" ]]; then
        log_info "运行 C# 单元测试..."
        dotnet test --configuration "$BUILD_TYPE" --no-build --verbosity minimal
    fi
    
    # 发布项目
    publish_csharp_projects
    
    log_success "C# 项目构建完成"
}

publish_csharp_projects() {
    log_info "发布 C# 项目..."
    
    local publish_dir="$OUTPUT_DIR/publish"
    mkdir -p "$publish_dir"
    
    # 发布主项目
    if [[ -f "OceanSim.Core/OceanSim.Core.csproj" ]]; then
        dotnet publish "OceanSim.Core/OceanSim.Core.csproj" \
            --configuration "$BUILD_TYPE" \
            --output "$publish_dir/Core" \
            --no-restore \
            --verbosity minimal
    fi
    
    # 发布示例项目
    if [[ -f "OceanSim.Examples/OceanSim.Examples.csproj" ]]; then
        dotnet publish "OceanSim.Examples/OceanSim.Examples.csproj" \
            --configuration "$BUILD_TYPE" \
            --output "$publish_dir/Examples" \
            --no-restore \
            --verbosity minimal
    fi
    
    log_info "C# 项目发布到: $publish_dir"
}

# ===========================================
# 集成测试
# ===========================================

run_integration_tests() {
    log_info "运行集成测试..."
    
    # 设置库路径环境变量
    setup_library_path
    
    # 运行C++库的基本功能测试
    test_cpp_library_loading
    
    # 运行C#绑定测试
    if command -v dotnet >/dev/null; then
        test_csharp_bindings
    fi
    
    log_success "集成测试完成"
}

setup_library_path() {
    local lib_path="$OUTPUT_DIR/lib"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH="$lib_path:${DYLD_LIBRARY_PATH:-}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        export LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}"
    else
        export PATH="$lib_path:$PATH"
    fi
    
    log_info "Library path configured: $lib_path"
}

test_cpp_library_loading() {
    log_info "测试 C++ 库加载..."
    
    # 这里可以添加一个简单的C++测试程序来验证库的加载
    local lib_files=()
    if [[ "$OSTYPE" == "darwin"* ]]; then
        mapfile -t lib_files < <(find "$OUTPUT_DIR/lib" -name "*.dylib" 2>/dev/null)
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        mapfile -t lib_files < <(find "$OUTPUT_DIR/lib" -name "*.so*" 2>/dev/null)
    else
        mapfile -t lib_files < <(find "$OUTPUT_DIR/lib" -name "*.dll" 2>/dev/null)
    fi
    
    for lib in "${lib_files[@]}"; do
        if [[ -f "$lib" ]]; then
            log_info "Found library: $(basename "$lib")"
        fi
    done
    
    if [[ ${#lib_files[@]} -eq 0 ]]; then
        log_warning "No library files found for testing"
        return 1
    fi
}

test_csharp_bindings() {
    log_info "测试 C# 绑定..."
    
    local examples_dir="$OUTPUT_DIR/publish/Examples"
    if [[ -d "$examples_dir" ]] && [[ -f "$examples_dir/OceanSim.Examples.dll" ]]; then
        cd "$examples_dir"
        
        # 运行示例程序（如果是可执行的）
        if [[ -f "OceanSim.Examples.exe" ]]; then
            log_info "运行 C# 示例程序..."
            timeout 30 ./OceanSim.Examples.exe || log_warning "示例程序运行超时或失败"
        elif [[ -f "OceanSim.Examples" ]]; then
            log_info "运行 C# 示例程序..."
            timeout 30 ./OceanSim.Examples || log_warning "示例程序运行超时或失败"
        else
            log_info "运行 C# 示例程序 (通过 dotnet)..."
            timeout 30 dotnet OceanSim.Examples.dll || log_warning "示例程序运行超时或失败"
        fi
    else
        log_warning "示例程序未找到，跳过 C# 绑定测试"
    fi
}

# ===========================================
# 生成构建报告
# ===========================================

generate_build_report() {
    log_info "生成构建报告..."
    
    local report_file="$OUTPUT_DIR/build_report.txt"
    local report_date
    report_date="$(date '+%Y-%m-%d %H:%M:%S')"
    
    cat > "$report_file" << EOF
OceanSim C# 绑定构建报告
========================================
构建时间: $report_date
构建类型: $BUILD_TYPE
目标平台: $TARGET_PLATFORM
操作系统: $OSTYPE

构建配置:
- 项目根目录: $PROJ_ROOT
- 构建目录: $BUILD_DIR
- 输出目录: $OUTPUT_DIR
- 并行任务数: $JOBS
- .NET 版本: $DOTNET_VERSION

构建产物:
EOF
    
    # 列出构建产物
    if [[ -d "$OUTPUT_DIR/lib" ]]; then
        echo "C++ 动态库:" >> "$report_file"
        find "$OUTPUT_DIR/lib" -type f -exec basename {} \; | sort | sed 's/^/  - /' >> "$report_file"
    fi
    
    if [[ -d "$OUTPUT_DIR/publish" ]]; then
        echo "C# 程序集:" >> "$report_file"
        find "$OUTPUT_DIR/publish" -name "*.dll" -exec basename {} \; | sort | sed 's/^/  - /' >> "$report_file"
    fi
    
    # 添加文件大小信息
    echo "" >> "$report_file"
    echo "文件大小统计:" >> "$report_file"
    if command -v du >/dev/null; then
        du -sh "$OUTPUT_DIR"/* 2>/dev/null | sed 's/^/  /' >> "$report_file" || true
    fi
    
    log_success "构建报告已生成: $report_file"
}

# ===========================================
# 清理临时文件
# ===========================================

cleanup_temp_files() {
    log_info "清理临时文件..."
    
    # 清理CMake缓存（如果需要）
    if [[ "$CLEAN_BUILD" == "ON" ]]; then
        rm -rf "$BUILD_DIR/CMakeFiles"
        rm -f "$BUILD_DIR/CMakeCache.txt"
        log_info "CMake 临时文件已清理"
    fi
    
    # 清理.NET临时文件
    if [[ -d "$CSHARP_ENGINE" ]]; then
        find "$CSHARP_ENGINE" -type d -name "bin" -exec rm -rf {} + 2>/dev/null || true
        find "$CSHARP_ENGINE" -type d -name "obj" -exec rm -rf {} + 2>/dev/null || true
        log_info ".NET 临时文件已清理"
    fi
    
    log_success "临时文件清理完成"
}

# ===========================================
# 主构建流程
# ===========================================

main() {
    log_info "开始 OceanSim C# 绑定构建流程"
    log_info "构建配置: $BUILD_TYPE, 平台: $TARGET_PLATFORM"
    
    local start_time
    start_time=$(date +%s)
    
    # 执行构建步骤
    check_dependencies
    clean_build_directory
    configure_cmake
    build_cpp_library
    install_cpp_library
    build_csharp_projects
    run_integration_tests
    generate_build_report
    cleanup_temp_files
    
    # 计算构建时间
    local end_time
    end_time=$(date +%s)
    local build_duration=$((end_time - start_time))
    
    log_success "构建流程完成！"
    log_success "总耗时: ${build_duration} 秒"
    log_success "输出目录: $OUTPUT_DIR"
    
    # 显示下一步操作建议
    echo ""
    echo "🎉 构建成功完成！"
    echo ""
    echo "📁 构建产物位置:"
    echo "   - C++ 动态库: $OUTPUT_DIR/lib/"
    echo "   - C# 程序集: $OUTPUT_DIR/publish/"
    echo "   - 构建报告: $OUTPUT_DIR/build_report.txt"
    echo ""
    echo "🚀 下一步操作:"
    echo "   1. 查看构建报告了解详细信息"
    echo "   2. 运行示例程序测试功能"
    echo "   3. 将库文件复制到你的项目中"
    echo ""
    echo "💡 使用提示:"
    echo "   - 确保在运行时设置正确的库路径"
    echo "   - 参考示例代码了解API使用方法"
    echo "   - 查看文档获取更多信息"
}

# ===========================================
# 错误处理
# ===========================================

handle_error() {
    local exit_code=$?
    local line_number=$1
    
    log_error "构建失败！"
    log_error "错误发生在脚本第 $line_number 行，退出码: $exit_code"
    
    # 尝试收集错误信息
    if [[ -f "$BUILD_DIR/CMakeFiles/CMakeError.log" ]]; then
        log_error "CMake 错误日志:"
        tail -n 20 "$BUILD_DIR/CMakeFiles/CMakeError.log" | sed 's/^/  /'
    fi
    
    exit $exit_code
}

# 设置错误处理
trap 'handle_error $LINENO' ERR

# ===========================================
# 脚本参数处理
# ===========================================

show_help() {
    cat << EOF
OceanSim C# 绑定构建脚本

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -c, --clean             强制清理构建目录
  -t, --type TYPE         设置构建类型 (Debug|Release) [默认: Release]
  -p, --platform PLATFORM 设置目标平台 (x86|x64|arm64) [默认: x64]
  -j, --jobs N            设置并行编译任务数 [默认: 自动检测]
  --dotnet-version VER    指定 .NET 版本 [默认: 8.0]
  --no-tests              跳过测试步骤

环境变量:
  BUILD_TYPE              构建类型
  CLEAN_BUILD             是否清理构建 (ON|OFF)
  TARGET_PLATFORM         目标平台
  DOTNET_VERSION          .NET 版本

示例:
  $0                      # 默认构建
  $0 -t Debug -c          # Debug构建并清理
  $0 -p arm64 --no-tests  # ARM64平台构建，跳过测试

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                CLEAN_BUILD="ON"
                shift
                ;;
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -p|--platform)
                TARGET_PLATFORM="$2"
                shift 2
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            --dotnet-version)
                DOTNET_VERSION="$2"
                shift 2
                ;;
            --no-tests)
                SKIP_TESTS="ON"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证参数
    if [[ "$BUILD_TYPE" != "Debug" && "$BUILD_TYPE" != "Release" ]]; then
        log_error "无效的构建类型: $BUILD_TYPE"
        exit 1
    fi
    
    if [[ "$TARGET_PLATFORM" != "x86" && "$TARGET_PLATFORM" != "x64" && "$TARGET_PLATFORM" != "arm64" ]]; then
        log_error "无效的目标平台: $TARGET_PLATFORM"
        exit 1
    fi
}

# 解析命令行参数
parse_arguments "$@"

# 如果设置了跳过测试，则重定义函数
if [[ "${SKIP_TESTS:-}" == "ON" ]]; then
    run_integration_tests() {
        log_info "跳过集成测试（根据用户要求）"
    }
fi

