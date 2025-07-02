#!/bin/bash
# ==============================================================================
# 海洋仿真系统一键启动脚本
# 文件: startup.sh (Linux/Mac) 或 startup.bat (Windows)
# ==============================================================================

echo "=== Ocean Simulation System 启动脚本 ==="
echo "C#主控 + Python数据处理 + C++计算 + Python可视化"
echo "================================================"

# 检查操作系统
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    IS_WINDOWS=true
    PYTHON_CMD="python"
    DOTNET_CMD="dotnet.exe"
else
    IS_WINDOWS=false
    PYTHON_CMD="python3"
    DOTNET_CMD="dotnet"
fi

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"

# 函数：检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ 错误: $1 未安装或不在PATH中"
        return 1
    else
        echo "✅ $1 已安装"
        return 0
    fi
}

# 函数：检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 文件不存在 - $1"
        return 1
    else
        echo "✅ 文件存在 - $1"
        return 0
    fi
}

# 1. 环境检查
echo ""
echo "=== 1. 环境检查 ==="
check_command $PYTHON_CMD || exit 1
check_command $DOTNET_CMD || exit 1

# 检查.NET版本
DOTNET_VERSION=$($DOTNET_CMD --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ .NET版本: $DOTNET_VERSION"
else
    echo "❌ 无法获取.NET版本"
    exit 1
fi

# 2. Python环境设置
echo ""
echo "=== 2. Python环境设置 ==="

# 检查是否需要安装Python接口
if [ ! -d "venv_oceansim" ]; then
    echo "🔧 首次运行，安装Python-C#接口..."
    $PYTHON_CMD setup_python_csharp_interface.py
    if [ $? -ne 0 ]; then
        echo "❌ Python接口安装失败"
        exit 1
    fi
else
    echo "✅ Python虚拟环境已存在"
fi

# 激活虚拟环境
if [ "$IS_WINDOWS" = true ]; then
    source venv_oceansim/Scripts/activate
else
    source venv_oceansim/bin/activate
fi

if [ $? -eq 0 ]; then
    echo "✅ Python虚拟环境已激活"
else
    echo "❌ 无法激活Python虚拟环境"
    exit 1
fi

# 3. C++库检查
echo ""
echo "=== 3. C++库检查 ==="

if [ "$IS_WINDOWS" = true ]; then
    CPP_LIB="Build/Release/Cpp/oceansim_csharp.dll"
else
    CPP_LIB="Build/Release/Cpp/liboceansim_csharp.so"
fi

if [ ! -f "$CPP_LIB" ]; then
    echo "🔧 C++库不存在，开始构建..."
    
    if [ -d "Source/CppCore" ]; then
        cd Source/CppCore
        
        # 创建构建目录
        mkdir -p build
        cd build
        
        # CMake配置和构建
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CSHARP_BINDINGS=ON
        if [ $? -ne 0 ]; then
            echo "❌ CMake配置失败"
            exit 1
        fi
        
        cmake --build . --config Release
        if [ $? -ne 0 ]; then
            echo "❌ C++库构建失败"
            exit 1
        fi
        
        # 复制库文件到指定位置
        mkdir -p "$PROJECT_ROOT/Build/Release/Cpp"
        if [ "$IS_WINDOWS" = true ]; then
            cp Release/oceansim_csharp.dll "$PROJECT_ROOT/Build/Release/Cpp/"
        else
            cp liboceansim_csharp.so "$PROJECT_ROOT/Build/Release/Cpp/"
        fi
        
        cd "$PROJECT_ROOT"
        echo "✅ C++库构建完成"
    else
        echo "❌ C++源码目录不存在"
        exit 1
    fi
else
    echo "✅ C++库已存在: $CPP_LIB"
fi

# 4. C#项目构建
echo ""
echo "=== 4. C#项目构建 ==="

if [ ! -f "OceanSimulation.csproj" ]; then
    echo "❌ C#项目文件不存在: OceanSimulation.csproj"
    exit 1
fi

echo "🔧 构建C#项目..."
$DOTNET_CMD build OceanSimulation.csproj --configuration Release
if [ $? -ne 0 ]; then
    echo "❌ C#项目构建失败"
    exit 1
else
    echo "✅ C#项目构建完成"
fi

# 5. 启动Python服务（后台）
echo ""
echo "=== 5. 启动Python服务 ==="

if [ -f "Source/PythonEngine/start_python_engine.py" ]; then
    echo "🚀 启动Python引擎服务..."
    $PYTHON_CMD Source/PythonEngine/start_python_engine.py &
    PYTHON_PID=$!
    echo "✅ Python服务已启动 (PID: $PYTHON_PID)"
    
    # 等待Python服务启动
    sleep 3
else
    echo "⚠️  Python启动脚本不存在，将在C#中直接调用Python"
fi

# 6. 启动C#主应用
echo ""
echo "=== 6. 启动C#主应用 ==="

echo "🚀 启动Ocean Simulation主程序..."
$DOTNET_CMD run --project OceanSimulation.csproj

# 程序退出后的清理
EXIT_CODE=$?

# 7. 清理工作
echo ""
echo "=== 7. 清理工作 ==="

if [ ! -z "$PYTHON_PID" ]; then
    echo "🧹 停止Python服务..."
    kill $PYTHON_PID 2>/dev/null
    echo "✅ Python服务已停止"
fi

echo "🧹 停用Python虚拟环境..."
deactivate 2>/dev/null

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 程序正常退出"
else
    echo "❌ 程序异常退出 (退出代码: $EXIT_CODE)"
fi

echo "=== 启动脚本执行完毕 ==="
exit $EXIT_CODE