@echo off
REM ==============================================================================
REM 海洋仿真系统一键启动脚本 (Windows版本)
REM 文件: startup.bat
REM ==============================================================================

echo === Ocean Simulation System 启动脚本 ===
echo C#主控 + Python数据处理 + C++计算 + Python可视化
echo ================================================

REM 设置代码页为UTF-8
chcp 65001 >nul

REM 设置项目根目录
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo 项目根目录: %PROJECT_ROOT%

REM 1. 环境检查
echo.
echo === 1. 环境检查 ===

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: Python未安装或不在PATH中
    pause
    exit /b 1
) else (
    echo ✅ Python已安装
    python --version
)

REM 检查.NET
dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: .NET未安装或不在PATH中
    pause
    exit /b 1
) else (
    echo ✅ .NET已安装
    for /f "tokens=*" %%i in ('dotnet --version') do echo .NET版本: %%i
)

REM 2. Python环境设置
echo.
echo === 2. Python环境设置 ===

REM 检查虚拟环境
if not exist "venv_oceansim" (
    echo 🔧 首次运行，安装Python-C#接口...
    python setup_python_csharp_interface.py
    if %errorlevel% neq 0 (
        echo ❌ Python接口安装失败
        pause
        exit /b 1
    )
) else (
    echo ✅ Python虚拟环境已存在
)

REM 激活虚拟环境
call venv_oceansim\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ 无法激活Python虚拟环境
    pause
    exit /b 1
) else (
    echo ✅ Python虚拟环境已激活
)

REM 3. C++库检查
echo.
echo === 3. C++库检查 ===

set CPP_LIB=Build\Release\Cpp\oceansim_csharp.dll

if not exist "%CPP_LIB%" (
    echo 🔧 C++库不存在，开始构建...
    
    if exist "Source\CppCore" (
        cd Source\CppCore
        
        REM 创建构建目录
        if not exist "build" mkdir build
        cd build
        
        REM CMake配置
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CSHARP_BINDINGS=ON
        if %errorlevel% neq 0 (
            echo ❌ CMake配置失败
            pause
            exit /b 1
        )
        
        REM 构建
        cmake --build . --config Release
        if %errorlevel% neq 0 (
            echo ❌ C++库构建失败
            pause
            exit /b 1
        )
        
        REM 复制库文件
        if not exist "%PROJECT_ROOT%Build\Release\Cpp" mkdir "%PROJECT_ROOT%Build\Release\Cpp"
        copy Release\oceansim_csharp.dll "%PROJECT_ROOT%Build\Release\Cpp\"
        
        cd /d "%PROJECT_ROOT%"
        echo ✅ C++库构建完成
    ) else (
        echo ❌ C++源码目录不存在
        pause
        exit /b 1
    )
) else (
    echo ✅ C++库已存在: %CPP_LIB%
)

REM 4. C#项目构建
echo.
echo === 4. C#项目构建 ===

if not exist "OceanSimulation.csproj" (
    echo ❌ C#项目文件不存在: OceanSimulation.csproj
    pause
    exit /b 1
)

echo 🔧 构建C#项目...
dotnet build OceanSimulation.csproj --configuration Release
if %errorlevel% neq 0 (
    echo ❌ C#项目构建失败
    pause
    exit /b 1
) else (
    echo ✅ C#项目构建完成
)

REM 5. 启动Python服务（后台）
echo.
echo === 5. 启动Python服务 ===

if exist "Source\PythonEngine\start_python_engine.py" (
    echo 🚀 启动Python引擎服务...
    start "Python Engine" python Source\PythonEngine\start_python_engine.py
    echo ✅ Python服务已在后台启动
    
    REM 等待Python服务启动
    timeout /t 3 /nobreak >nul
) else (
    echo ⚠️  Python启动脚本不存在，将在C#中直接调用Python
)

REM 6. 启动C#主应用
echo.
echo === 6. 启动C#主应用 ===

echo 🚀 启动Ocean Simulation主程序...
dotnet run --project OceanSimulation.csproj

set EXIT_CODE=%errorlevel%

REM 7. 清理工作
echo.
echo === 7. 清理工作 ===

echo 🧹 停用Python虚拟环境...
call venv_oceansim\Scripts\deactivate.bat 2>nul

echo.
if %EXIT_CODE% equ 0 (
    echo 🎉 程序正常退出
) else (
    echo ❌ 程序异常退出 ^(退出代码: %EXIT_CODE%^)
)

echo === 启动脚本执行完毕 ===
echo 按任意键退出...
pause >nul
exit /b %EXIT_CODE%