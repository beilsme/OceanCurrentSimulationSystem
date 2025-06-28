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
