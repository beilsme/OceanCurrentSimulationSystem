@echo off
REM ============================================================================
REM 洋流模拟系统Python引擎安装脚本 (Windows)
REM 放置位置: Scripts/install.bat
REM 使用方法: 双击运行或在命令行中执行 install.bat
REM ============================================================================

setlocal EnableDelayedExpansion

REM 设置控制台编码为UTF-8
chcp 65001 >nul

echo 🌊 洋流模拟系统Python引擎安装程序
echo ========================================

REM 检查管理员权限
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] 检测到管理员权限
) else (
    echo [WARN] 建议以管理员身份运行以安装系统依赖
)

REM 检查Python
echo [STEP] 检查Python环境...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 未找到Python，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python版本: %PYTHON_VERSION%

REM 检查pip
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] 未找到pip，请确保pip已安装
    pause
    exit /b 1
)

REM 检查项目目录
if not exist "Source\PythonEngine\main.py" (
    if not exist "main.py" (
        echo [ERROR] 请在项目根目录运行此脚本
        pause
        exit /b 1
    )
)

REM 创建目录结构
echo [STEP] 创建项目目录结构...
mkdir "Data\NetCDF\Historical" 2>nul
mkdir "Data\NetCDF\RealTime" 2>nul
mkdir "Data\NetCDF\Forecast" 2>nul
mkdir "Data\Models\LSTM" 2>nul
mkdir "Data\Models\PINN" 2>nul
mkdir "Data\Models\TrainingData" 2>nul
mkdir "Data\Results\Simulations" 2>nul
mkdir "Data\Results\Predictions" 2>nul
mkdir "Data\Results\Analysis" 2>nul
mkdir "Data\Cache\PythonCache" 2>nul
mkdir "Data\Export" 2>nul
mkdir "Logs" 2>nul
mkdir "Build\Release\Cpp" 2>nul
mkdir "Configuration" 2>nul
echo [INFO] 目录结构创建完成 ✓

REM 设置虚拟