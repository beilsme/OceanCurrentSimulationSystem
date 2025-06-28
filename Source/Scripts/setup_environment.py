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
