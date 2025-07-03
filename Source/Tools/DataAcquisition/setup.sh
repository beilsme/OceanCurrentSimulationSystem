#!/bin/bash

# TOPAZ EnKF系统环境设置脚本
# 使用方法: chmod +x setup.sh && ./setup.sh

set -e  # 遇到错误时退出

echo "🌊 TOPAZ EnKF系统环境设置"
echo "=========================="

# 检测操作系统
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "检测到操作系统: $OS"

# 创建目录结构
echo "📁 创建目录结构..."
mkdir -p ocean_data/{sst,sla,ice,argo,atmos,bathymetry,logs,config}
mkdir -p test_data
mkdir -p scripts
echo "目录创建完成"

# 检查Python环境
echo "🐍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3未安装，请先安装Python 3.8或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python版本: $PYTHON_VERSION"

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未安装，正在安装..."
    if [[ "$OS" == "linux" ]]; then
        sudo apt-get update && sudo apt-get install -y python3-pip
    elif [[ "$OS" == "macos" ]]; then
        curl https://bootstrap.pypa.io/get-pip.py | python3
    fi
fi

# 安装系统依赖
echo "📦 安装系统依赖..."
if [[ "$OS" == "linux" ]]; then
    echo "正在安装Linux依赖包..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            libnetcdf-dev \
            libhdf5-dev \
            libgeos-dev \
            libproj-dev \
            gcc \
            g++ \
            make \
            git
    elif command -v yum &> /dev/null; then
        sudo yum install -y \
            netcdf-devel \
            hdf5-devel \
            geos-devel \
            proj-devel \
            gcc \
            gcc-c++ \
            make \
            git
    fi
elif [[ "$OS" == "macos" ]]; then
    echo "正在安装macOS依赖包..."
    if command -v brew &> /dev/null; then
        brew install netcdf hdf5 geos proj
    else
        echo "⚠️ 建议安装Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    fi
fi

# 创建Python虚拟环境
echo "🔧 创建Python虚拟环境..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "虚拟环境创建完成"
else
    echo "虚拟环境已存在"
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip

# 安装Python依赖
echo "📚 安装Python依赖包..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    echo "安装基础依赖包..."
    pip install requests numpy netCDF4 xarray PyYAML pandas scipy matplotlib
    
    # 尝试安装ECMWF API客户端
    echo "安装ECMWF API客户端..."
    pip install ecmwf-api-client cdsapi || echo "⚠️ ECMWF API客户端安装失败，可能需要手动安装"
fi

# 创建配置文件模板
echo "📋 创建配置文件模板..."

# .netrc模板
cat > .netrc.template << 'EOF'
# AVISO认证配置
machine my.aviso.altimetry.fr
login YOUR_AVISO_USERNAME
password YOUR_AVISO_PASSWORD

# NASA EarthData认证配置
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
EOF

# .cdsapirc模板
cat > .cdsapirc.template << 'EOF'
# ECMWF Climate Data Store API配置
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF

# 环境变量模板
cat > .env.template << 'EOF'
# 环境变量配置文件
# 复制为.env并填写实际值

# AVISO认证
AVISO_USERNAME=your_username
AVISO_PASSWORD=your_password

# NASA EarthData认证
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# ECMWF认证
CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
CDSAPI_KEY=your_uid:your_api_key

# 代理设置(如果需要)
# HTTP_PROXY=http://proxy.example.com:8080
# HTTPS_PROXY=https://proxy.example.com:8080
EOF

# 创建快速测试脚本
cat > scripts/quick_test.py << 'EOF'
#!/usr/bin/env python3
"""快速测试脚本"""

import sys
import subprocess
import importlib

def test_imports():
    """测试关键包导入"""
    required_packages = [
        'numpy', 'netCDF4', 'xarray', 'requests', 
        'yaml', 'pandas', 'scipy'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_data_access():
    """测试数据访问"""
    try:
        import requests
        # 测试网络连接
        response = requests.get('https://httpbin.org/status/200', timeout=10)
        if response.status_code == 200:
            print("✅ 网络连接正常")
            return True
        else:
            print("❌ 网络连接异常")
            return False
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TOPAZ系统环境测试")
    print("=" * 30)
    
    print("\n📦 测试包导入...")
    imports_ok = test_imports()
    
    print("\n🌐 测试网络连接...")
    network_ok = test_data_access()
    
    print("\n📊 测试结果:")
    if imports_ok and network_ok:
        print("🎉 环境测试通过! 系统准备就绪")
        sys.exit(0)
    else:
        print("⚠️ 环境测试失败，请检查安装")
        sys.exit(1)
EOF

chmod +x scripts/quick_test.py

# 创建使用说明
cat > README_SETUP.md << 'EOF'
# TOPAZ EnKF系统使用指南

## 环境激活
每次使用前需要激活虚拟环境:
```bash
source venv/bin/activate
```

## 认证配置
1. 复制配置模板:
```bash
cp .netrc.template ~/.netrc
cp .cdsapirc.template ~/.cdsapirc
chmod 600 ~/.netrc ~/.cdsapirc
```

2. 编辑配置文件，填入您的用户名和密码

## 快速测试
```bash
# 测试环境
python scripts/quick_test.py

# 下载测试数据(无需认证)
python quick_start.py

# 下载完整数据(需要认证)
python ocean_data_downloader.py
```

## 常见问题
- 如果遇到SSL错误，尝试: `pip install --upgrade certifi`
- 如果网络慢，考虑使用代理或国内镜像
- 权限问题: 确保.netrc文件权限为600

## 数据目录结构
```
ocean_data/
├── sst/           # 海表温度
├── sla/           # 海平面高度异常  
├── ice/           # 海冰数据
├── argo/          # Argo剖面
├── atmos/         # 大气强迫
├── bathymetry/    # 海底地形
└── logs/          # 日志文件
```
EOF

# 测试环境
echo "🧪 测试环境..."
python scripts/quick_test.py

echo ""
echo "🎉 环境设置完成!"
echo "==================="
echo "✅ 虚拟环境: venv/"
echo "✅ 配置模板: .netrc.template, .cdsapirc.template"
echo "✅ 测试脚本: scripts/quick_test.py"
echo "✅ 使用说明: README_SETUP.md"
echo ""
echo "下一步:"
echo "1. 配置认证信息 (参考 README_SETUP.md)"
echo "2. 运行快速测试: python quick_start.py"
echo "3. 开始下载数据!"
echo ""
echo "激活环境命令: source venv/bin/activate"