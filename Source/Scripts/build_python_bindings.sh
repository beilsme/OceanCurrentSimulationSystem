#!/usr/bin/env bash
# ==============================================================================
# 文件路径：Source/Scripts/build_python_bindings.sh
# 作者：beilsm
# 版本号：v1.0.3
# 创建时间：2025-07-01 19:45
# 最新更改时间：2025-07-01 22:30
# ==============================================================================
# ✅ 更新说明：
#   - v1.0.3
#       • 自动关联到 Source/PythonEngine/.venv 环境
#       • 直接安装模块到目标虚拟环境
# ==============================================================================

set -euo pipefail

### --- 可按需修改 ---
PY_VER="3.12"
VENV_DIR=".venv"
# 使用 PythonEngine 目录下的虚拟环境
TARGET_VENV_DIR="Source/PythonEngine/.venv"
if command -v nproc >/dev/null; then
  JOBS=$(nproc --all)
elif command -v sysctl >/dev/null; then
  JOBS=$(sysctl -n hw.ncpu)
else
  JOBS=4
fi
### -------------------

# 项目根目录：脚本所在目录向上两级
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ⚠️ 每次强制清理旧构建目录 & 旧 .so（可用环境变量跳过）
CLEAN=${CLEAN_BUILD:-ON}      # 调试期默认清理；export CLEAN_BUILD=OFF 可跳过
if [[ "$CLEAN" == "ON" ]]; then
  echo "🧹 清理旧构建缓存 ..."
  rm -rf "$PROJ_ROOT/Source/CppCore/cmake-build-python"
fi
# -------------------------------------------------------------------------

CPP_CORE="$PROJ_ROOT/Source/CppCore"
PY_ENGINE="$PROJ_ROOT/Source/PythonEngine"
BUILD_DIR="$CPP_CORE/cmake-build-python"
TARGET_VENV="$PROJ_ROOT/$TARGET_VENV_DIR"

echo "🔄 [1/7] Homebrew 依赖检测 ..."
if command -v brew >/dev/null; then
  brew bundle --file=- 2>/dev/null <<BREWFILE
brew "cmake"
brew "eigen"
brew "pybind11"
brew "libomp"
brew "tbb"
brew "netcdf"
brew "hdf5"
BREWFILE
else
  echo "brew 未安装，跳过依赖检测"
fi

echo "🔄 [2/7] 检查目标虚拟环境 ..."
if [[ ! -d "$TARGET_VENV" ]]; then
  echo "创建目标虚拟环境: $TARGET_VENV"
  if command -v brew >/dev/null; then
      PYTHON_BIN="$(brew --prefix python@$PY_VER)/bin/python$PY_VER"
    else
      PYTHON_BIN="$(command -v python${PY_VER} || command -v python3)"
    fi
    "$PYTHON_BIN" -m venv "$TARGET_VENV"
fi

echo "🔄 [3/7] 准备编译环境 ..."
# 激活目标虚拟环境
source "$TARGET_VENV/bin/activate"
pip -q install --upgrade pip
pip -q install numpy scipy matplotlib pybind11

echo "🔄 [4/7] 环境变量配置 ..."
if command -v brew >/dev/null; then
  export CMAKE_PREFIX_PATH="$(brew --prefix pybind11)/share/cmake/pybind11:$(brew --prefix eigen)/share/eigen3"
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"
else
  export CMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/cmake/pybind11:/usr/include/eigen3"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi
export CXXFLAGS="-Wall -Wextra -Wpedantic"

echo "🔄 [5/7] 生成构建目录 ..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$CPP_CORE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_CSHARP_BINDINGS=OFF \
  -DBUILD_TESTS=OFF \
  -DPython3_EXECUTABLE="$TARGET_VENV/bin/python" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_FIND_FRAMEWORK=NEVER \
  -DCMAKE_CXX_FLAGS="-DPYBIND11_DETAILED_ERROR_MESSAGES"

echo "🔄 [6/7] 开始并行编译 ..."
cmake --build . -j"$JOBS"

echo "🔄 [7/7] 安装模块到目标环境 ..."
SO_NAME=$(ls *_cpython-*.so oceansim*.so 2>/dev/null | head -n1 || true)
if [[ -z "$SO_NAME" ]]; then
  echo "❌ 未找到编译生成的 .so 文件，请检查上方编译输出"
  exit 1
fi

# 获取目标环境的 site-packages 路径
SITE_PACKAGES="$TARGET_VENV/lib/python$PY_VER/site-packages"
mkdir -p "$SITE_PACKAGES"

# 复制模块到 site-packages
cp "$BUILD_DIR/$SO_NAME" "$SITE_PACKAGES/oceansim.so"

# 同时在 PythonEngine 目录下创建软链接（保持兼容性）
mkdir -p "$PY_ENGINE"
ln -sf "$SITE_PACKAGES/oceansim.so" "$PY_ENGINE/oceansim.so"

echo "✅ 编译完成！"
echo "📦 模块已安装到: $SITE_PACKAGES/oceansim.so"
echo "🔗 软链接创建于: $PY_ENGINE/oceansim.so"
echo ""
echo "🚀 使用方法："
echo "1. 激活环境: source $TARGET_VENV/bin/activate"
echo "2. 导入模块: python -c 'import oceansim; print(oceansim)'"
echo ""
echo "🔄 立即验证 ..."
source "$TARGET_VENV/bin/activate"
python - <<'PY'
import oceansim, sys
print("✅ oceansim 模块验证成功！")
print("Python 版本:", sys.version)
print("模块位置:", oceansim.__file__)
print("可用功能:", [x for x in dir(oceansim) if not x.startswith('_')])
PY