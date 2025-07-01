#!/usr/bin/env bash
# ==============================================================================
# 文件路径：Source/Scripts/build_python_bindings.sh
# 作者：beilsm
# 版本号：v1.0.2
# 创建时间：2025-07-01 19:45
# 最新更改时间：2025-07-01 20:25
# ==============================================================================
# ✅ 更新说明：
#   - v1.0.2
#       • 修正 PROJ_ROOT 计算，避免出现 Source/Source/CppCore
#       • DYLD_LIBRARY_PATH 改为 ${...:-} 防止未定义报错
# ==============================================================================

set -euo pipefail

### --- 可按需修改 ---
PY_VER="3.12"
VENV_DIR=".venv"
JOBS=$(sysctl -n hw.ncpu)
### -------------------

# 项目根目录：脚本所在目录向上两级
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CPP_CORE="$PROJ_ROOT/Source/CppCore"
PY_ENGINE="$PROJ_ROOT/Source/PythonEngine"
BUILD_DIR="$CPP_CORE/cmake-build-python"

echo "🔄 [1/6] Homebrew 依赖检测 ..."
brew bundle --file=- 2>/dev/null <<BREWFILE
brew "cmake"
brew "eigen"
brew "pybind11"
brew "libomp"
brew "tbb"
brew "netcdf"
brew "hdf5"
BREWFILE

echo "🔄 [2/6] Python 虚拟环境准备 ..."
if [[ ! -d "$PROJ_ROOT/$VENV_DIR" ]]; then
  "$(brew --prefix python@$PY_VER)/bin/python$PY_VER" -m venv "$PROJ_ROOT/$VENV_DIR"
fi
source "$PROJ_ROOT/$VENV_DIR/bin/activate"
pip -q install --upgrade pip
pip -q install numpy scipy matplotlib pybind11

echo "🔄 [3/6] 环境变量配置 ..."
export CMAKE_PREFIX_PATH="$(brew --prefix pybind11)/share/cmake/pybind11:$(brew --prefix eigen)/share/eigen3"
export CXXFLAGS="-Wall -Wextra -Wpedantic"
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"

echo "🔄 [4/6] 生成构建目录 ..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$CPP_CORE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_CSHARP_BINDINGS=OFF \
  -DBUILD_TESTS=OFF \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_FIND_FRAMEWORK=NEVER

echo "🔄 [5/6] 开始并行编译 ..."
cmake --build . -j"$JOBS"

echo "🔄 [6/6] 软链接 Python 动态库 ..."
SO_NAME=$(ls *_cpython-*.so oceansim*.so 2>/dev/null | head -n1 || true)
if [[ -z "$SO_NAME" ]]; then
  echo "❌ 未找到编译生成的 .so 文件，请检查上方编译输出"
  exit 1
fi
mkdir -p "$PY_ENGINE"
ln -sf "$BUILD_DIR/$SO_NAME" "$PY_ENGINE/oceansim.so"

echo "✅ 编译完成：$PY_ENGINE/oceansim.so"
echo "🚀 立即验证："
echo "source $PROJ_ROOT/$VENV_DIR/bin/activate && python -c 'import oceansim, sys; print(\"oceansim ✔\", sys.version)'"
echo "🔄 自动验证 oceansim ..."
source "$PROJ_ROOT/$VENV_DIR/bin/activate"
python - <<'PY'
import oceansim, sys
print("oceansim ✔ 自动验证通过！ Python", sys.version)
PY