# 1️⃣ 清理旧 build
cd Source/CppCore
rm -rf cmake-build-python
mkdir cmake-build-python && cd cmake-build-python

# 2️⃣ 重新配置 (锁 3.12)
cmake .. -DBUILD_PYTHON_BINDINGS=ON \
-DPython3_EXECUTABLE=$PWD/../../PythonEngine/.venv/bin/python \
-DPython3_FIND_STRATEGY=LOCATION \
-DPython3_FIND_FRAMEWORK=NEVER

# 3️⃣ 构建
cmake --build . --target oceansim -j$(sysctl -n hw.ncpu)

# 4️⃣ 安装到 venv
cp bindings/python/oceansim.cpython-312-darwin*.so \
../../PythonEngine/.venv/lib/python3.12/site-packages/
# 删掉之前写的 alias 文件（如果有）
rm -f ../../PythonEngine/.venv/lib/python3.12/site-packages/oceansim.py
