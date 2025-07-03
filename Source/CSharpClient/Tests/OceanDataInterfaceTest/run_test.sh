#!/bin/bash
echo "=== 运行 OceanDataInterface 测试 ==="

# 确保工作目录存在
mkdir -p ./TestOutput

# 构建项目
echo "构建项目..."
dotnet build

if [ $? -eq 0 ]; then
    echo "构建成功，运行测试..."
    dotnet run
else
    echo "构建失败！"
    exit 1
fi
