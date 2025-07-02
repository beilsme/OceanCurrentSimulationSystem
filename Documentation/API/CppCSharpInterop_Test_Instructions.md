# Cpp/C# 接口测试指南

本项目提供了 `InteropBasicTest` 用于快速验证 C++ 核心库与 C# 封装的互操作是否正常。

## 构建步骤
1. **编译 C++ 库**
   在 `Source` 目录下执行 `build_cpp.sh` 脚本，生成 `liboceansim_csharp` 动态库。
   ```bash
   cd Source
   bash build_cpp.sh
   ```
   编译完成后，可在 `Build/Release/CSharp/lib`（或脚本输出的安装目录）找到生成的库文件。

2. **构建 C# 集成测试**
   使用 `dotnet build` 进入测试项目目录即可：
   ```bash
   cd Source/CSharpClient/Tests/IntegrationTests/CppIntegrationTests
   dotnet build -c Release
   ```

3. **运行测试**
   ```bash
   dotnet run --no-build -c Release
   ```
   程序若能输出 `OceanSim 版本` 字样并返回 0，则说明接口连接成功。

