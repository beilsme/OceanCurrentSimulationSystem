#!/bin/bash

# =====================================
# 文件: create_test_console.sh
# 功能: 创建OceanDataInterface测试控制台应用
# 用法: ./create_test_console.sh
# =====================================

echo "=== 创建 OceanDataInterface 测试控制台应用 ==="

# 设置项目路径
PROJECT_NAME="OceanDataInterfaceTest"
PROJECT_DIR="./Source/CSharpClient/$PROJECT_NAME"

# 创建项目目录
echo "1. 创建项目目录..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 创建控制台项目
echo "2. 创建控制台项目..."
dotnet new console -n $PROJECT_NAME --force

cd $PROJECT_NAME

# 添加必要的包引用
echo "3. 添加NuGet包..."
dotnet add package Microsoft.Extensions.DependencyInjection
dotnet add package Microsoft.Extensions.Logging
dotnet add package Microsoft.Extensions.Logging.Console

# 添加项目引用（假设Domain和Infrastructure项目存在）
echo "4. 添加项目引用..."
dotnet add reference ../../OceanSimulation.Domain/OceanSimulation.Domain.csproj
dotnet add reference ../../OceanSimulation.Infrastructure/OceanSimulation.Infrastructure.csproj

# 创建测试程序
echo "5. 创建测试程序文件..."
cat > Program.cs << 'EOF'
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OceanSimulation.Infrastructure.ComputeEngines;
using OceanSimulation.Domain.ValueObjects;

namespace OceanDataInterfaceTest
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== OceanDataInterface 测试程序 ===\n");

            // 配置服务和日志
            var services = new ServiceCollection();
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Information);
            });

            var serviceProvider = services.BuildServiceProvider();
            var logger = serviceProvider.GetService<ILogger<OceanDataInterface>>();

            // 配置OceanDataInterface
            var config = new Dictionary<string, object>
            {
                ["PythonExecutablePath"] = "python",  // 或指定具体路径
                ["PythonEngineRootPath"] = "../../../PythonEngine",
                ["WorkingDirectory"] = "./TestOutput"
            };

            // 创建OceanDataInterface实例
            using var oceanInterface = new OceanDataInterface(logger, config);

            try
            {
                // 测试1: 初始化
                Console.WriteLine("1. 测试初始化...");
                var initResult = await oceanInterface.InitializeAsync();
                Console.WriteLine(initResult ? "✅ 初始化成功" : "❌ 初始化失败");

                if (!initResult)
                {
                    Console.WriteLine("初始化失败，无法继续测试");
                    return;
                }

                // 测试2: 生成测试矢量场数据并可视化
                Console.WriteLine("\n2. 测试矢量场可视化...");
                await TestVectorFieldVisualization(oceanInterface);

                // 测试3: 测试NetCDF文件处理（如果有文件的话）
                Console.WriteLine("\n3. 测试NetCDF文件处理...");
                await TestNetCDFProcessing(oceanInterface);

                // 测试4: 测试数据统计
                Console.WriteLine("\n4. 测试数据统计...");
                await TestDataStatistics(oceanInterface);

                // 测试5: 测试Shapefile导出
                Console.WriteLine("\n5. 测试Shapefile导出...");
                await TestShapefileExport(oceanInterface);

            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ 测试过程中发生错误: {ex.Message}");
                Console.WriteLine($"详细信息: {ex.StackTrace}");
            }

            Console.WriteLine("\n=== 测试完成 ===");
            Console.WriteLine("按任意键退出...");
            Console.ReadKey();
        }

        static async Task TestVectorFieldVisualization(OceanDataInterface oceanInterface)
        {
            try
            {
                // 创建测试数据
                var vectorField = CreateTestVectorField();
                var parameters = new VectorFieldParameters
                {
                    Skip = 3,
                    FontSize = 14,
                    DPI = 120,
                    ColorMap = "viridis"
                };

                var result = await oceanInterface.GenerateVectorFieldAsync(vectorField, parameters);

                if (result.Success)
                {
                    Console.WriteLine($"✅ 矢量场可视化成功");
                    Console.WriteLine($"   图像路径: {result.ImagePath}");
                }
                else
                {
                    Console.WriteLine($"❌ 矢量场可视化失败: {result.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 矢量场可视化异常: {ex.Message}");
            }
        }

        static async Task TestNetCDFProcessing(OceanDataInterface oceanInterface)
        {
            // 检查是否有测试NetCDF文件
            var testFiles = new[]
            {
                "./test_data.nc",
                "../../../Data/test_ocean.nc",
                "./sample.nc"
            };

            string foundFile = null;
            foreach (var file in testFiles)
            {
                if (File.Exists(file))
                {
                    foundFile = file;
                    break;
                }
            }

            if (foundFile == null)
            {
                Console.WriteLine("⚠️  未找到测试NetCDF文件，跳过NetCDF测试");
                Console.WriteLine("   可在以下位置放置测试文件:");
                foreach (var file in testFiles)
                {
                    Console.WriteLine($"   - {file}");
                }
                return;
            }

            try
            {
                var parameters = new NetCDFLoadParameters
                {
                    TimeIndex = 0,
                    DepthIndex = 0
                };

                var result = await oceanInterface.LoadNetCDFDataAsync(foundFile, parameters);

                if (result.Success)
                {
                    Console.WriteLine($"✅ NetCDF处理成功");
                    Console.WriteLine($"   数据维度: {result.DataSet.U.GetLength(0)}x{result.DataSet.U.GetLength(1)}");
                }
                else
                {
                    Console.WriteLine($"❌ NetCDF处理失败: {result.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ NetCDF处理异常: {ex.Message}");
            }
        }

        static async Task TestDataStatistics(OceanDataInterface oceanInterface)
        {
            try
            {
                var vectorField = CreateTestVectorField();
                var result = await oceanInterface.GetDataStatisticsAsync(vectorField);

                if (result.Success)
                {
                    Console.WriteLine($"✅ 数据统计成功");
                    Console.WriteLine($"   统计项数: {result.Statistics.Count}");

                    foreach (var stat in result.Statistics)
                    {
                        Console.WriteLine($"   - {stat.Key}: {stat.Value}");
                    }
                }
                else
                {
                    Console.WriteLine($"❌ 数据统计失败: {result.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 数据统计异常: {ex.Message}");
            }
        }

        static async Task TestShapefileExport(OceanDataInterface oceanInterface)
        {
            try
            {
                var vectorField = CreateTestVectorField();
                var parameters = new ShapefileExportParameters
                {
                    Skip = 5,
                    FileType = "shp"
                };

                var result = await oceanInterface.ExportVectorShapefileAsync(vectorField, parameters);

                if (result.Success)
                {
                    Console.WriteLine($"✅ Shapefile导出成功");
                    Console.WriteLine($"   输出路径: {result.OutputPath}");
                }
                else
                {
                    Console.WriteLine($"❌ Shapefile导出失败: {result.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Shapefile导出异常: {ex.Message}");
            }
        }

        static OceanDataSet CreateTestVectorField()
        {
            // 创建简单的测试数据
            const int rows = 10;
            const int cols = 10;

            var u = new double[rows, cols];
            var v = new double[rows, cols];
            var lat = new double[rows];
            var lon = new double[cols];

            // 生成测试坐标和数据
            for (int i = 0; i < rows; i++)
            {
                lat[i] = 30.0 + i * 0.5; // 30°N - 34.5°N
                for (int j = 0; j < cols; j++)
                {
                    if (i == 0) lon[j] = 120.0 + j * 0.5; // 120°E - 124.5°E

                    // 简单的测试流场
                    u[i, j] = 0.1 * Math.Sin(i * 0.5) * Math.Cos(j * 0.5);
                    v[i, j] = 0.1 * Math.Cos(i * 0.5) * Math.Sin(j * 0.5);
                }
            }

            return new OceanDataSet
            {
                U = u,
                V = v,
                Latitude = lat,
                Longitude = lon,
                Depth = 10.0,
                TimeInfo = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            };
        }
    }
}
EOF

# 创建项目文件（如果需要特殊配置）
echo "6. 配置项目文件..."
cat > ${PROJECT_NAME}.csproj << 'EOF'
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="6.0.0" />
    <PackageReference Include="Microsoft.Extensions.Logging" Version="6.0.0" />
    <PackageReference Include="Microsoft.Extensions.Logging.Console" Version="6.0.0" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="../../OceanSimulation.Domain/OceanSimulation.Domain.csproj" />
    <ProjectReference Include="../../OceanSimulation.Infrastructure/OceanSimulation.Infrastructure.csproj" />
  </ItemGroup>

</Project>
EOF

# 创建测试脚本
echo "7. 创建运行脚本..."
cat > run_test.sh << 'EOF'
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
EOF

chmod +x run_test.sh

# 创建简单的README
echo "8. 创建说明文档..."
cat > README.md << 'EOF'
# OceanDataInterface 测试程序

## 功能
测试 `OceanDataInterface.cs` 类的各种功能：
- Python引擎初始化
- 矢量场可视化生成
- NetCDF文件处理
- 数据统计分析
- Shapefile导出

## 运行方法

### 方法1: 使用脚本
```bash
./run_test.sh
```

### 方法2: 手动运行
```bash
dotnet build
dotnet run
```

## 准备工作
1. 确保Python环境已配置
2. 确保PythonEngine目录存在并包含必要的Python脚本
3. (可选) 准备测试NetCDF文件

## 输出
- 测试结果会在控制台显示
- 生成的图像和文件会保存在 `./TestOutput` 目录中
EOF

echo ""
echo "✅ 测试控制台应用创建完成！"
echo ""
echo "项目位置: $PROJECT_DIR/$PROJECT_NAME"
echo ""
echo "运行测试："
echo "  cd $PROJECT_DIR/$PROJECT_NAME"
echo "  ./run_test.sh"
echo ""
echo "或者："
echo "  cd $PROJECT_DIR/$PROJECT_NAME"
echo "  dotnet run"
