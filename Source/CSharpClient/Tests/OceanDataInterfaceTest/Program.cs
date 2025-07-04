using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OceanSimulation.Infrastructure.ComputeEngines;

namespace OceanDataInterfaceTest;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== 极简海洋数据可视化测试 ===\n");

        // 配置日志
        var services = new ServiceCollection();
        services.AddLogging(builder => builder.AddConsole().SetMinimumLevel(LogLevel.Debug));
        var serviceProvider = services.BuildServiceProvider();
        var logger = serviceProvider.GetService<ILogger<OceanDataInterface>>();

        // 配置路径 - 修正路径
        var config = new Dictionary<string, object>
        {
            ["PythonExecutablePath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python",
            ["PythonEngineRootPath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine",
            ["WorkingDirectory"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/Temp"
        };

        // 检查关键路径是否存在
        Console.WriteLine("🔍 检查关键路径...");
        foreach (var kvp in config)
        {
            var path = kvp.Value.ToString();
            var exists = kvp.Key.Contains("Executable") ? File.Exists(path) : Directory.Exists(path);
            Console.WriteLine($"  {kvp.Key}: {path} - {(exists ? "✅" : "❌")}");

            if (!exists)
            {
                Console.WriteLine($"❌ 路径不存在: {path}");
                if (kvp.Key == "WorkingDirectory")
                {
                    Console.WriteLine("🔧 尝试创建工作目录...");
                    try
                    {
                        Directory.CreateDirectory(path);
                        Console.WriteLine("✅ 工作目录创建成功");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"❌ 创建工作目录失败: {ex.Message}");
                    }
                }
            }
        }

        // 检查Python环境
        Console.WriteLine("\n🐍 检查Python环境...");
        var pythonPath = config["PythonExecutablePath"].ToString();
        if (File.Exists(pythonPath))
        {
            try
            {
                var process = new System.Diagnostics.Process
                {
                    StartInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = "--version",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false
                    }
                };
                process.Start();
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();

                Console.WriteLine($"  Python版本: {output.Trim()}");
                if (!string.IsNullOrEmpty(error))
                {
                    Console.WriteLine($"  错误信息: {error}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  检查Python版本失败: {ex.Message}");
            }
        }

        // 检查包装器脚本
        var wrapperPath = Path.Combine(config["PythonEngineRootPath"].ToString(), "wrappers", "ocean_data_wrapper.py");
        Console.WriteLine($"\n📝 检查包装器脚本: {wrapperPath}");
        Console.WriteLine($"  存在: {(File.Exists(wrapperPath) ? "✅" : "❌")}");

        // 检查数据文件
        string dataFile = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/Data/NetCDF/merged_data.nc";
        Console.WriteLine($"\n📁 检查数据文件: {dataFile}");
        Console.WriteLine($"  存在: {(File.Exists(dataFile) ? "✅" : "❌")}");

        if (!File.Exists(dataFile))
        {
            Console.WriteLine("❌ 数据文件不存在，测试无法继续");
            Console.WriteLine("\n💡 建议:");
            Console.WriteLine("  1. 检查数据文件路径是否正确");
            Console.WriteLine("  2. 确保NetCDF数据文件已准备好");
            Console.WriteLine("\n✨ 测试完成，按任意键退出...");
            Console.ReadKey();
            return;
        }

        using var oceanInterface = new OceanDataInterface(logger!, config);

        try
        {
            // 1. 初始化
            Console.WriteLine("\n🔄 初始化Python环境...");
            if (!await oceanInterface.InitializeAsync())
            {
                Console.WriteLine("❌ 初始化失败");
                return;
            }
            Console.WriteLine("✅ 初始化成功");

            // 2. 生成可视化
            Console.WriteLine($"\n🎨 正在生成可视化图像...");
            Console.WriteLine($"📁 数据文件: {dataFile}");

            string imagePath = await oceanInterface.GenerateVisualizationFromFileAsync(dataFile);

            if (!string.IsNullOrEmpty(imagePath) && File.Exists(imagePath))
            {
                Console.WriteLine($"\n🎉 成功生成可视化图像!");
                Console.WriteLine($"📸 图像位置: {Path.GetFullPath(imagePath)}");

                // 尝试打开图像
                Console.WriteLine($"\n💡 提示: 可以用以下命令查看图像:");
                Console.WriteLine($"   open \"{Path.GetFullPath(imagePath)}\"");
            }
            else
            {
                Console.WriteLine("❌ 可视化生成失败");

                // 检查临时目录中是否有文件
                var tempDir = config["WorkingDirectory"].ToString();
                if (Directory.Exists(tempDir))
                {
                    var files = Directory.GetFiles(tempDir);
                    Console.WriteLine($"\n🔍 临时目录内容 ({tempDir}):");
                    if (files.Length == 0)
                    {
                        Console.WriteLine("  目录为空");
                    }
                    else
                    {
                        foreach (var file in files)
                        {
                            var fileInfo = new FileInfo(file);
                            Console.WriteLine($"  - {Path.GetFileName(file)} ({fileInfo.Length} bytes)");
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n💥 发生错误: {ex.Message}");
            Console.WriteLine($"📝 详细信息: {ex}");
        }

        Console.WriteLine("\n✨ 测试完成，按任意键退出...");
        Console.ReadKey();
    }
}
