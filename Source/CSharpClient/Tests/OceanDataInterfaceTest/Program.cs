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
        services.AddLogging(builder => builder.AddConsole());
        var serviceProvider = services.BuildServiceProvider();
        var logger = serviceProvider.GetService<ILogger<OceanDataInterface>>();

        // 配置路径
        var config = new Dictionary<string, object>
        {
            ["PythonExecutablePath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python",
            ["PythonEngineRootPath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine",
            ["WorkingDirectory"] = "./TestOutput"
        };

        using var oceanInterface = new OceanDataInterface(logger!, config);

        try
        {
            // 1. 初始化
            Console.WriteLine("🔄 初始化Python环境...");
            if (!await oceanInterface.InitializeAsync())
            {
                Console.WriteLine("❌ 初始化失败");
                return;
            }
            Console.WriteLine("✅ 初始化成功");

            // 2. 生成可视化
            string dataFile = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/Data/NetCDF/merged_data.nc";

            if (!File.Exists(dataFile))
            {
                Console.WriteLine($"❌ 数据文件不存在: {dataFile}");
                return;
            }

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
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n💥 发生错误: {ex.Message}");
        }

        Console.WriteLine("\n✨ 测试完成，按任意键退出...");
        Console.ReadKey();
    }
}
