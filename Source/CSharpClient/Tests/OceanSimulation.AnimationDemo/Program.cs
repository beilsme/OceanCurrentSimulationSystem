// =====================================
// 简化版洋流动画生成使用示例
// =====================================
using Microsoft.Extensions.Logging;

using OceanSimulation.Infrastructure.ComputeEngines;

class Program
{
    static async Task Main(string[] args)
    {
        // 配置日志
        using var loggerFactory = LoggerFactory.Create(builder =>
            builder.AddConsole().SetMinimumLevel(LogLevel.Information));
        var logger = loggerFactory.CreateLogger<OceanAnimationInterface>();

        // 配置参数
        var config = new Dictionary<string, object>
        {
            ["PythonExecutablePath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python",
            ["PythonEngineRootPath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine",
            ["WorkingDirectory"] = "./TestOutput"

        };

        // 创建动画生成接口
        using var animationInterface = new OceanAnimationInterface(logger, config);

        // 初始化
        if (!await animationInterface.InitializeAsync())
        {
            Console.WriteLine("初始化失败");
            return;
        }

        // NetCDF文件路径
        string netcdfPath = Path.GetFullPath(
            "../../../../Source/Data/NetCDF/merged_data.nc"
        );



        Console.WriteLine("=== 洋流动画生成演示 ===\n");

        // 生成动画 - 默认参数
        Console.WriteLine("生成洋流时间序列动画...");
        string animationPath = await animationInterface.GenerateOceanAnimationAsync(netcdfPath);

        if (!string.IsNullOrEmpty(animationPath))
        {
            Console.WriteLine($"✓ 动画生成成功: {animationPath}");
        }
        else
        {
            Console.WriteLine("✗ 动画生成失败");
            return;
        }

        // 生成自定义参数的动画
        Console.WriteLine("\n生成自定义动画（更多帧，更快播放）...");
        string customOutputPath = "./custom_ocean_animation.gif";

        string customAnimationPath = await animationInterface.GenerateOceanAnimationAsync(
            netcdfPath,
            customOutputPath,
            maxFrames: 30,      // 最多30帧
            frameDelay: 300     // 每帧300ms
        );

        if (!string.IsNullOrEmpty(customAnimationPath))
        {
            Console.WriteLine($"✓ 自定义动画生成成功: {customAnimationPath}");
        }

        Console.WriteLine("\n=== 生成完成 ===");
        Console.WriteLine("提示: 生成的GIF文件可以在浏览器或图片查看器中播放");
    }
}
