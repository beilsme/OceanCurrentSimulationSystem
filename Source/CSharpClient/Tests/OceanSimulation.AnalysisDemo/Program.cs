// =====================================
// 海洋统计分析使用示例
// =====================================
using Microsoft.Extensions.Logging;
using OceanSimulation.Infrastructure.ComputeEngines;

class Program
{
    static async Task Main(string[] args)
    {
        using var loggerFactory = LoggerFactory.Create(builder =>
            builder.AddConsole().SetMinimumLevel(LogLevel.Information));
        var logger = loggerFactory.CreateLogger<OceanStatisticalAnalysis>();

        var config = new Dictionary<string, object>
        {
            ["PythonExecutablePath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python",
            ["PythonEngineRootPath"] = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine",
            ["WorkingDirectory"] = "./Temp"
        };

        using var analysis = new OceanStatisticalAnalysis(logger, config);

        if (!await analysis.InitializeAsync())
        {
            Console.WriteLine("初始化失败");
            return;
        }

        string netcdfPath = Path.GetFullPath(
                    "../../../../Source/Data/NetCDF/merged_data.nc"
                );

        Console.WriteLine("=== 海洋统计分析演示 ===\n");

        // 1. 计算涡度和散度场
        Console.WriteLine("1. 计算涡度散度场...");
        string vorticityPath = await analysis.CalculateVorticityDivergenceFieldAsync(netcdfPath);

        if (!string.IsNullOrEmpty(vorticityPath))
        {
            Console.WriteLine($"✓ 涡度散度场图生成: {Path.GetFileName(vorticityPath)}");
        }

        // 2. 计算流速统计
        Console.WriteLine("\n2. 计算流速统计...");
        var statistics = await analysis.CalculateFlowStatisticsAsync(netcdfPath);

        if (statistics != null)
        {
            Console.WriteLine("✓ 统计分析完成:");

            if (statistics.FlowStatistics != null)
            {
                var flow = statistics.FlowStatistics;
                Console.WriteLine($"  流速统计:");
                Console.WriteLine($"    平均流速: {flow.MeanSpeed:F3} m/s");
                Console.WriteLine($"    最大流速: {flow.MaxSpeed:F3} m/s");
                Console.WriteLine($"    主导方向: {flow.DominantDirection:F1}°");
                Console.WriteLine($"    动能密度: {flow.KineticEnergyDensity:F1} J/m³");
            }

            if (statistics.VorticityStatistics != null)
            {
                var vort = statistics.VorticityStatistics;
                Console.WriteLine($"  涡度统计:");
                Console.WriteLine($"    平均涡度: {vort.MeanVorticity:E3} s⁻¹");
                Console.WriteLine($"    气旋数量: {vort.CycloneCount}");
                Console.WriteLine($"    反气旋数量: {vort.AnticycloneCount}");
            }

            if (statistics.DivergenceStatistics != null)
            {
                var div = statistics.DivergenceStatistics;
                Console.WriteLine($"  散度统计:");
                Console.WriteLine($"    平均散度: {div.MeanDivergence:E3} s⁻¹");
                Console.WriteLine($"    辐合区: {div.ConvergenceZones}");
                Console.WriteLine($"    辐散区: {div.DivergenceZones}");
            }
        }

        Console.WriteLine("\n=== 分析完成 ===");
    }
}
