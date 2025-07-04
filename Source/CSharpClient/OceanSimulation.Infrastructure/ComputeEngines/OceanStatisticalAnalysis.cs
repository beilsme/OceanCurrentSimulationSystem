// =====================================
// 文件: OceanStatisticalAnalysis.cs
// 功能: 海洋统计分析 - 流速分布、涡度场、散度场计算
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System.Text.Json;
using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.ValueObjects;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 海洋统计分析接口 - 计算专业海洋学指标
    /// </summary>
    public class OceanStatisticalAnalysis : IDisposable
    {
        private readonly ILogger<OceanStatisticalAnalysis> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private bool _isInitialized = false;

        public OceanStatisticalAnalysis(ILogger<OceanStatisticalAnalysis> logger,
                                       Dictionary<string, object> configuration)
        {
            _logger = logger;
            var config = configuration ?? throw new ArgumentNullException(nameof(configuration));

            _pythonExecutablePath = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python";
            _pythonEngineRootPath = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine";

            _workingDirectory = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/Temp";


            Directory.CreateDirectory(_workingDirectory);
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("初始化海洋统计分析环境");

                // 异步验证Python环境
                var pythonCheck = await Task.Run(() =>
                {
                    try
                    {
                        var process = new System.Diagnostics.Process
                        {
                            StartInfo = new System.Diagnostics.ProcessStartInfo
                            {
                                FileName = _pythonExecutablePath,
                                Arguments = "--version",
                                RedirectStandardOutput = true,
                                RedirectStandardError = true,
                                UseShellExecute = false,
                                CreateNoWindow = true
                            }
                        };
                        process.Start();
                        process.WaitForExit(5000); // 5秒超时
                        return process.ExitCode == 0;
                    }
                    catch
                    {
                        return false;
                    }
                });

                if (!pythonCheck)
                {
                    _logger.LogError("Python环境验证失败");
                    return false;
                }

                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到ocean_data_wrapper.py: {wrapperPath}");
                    return false;
                }

                _isInitialized = true;
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 计算涡度场和散度场
        /// </summary>
        public async Task<string> CalculateVorticityDivergenceFieldAsync(
            string netcdfPath,
            string outputPath = "",
            int timeIndex = 0,
            int depthIndex = 0)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("尚未初始化");

            try
            {
                if (string.IsNullOrEmpty(outputPath))
                    outputPath = Path.Combine(_workingDirectory, $"vorticity_divergence_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var inputData = new
                {
                    action = "calculate_vorticity_divergence",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        output_path = outputPath,
                        time_index = timeIndex,
                        depth_index = depthIndex
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    _logger.LogInformation($"涡度散度场计算完成: {outputPath}");
                    return outputPath;
                }

                _logger.LogError("涡度散度场计算失败");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "计算涡度散度场失败");
                return "";
            }
        }

        /// <summary>
        /// 计算流速统计分布
        /// </summary>
        public async Task<OceanographicStatistics?> CalculateFlowStatisticsAsync(
            string netcdfPath,
            int timeIndex = 0,
            int depthIndex = 0)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("尚未初始化");

            try
            {
                var inputData = new
                {
                    action = "calculate_flow_statistics",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        time_index = timeIndex,
                        depth_index = depthIndex
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean())
                {
                    var statisticsElement = result.GetProperty("statistics");
                    return JsonSerializer.Deserialize<OceanographicStatistics>(statisticsElement.GetRawText());
                }

                _logger.LogError("流速统计计算失败");
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "计算流速统计失败");
                return null;
            }
        }

        public void Dispose()
        {
            try
            {
                var tempFiles = Directory.GetFiles(_workingDirectory, "*.json");
                foreach (var file in tempFiles)
                    File.Delete(file);
            }
            catch { }
        }

        #region 私有方法

        private async Task<JsonElement> ExecuteAnalysisAsync(object inputData)
        {
            var inputFile = await SaveJsonAsync(inputData, "analysis_input");
            var outputFile = await ExecutePythonScriptAsync(inputFile);
            return await ReadJsonAsync(outputFile);
        }

        private async Task<string> ExecutePythonScriptAsync(string inputFile)
        {
            var scriptPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
            var outputFile = Path.Combine(_workingDirectory, $"output_{Guid.NewGuid():N}.json");

            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = _pythonExecutablePath,
                    Arguments = $"\"{scriptPath}\" \"{inputFile}\" \"{outputFile}\"",
                    WorkingDirectory = _pythonEngineRootPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            if (process.ExitCode != 0)
                throw new InvalidOperationException($"Python脚本执行失败:\nSTDOUT: {stdout}\nSTDERR: {stderr}");

            return outputFile;
        }

        private async Task<string> SaveJsonAsync(object data, string prefix)
        {
            var file = Path.Combine(_workingDirectory, $"{prefix}_{Guid.NewGuid():N}.json");
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            await File.WriteAllTextAsync(file, json);
            return file;
        }

        private async Task<JsonElement> ReadJsonAsync(string outputFile)
        {
            var json = await File.ReadAllTextAsync(outputFile);
            var document = JsonDocument.Parse(json);
            var result = document.RootElement;
            File.Delete(outputFile);
            return result;
        }

        private T GetConfigValue<T>(Dictionary<string, object> config, string key, T defaultValue)
        {
            return config.TryGetValue(key, out var value) && value is T typedValue
                   ? typedValue
                   : defaultValue;
        }

        #endregion
    }


}
