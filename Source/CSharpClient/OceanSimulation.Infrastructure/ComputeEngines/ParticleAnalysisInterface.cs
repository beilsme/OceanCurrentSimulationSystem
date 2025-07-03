// =====================================
// 文件: ParticleAnalysisInterface.cs
// 功能: 拉格朗日粒子追踪分析接口
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.ValueObjects;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 拉格朗日粒子追踪分析接口
    /// 提供粒子释放、轨迹追踪和结果分析的完整功能
    /// </summary>
    public class ParticleAnalysisInterface : IDisposable
    {
        private readonly ILogger<ParticleAnalysisInterface> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private bool _isInitialized = false;

        public ParticleAnalysisInterface(ILogger<ParticleAnalysisInterface> logger,
                                        Dictionary<string, object> configuration)
        {
            _logger = logger;
            var config = configuration ?? throw new ArgumentNullException(nameof(configuration));

            _pythonExecutablePath = GetConfigValue(config, "PythonExecutablePath", "python3");
            _pythonEngineRootPath = GetConfigValue(config, "PythonEngineRootPath", "../../../PythonEngine");
            _workingDirectory = Path.Combine(_pythonEngineRootPath, "Temp");

            Directory.CreateDirectory(_workingDirectory);
        }

        /// <summary>
        /// 初始化粒子分析环境
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("初始化拉格朗日粒子分析环境");

                var pythonCheck = await Task.Run(() =>
                {
                    try
                    {
                        var process = new Process
                        {
                            StartInfo = new ProcessStartInfo
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
                        process.WaitForExit(5000);
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
                _logger.LogInformation("粒子分析环境初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "粒子分析环境初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 执行拉格朗日粒子追踪模拟
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="config">粒子追踪配置</param>
        /// <returns>追踪结果包含轨迹数据和统计分析</returns>
        public async Task<ParticleTrackingResult?> ExecuteParticleTrackingAsync(
            string netcdfPath,
            ParticleTrackingConfig config)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化，请先调用InitializeAsync方法");

            try
            {
                _logger.LogInformation($"开始执行拉格朗日粒子追踪模拟，粒子数量: {config.ParticleCount}，时间步数: {config.TimeSteps}");

                if (!File.Exists(netcdfPath))
                    throw new FileNotFoundException($"NetCDF文件不存在: {netcdfPath}");

                var outputPath = config.OutputPath ?? Path.Combine(_workingDirectory,
                    $"particle_tracking_{DateTime.Now:yyyyMMdd_HHmmss}.gif");

                var inputData = new
                {
                    action = "lagrangian_particle_tracking",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        output_path = outputPath,
                        particle_count = config.ParticleCount,
                        time_steps = config.TimeSteps,
                        dt = config.TimeStepSize,
                        trail_length = config.TrailLength,
                        initial_positions = config.InitialPositionType,
                        bounds = config.Bounds
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean())
                {
                    var trackingResult = ParseTrackingResult(result);
                    trackingResult.Success = true;
                    trackingResult.OutputPath = result.GetProperty("output_path").GetString() ?? "";

                    _logger.LogInformation($"粒子追踪模拟成功完成，输出文件: {trackingResult.OutputPath}");
                    return trackingResult;
                }

                var errorMsg = GetErrorMessage(result);
                _logger.LogError($"粒子追踪模拟执行失败: {errorMsg}");
                return new ParticleTrackingResult { Success = false, ErrorMessage = errorMsg };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "执行粒子追踪模拟时发生异常");
                return new ParticleTrackingResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 分析预定义位置的粒子扩散模式
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="releasePositions">粒子释放位置列表</param>
        /// <param name="analysisConfig">分析配置参数</param>
        /// <returns>扩散分析结果</returns>
        public async Task<ParticleTrackingResult?> AnalyzeDispersionPatternsAsync(
            string netcdfPath,
            List<(double latitude, double longitude)> releasePositions,
            ParticleTrackingConfig analysisConfig)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化");

            try
            {
                _logger.LogInformation($"开始分析粒子扩散模式，释放点数量: {releasePositions.Count}");

                var positionsList = releasePositions.Select(pos => new[] { pos.latitude, pos.longitude }).ToList();

                var customConfig = new ParticleTrackingConfig
                {
                    ParticleCount = releasePositions.Count,
                    TimeSteps = analysisConfig.TimeSteps,
                    TimeStepSize = analysisConfig.TimeStepSize,
                    TrailLength = analysisConfig.TrailLength,
                    InitialPositionType = "custom",
                    OutputPath = analysisConfig.OutputPath ?? Path.Combine(_workingDirectory,
                        $"dispersion_analysis_{DateTime.Now:yyyyMMdd_HHmmss}.gif"),
                    Bounds = new Dictionary<string, object> { ["positions"] = positionsList }
                };

                return await ExecuteParticleTrackingAsync(netcdfPath, customConfig);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "分析粒子扩散模式时发生异常");
                return new ParticleTrackingResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 执行线性释放粒子追踪分析
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="startPosition">起始位置</param>
        /// <param name="endPosition">结束位置</param>
        /// <param name="particleCount">粒子数量</param>
        /// <param name="analysisConfig">分析配置</param>
        /// <returns>线性释放追踪结果</returns>
        public async Task<ParticleTrackingResult?> ExecuteLinearReleaseTrackingAsync(
            string netcdfPath,
            (double latitude, double longitude) startPosition,
            (double latitude, double longitude) endPosition,
            int particleCount,
            ParticleTrackingConfig analysisConfig)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化");

            try
            {
                _logger.LogInformation($"执行线性释放粒子追踪，从 ({startPosition.latitude:F3}, {startPosition.longitude:F3}) 到 ({endPosition.latitude:F3}, {endPosition.longitude:F3})");

                var linearConfig = new ParticleTrackingConfig
                {
                    ParticleCount = particleCount,
                    TimeSteps = analysisConfig.TimeSteps,
                    TimeStepSize = analysisConfig.TimeStepSize,
                    TrailLength = analysisConfig.TrailLength,
                    InitialPositionType = "line",
                    OutputPath = analysisConfig.OutputPath ?? Path.Combine(_workingDirectory,
                        $"linear_release_{DateTime.Now:yyyyMMdd_HHmmss}.gif"),
                    Bounds = new Dictionary<string, object>
                    {
                        ["start_lat"] = startPosition.latitude,
                        ["start_lon"] = startPosition.longitude,
                        ["end_lat"] = endPosition.latitude,
                        ["end_lon"] = endPosition.longitude
                    }
                };

                return await ExecuteParticleTrackingAsync(netcdfPath, linearConfig);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "执行线性释放粒子追踪时发生异常");
                return new ParticleTrackingResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 批量执行多时间段粒子追踪分析
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="baseConfig">基础配置</param>
        /// <param name="timeIntervals">时间间隔数组</param>
        /// <returns>多时间段追踪结果列表</returns>
        public async Task<List<ParticleTrackingResult>> BatchTimeIntervalAnalysisAsync(
            string netcdfPath,
            ParticleTrackingConfig baseConfig,
            int[] timeIntervals)
        {
            var results = new List<ParticleTrackingResult>();

            foreach (var interval in timeIntervals)
            {
                try
                {
                    var intervalConfig = new ParticleTrackingConfig
                    {
                        ParticleCount = baseConfig.ParticleCount,
                        TimeSteps = interval,
                        TimeStepSize = baseConfig.TimeStepSize,
                        TrailLength = baseConfig.TrailLength,
                        InitialPositionType = baseConfig.InitialPositionType,
                        OutputPath = Path.Combine(_workingDirectory,
                            $"batch_analysis_{interval}steps_{DateTime.Now:yyyyMMdd_HHmmss}.gif"),
                        Bounds = baseConfig.Bounds
                    };

                    var result = await ExecuteParticleTrackingAsync(netcdfPath, intervalConfig);
                    if (result != null)
                    {
                        results.Add(result);
                        _logger.LogInformation($"完成时间间隔 {interval} 步的分析");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, $"时间间隔 {interval} 步的分析执行失败");
                    results.Add(new ParticleTrackingResult { Success = false, ErrorMessage = ex.Message });
                }
            }

            _logger.LogInformation($"批量时间间隔分析完成，成功执行 {results.Count(r => r.Success)}/{timeIntervals.Length} 个分析");
            return results;
        }

        public void Dispose()
        {
            _logger.LogInformation("清理粒子分析环境资源");
            try
            {
                var tempFiles = Directory.GetFiles(_workingDirectory, "*.json")
                                        .Concat(Directory.GetFiles(_workingDirectory, "*.tmp"));
                foreach (var file in tempFiles)
                {
                    File.Delete(file);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "清理临时文件时发生异常");
            }
        }

        #region 私有方法

        private async Task<JsonElement> ExecuteAnalysisAsync(object inputData)
        {
            var inputFile = await SaveJsonAsync(inputData, "particle_analysis_input");
            var outputFile = await ExecutePythonScriptAsync(inputFile);
            return await ReadJsonAsync(outputFile);
        }

        private async Task<string> ExecutePythonScriptAsync(string inputFile)
        {
            var scriptPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
            var outputFile = Path.Combine(_workingDirectory, $"output_{Guid.NewGuid():N}.json");

            var process = new Process
            {
                StartInfo = new ProcessStartInfo
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
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                var errorMessage = $"Python脚本执行失败，退出码: {process.ExitCode}，错误信息: {error}，输出信息: {output}";
                _logger.LogError(errorMessage);
                throw new InvalidOperationException(errorMessage);
            }

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
            if (!File.Exists(outputFile))
                throw new FileNotFoundException($"输出文件不存在: {outputFile}");

            var json = await File.ReadAllTextAsync(outputFile);
            var document = JsonDocument.Parse(json);
            var result = document.RootElement;

            File.Delete(outputFile);
            return result;
        }

        private ParticleTrackingResult ParseTrackingResult(JsonElement result)
        {
            var statisticsElement = result.GetProperty("statistics");
            var metadataElement = result.GetProperty("metadata");

            return new ParticleTrackingResult
            {
                Statistics = JsonSerializer.Deserialize<ParticleTrackingStatistics>(statisticsElement.GetRawText()),
                Metadata = new ParticleTrackingMetadata
                {
                    ParticleCount = metadataElement.GetProperty("particle_count").GetInt32(),
                    TimeSteps = metadataElement.GetProperty("time_steps").GetInt32(),
                    SimulationDurationHours = metadataElement.GetProperty("simulation_duration_hours").GetDouble(),
                    TrailLength = metadataElement.GetProperty("trail_length").GetInt32()
                }
            };
        }

        private string GetErrorMessage(JsonElement result)
        {
            return result.TryGetProperty("message", out var msgElement)
                   ? msgElement.GetString() ?? "未知错误"
                   : "处理执行失败";
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
