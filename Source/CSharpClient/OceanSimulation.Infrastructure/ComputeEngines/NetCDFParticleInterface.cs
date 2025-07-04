// =====================================
// 文件: NetCDFParticleInterface.cs
// 功能: NetCDF粒子追踪C#接口
// =====================================

using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Diagnostics;


namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// NetCDF粒子追踪接口 - 基于您提供的台湾海峡案例
    /// </summary>
    public class NetCDFParticleInterface : IDisposable
    {
        private readonly ILogger<NetCDFParticleInterface> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private bool _isInitialized = false;

        public NetCDFParticleInterface(ILogger<NetCDFParticleInterface> logger,
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
        /// 初始化NetCDF粒子追踪环境
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("初始化NetCDF粒子追踪环境");

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

                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "netcdf_particle_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到netcdf_particle_wrapper.py: {wrapperPath}");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("NetCDF粒子追踪环境初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "NetCDF粒子追踪环境初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 追踪单个粒子轨迹
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="startPosition">起始位置（纬度，经度）</param>
        /// <param name="config">追踪配置</param>
        /// <returns>单粒子追踪结果</returns>
        public async Task<SingleParticleResult?> TrackSingleParticleAsync(
            string netcdfPath,
            (double latitude, double longitude) startPosition,
            ParticleTrackingConfig config)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化，请先调用InitializeAsync方法");

            try
            {
                _logger.LogInformation($"开始单粒子追踪: 起点({startPosition.latitude:F3}, {startPosition.longitude:F3})");

                if (!File.Exists(netcdfPath))
                    throw new FileNotFoundException($"NetCDF文件不存在: {netcdfPath}");

                var inputData = new
                {
                    action = "netcdf_particle_tracking",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        action = "track_single",
                        start_lat = startPosition.latitude,
                        start_lon = startPosition.longitude,
                        time_step_hours = config.TimeStepHours,
                        max_time_steps = config.MaxTimeSteps,
                        depth_level = config.DepthLevel,
                        lon_range = config.LonRange,
                        lat_range = config.LatRange
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean())
                {
                    var particleResult = ParseSingleParticleResult(result);
                    particleResult.Success = true;

                    _logger.LogInformation($"单粒子追踪成功完成，轨迹点数: {particleResult.TrajectoryPoints}");
                    return particleResult;
                }

                var errorMsg = GetErrorMessage(result);
                _logger.LogError($"单粒子追踪失败: {errorMsg}");
                return new SingleParticleResult { Success = false, ErrorMessage = errorMsg };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "执行单粒子追踪时发生异常");
                return new SingleParticleResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 追踪多个粒子轨迹
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="startPositions">起始位置列表</param>
        /// <param name="config">追踪配置</param>
        /// <returns>多粒子追踪结果</returns>
        public async Task<MultipleParticleResult?> TrackMultipleParticlesAsync(
            string netcdfPath,
            List<(double latitude, double longitude)> startPositions,
            ParticleTrackingConfig config)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化");

            try
            {
                _logger.LogInformation($"开始多粒子追踪: {startPositions.Count} 个粒子");

                var positionsList = startPositions.Select(pos => new[] { pos.latitude, pos.longitude }).ToList();

                var inputData = new
                {
                    action = "netcdf_particle_tracking",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        action = "track_multiple",
                        start_positions = positionsList,
                        time_step_hours = config.TimeStepHours,
                        max_time_steps = config.MaxTimeSteps,
                        depth_level = config.DepthLevel,
                        lon_range = config.LonRange,
                        lat_range = config.LatRange
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean())
                {
                    var multiResult = ParseMultipleParticleResult(result);
                    multiResult.Success = true;

                    _logger.LogInformation($"多粒子追踪完成: {multiResult.SuccessfulParticles}/{multiResult.TotalParticles}");
                    return multiResult;
                }

                var errorMsg = GetErrorMessage(result);
                _logger.LogError($"多粒子追踪失败: {errorMsg}");
                return new MultipleParticleResult { Success = false, ErrorMessage = errorMsg };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "执行多粒子追踪时发生异常");
                return new MultipleParticleResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 创建轨迹可视化图像
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="trajectories">轨迹数据</param>
        /// <param name="outputPath">输出图像路径</param>
        /// <param name="title">图像标题</param>
        /// <returns>可视化创建结果</returns>
        public async Task<VisualizationResult?> CreateTrajectoryVisualizationAsync(
            string netcdfPath,
            object trajectories,
            string? outputPath = null,
            string title = "粒子轨迹可视化")
        {
            if (!_isInitialized)
                throw new InvalidOperationException("环境尚未初始化");

            try
            {
                outputPath ??= Path.Combine(_workingDirectory,
                    $"particle_visualization_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var inputData = new
                {
                    action = "netcdf_particle_tracking",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        action = "create_animation",
                        trajectories = trajectories,
                        output_path = outputPath,
                        title = title
                    }
                };

                var result = await ExecuteAnalysisAsync(inputData);

                if (result.GetProperty("success").GetBoolean())
                {
                    var vizResult = new VisualizationResult
                    {
                        Success = true,
                        OutputPath = result.GetProperty("output_path").GetString() ?? "",
                        Title = title
                    };

                    if (result.TryGetProperty("animation_info", out var animInfo))
                    {
                        vizResult.ParticleCount = animInfo.GetProperty("particle_count").GetInt32();

                        if (animInfo.TryGetProperty("geographic_extent", out var extent))
                        {
                            var extentArray = extent.EnumerateArray().Select(e => e.GetDouble()).ToArray();
                            if (extentArray.Length >= 4)
                            {
                                vizResult.GeographicExtent = new GeographicBounds
                                {
                                    LonMin = extentArray[0],
                                    LonMax = extentArray[1],
                                    LatMin = extentArray[2],
                                    LatMax = extentArray[3]
                                };
                            }
                        }
                    }

                    _logger.LogInformation($"轨迹可视化创建成功: {vizResult.OutputPath}");
                    return vizResult;
                }

                var errorMsg = GetErrorMessage(result);
                _logger.LogError($"轨迹可视化创建失败: {errorMsg}");
                return new VisualizationResult { Success = false, ErrorMessage = errorMsg };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "创建轨迹可视化时发生异常");
                return new VisualizationResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 执行台湾海峡粒子漂移预设场景
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="startPosition">起始位置（可选，默认台湾海峡中部）</param>
        /// <param name="simulationDays">模拟天数</param>
        /// <returns>台湾海峡漂移结果</returns>
        public async Task<TaiwanStraitResult?> ExecuteTaiwanStraitScenarioAsync(
            string netcdfPath,
            (double latitude, double longitude)? startPosition = null,
            int simulationDays = 30)
        {
            try
            {
                // 默认台湾海峡中部位置
                var position = startPosition ?? (22.5, 119.5);

                _logger.LogInformation($"执行台湾海峡粒子漂移场景: {simulationDays} 天");

                var config = new ParticleTrackingConfig
                {
                    TimeStepHours = 3.0,
                    MaxTimeSteps = simulationDays * 8, // 每天8个时间步（3小时间隔）
                    DepthLevel = 0, // 表层
                    LonRange = new[] { 118.0, 124.0 }, // 台湾海峡经度范围
                    LatRange = new[] { 21.0, 26.5 }    // 台湾海峡纬度范围
                };

                var result = await TrackSingleParticleAsync(netcdfPath, position, config);

                if (result?.Success == true)
                {
                    var taiwanResult = new TaiwanStraitResult
                    {
                        Success = true,
                        StartPosition = new GeoPosition { Latitude = position.latitude, Longitude = position.longitude },
                        EndPosition = new GeoPosition { Latitude = result.EndPosition.Latitude, Longitude = result.EndPosition.Longitude },
                        TotalDistanceKm = result.Statistics.TotalDistanceKm,
                        DirectDistanceKm = result.Statistics.DirectDistanceKm,
                        SimulationDays = simulationDays,
                        TrajectoryPoints = result.TrajectoryPoints,
                        AverageSpeedMs = result.Statistics.AverageSpeedMs,
                        MaxSpeedMs = result.Statistics.MaxSpeedMs,
                        Trajectory = result.Trajectory,
                        DriftDirection = CalculateDriftDirection(result.StartPosition, result.EndPosition)
                    };

                    _logger.LogInformation($"台湾海峡场景完成: 漂移距离 {taiwanResult.TotalDistanceKm:F2} km");
                    return taiwanResult;
                }

                return new TaiwanStraitResult { Success = false, ErrorMessage = result?.ErrorMessage ?? "未知错误" };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "执行台湾海峡场景时发生异常");
                return new TaiwanStraitResult { Success = false, ErrorMessage = ex.Message };
            }
        }

        /// <summary>
        /// 批量执行多个起始点的台湾海峡场景
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="startPositions">起始位置列表</param>
        /// <param name="simulationDays">模拟天数</param>
        /// <returns>批量台湾海峡结果</returns>
        public async Task<List<TaiwanStraitResult>> ExecuteBatchTaiwanStraitScenariosAsync(
            string netcdfPath,
            List<(double latitude, double longitude)> startPositions,
            int simulationDays = 15)
        {
            var results = new List<TaiwanStraitResult>();

            foreach (var position in startPositions)
            {
                try
                {
                    var result = await ExecuteTaiwanStraitScenarioAsync(netcdfPath, position, simulationDays);
                    if (result != null)
                    {
                        results.Add(result);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, $"位置 ({position.latitude}, {position.longitude}) 的场景执行失败");
                    results.Add(new TaiwanStraitResult
                    {
                        Success = false,
                        ErrorMessage = ex.Message,
                        StartPosition = new GeoPosition { Latitude = position.latitude, Longitude = position.longitude }
                    });
                }
            }

            _logger.LogInformation($"批量台湾海峡场景完成: {results.Count(r => r.Success)}/{startPositions.Count} 成功");
            return results;
        }

        #region 私有方法

        private async Task<JsonElement> ExecuteAnalysisAsync(object inputData)
        {
            var inputFile = await SaveJsonAsync(inputData, "netcdf_particle_input");
            var outputFile = await ExecutePythonScriptAsync(inputFile);
            return await ReadJsonAsync(outputFile);
        }

        private async Task<string> ExecutePythonScriptAsync(string inputFile)
        {
            var scriptPath = Path.Combine(_pythonEngineRootPath, "wrappers", "netcdf_particle_wrapper.py");
            var outputFile = Path.Combine(_workingDirectory, $"output_{Guid.NewGuid():N}.json");

            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = _pythonExecutablePath,
                    Arguments = $"-c \"import sys; sys.path.append('{_pythonEngineRootPath}'); " +
                              $"from wrappers.netcdf_particle_wrapper import handle_netcdf_particle_request; " +
                              $"import json; " +
                              $"with open('{inputFile}', 'r') as f: data = json.load(f); " +
                              $"result = handle_netcdf_particle_request(data); " +
                              $"with open('{outputFile}', 'w') as f: json.dump(result, f)\"",
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

        private SingleParticleResult ParseSingleParticleResult(JsonElement result)
        {
            var trajectory = result.GetProperty("trajectory");
            var statistics = result.GetProperty("statistics");
            var startPos = result.GetProperty("start_position");
            var endPos = result.GetProperty("end_position");

            return new SingleParticleResult
            {
                Trajectory = new ParticleTrajectory
                {
                    Latitudes = trajectory.GetProperty("latitudes").EnumerateArray()
                        .Select(x => x.GetDouble()).ToList(),
                    Longitudes = trajectory.GetProperty("longitudes").EnumerateArray()
                        .Select(x => x.GetDouble()).ToList(),
                    Times = trajectory.GetProperty("times").EnumerateArray()
                        .Select(x => x.GetString() ?? "").ToList()
                },
                Statistics = new ParticleStatistics
                {
                    TotalPoints = statistics.GetProperty("total_points").GetInt32(),
                    TotalDistanceKm = statistics.GetProperty("total_distance_km").GetDouble(),
                    DirectDistanceKm = statistics.GetProperty("direct_distance_km").GetDouble(),
                    AverageSpeedMs = statistics.GetProperty("avg_speed_ms").GetDouble(),
                    MaxSpeedMs = statistics.GetProperty("max_speed_ms").GetDouble(),
                    SimulationHours = statistics.GetProperty("simulation_hours").GetDouble()
                },
                StartPosition = new GeoPosition
                {
                    Latitude = startPos.GetProperty("lat").GetDouble(),
                    Longitude = startPos.GetProperty("lon").GetDouble()
                },
                EndPosition = new GeoPosition
                {
                    Latitude = endPos.GetProperty("lat").GetDouble(),
                    Longitude = endPos.GetProperty("lon").GetDouble()
                }
            };
        }

        private MultipleParticleResult ParseMultipleParticleResult(JsonElement result)
        {
            var summary = result.GetProperty("summary");
            var trajectories = result.GetProperty("trajectories");

            var particleResults = new List<SingleParticleResult?>();

            foreach (var traj in trajectories.EnumerateArray())
            {
                if (traj.ValueKind == JsonValueKind.Null)
                {
                    particleResults.Add(null);
                }
                else
                {
                    // 这里需要根据实际的多粒子结果结构来解析
                    particleResults.Add(null); // 简化处理
                }
            }

            return new MultipleParticleResult
            {
                TotalParticles = summary.GetProperty("total_particles").GetInt32(),
                SuccessfulParticles = summary.GetProperty("successful_particles").GetInt32(),
                FailedParticles = summary.GetProperty("failed_particles").GetInt32(),
                ParticleResults = particleResults
            };
        }

        private string CalculateDriftDirection(GeoPosition start, GeoPosition end)
        {
            var deltaLat = end.Latitude - start.Latitude;
            var deltaLon = end.Longitude - start.Longitude;

            if (Math.Abs(deltaLat) > Math.Abs(deltaLon))
            {
                return deltaLat > 0 ? "北向" : "南向";
            }
            else
            {
                return deltaLon > 0 ? "东向" : "西向";
            }
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

        public void Dispose()
        {
            _logger.LogInformation("清理NetCDF粒子追踪环境资源");
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
    }

    #region 数据结构定义

    /// <summary>
    /// 粒子追踪配置
    /// </summary>
    public class ParticleTrackingConfig
    {
        public double TimeStepHours { get; set; } = 3.0;
        public int? MaxTimeSteps { get; set; }
        public int DepthLevel { get; set; } = 0;
        public double[] LonRange { get; set; } = { 118.0, 124.0 };
        public double[] LatRange { get; set; } = { 21.0, 26.5 };
    }

    /// <summary>
    /// 单粒子追踪结果
    /// </summary>
    public class SingleParticleResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public ParticleTrajectory Trajectory { get; set; } = new();
        public ParticleStatistics Statistics { get; set; } = new();
        public GeoPosition StartPosition { get; set; } = new();
        public GeoPosition EndPosition { get; set; } = new();
        public int TrajectoryPoints => Trajectory.Latitudes.Count;
    }

    /// <summary>
    /// 多粒子追踪结果
    /// </summary>
    public class MultipleParticleResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public int TotalParticles { get; set; }
        public int SuccessfulParticles { get; set; }
        public int FailedParticles { get; set; }
        public List<SingleParticleResult?> ParticleResults { get; set; } = new();
    }

    /// <summary>
    /// 粒子轨迹数据
    /// </summary>
    public class ParticleTrajectory
    {
        public List<double> Latitudes { get; set; } = new();
        public List<double> Longitudes { get; set; } = new();
        public List<string> Times { get; set; } = new();
    }

    /// <summary>
    /// 粒子统计信息
    /// </summary>
    public class ParticleStatistics
    {
        public int TotalPoints { get; set; }
        public double TotalDistanceKm { get; set; }
        public double DirectDistanceKm { get; set; }
        public double AverageSpeedMs { get; set; }
        public double MaxSpeedMs { get; set; }
        public double SimulationHours { get; set; }
    }

    /// <summary>
    /// 地理位置
    /// </summary>
    public class GeoPosition
    {
        public double Latitude { get; set; }
        public double Longitude { get; set; }
    }

    /// <summary>
    /// 可视化结果
    /// </summary>
    public class VisualizationResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public string OutputPath { get; set; } = "";
        public string Title { get; set; } = "";
        public int ParticleCount { get; set; }
        public GeographicBounds? GeographicExtent { get; set; }
    }

    /// <summary>
    /// 台湾海峡漂移结果
    /// </summary>
    public class TaiwanStraitResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public GeoPosition StartPosition { get; set; } = new();
        public GeoPosition EndPosition { get; set; } = new();
        public double TotalDistanceKm { get; set; }
        public double DirectDistanceKm { get; set; }
        public int SimulationDays { get; set; }
        public int TrajectoryPoints { get; set; }
        public double AverageSpeedMs { get; set; }
        public double MaxSpeedMs { get; set; }
        public string DriftDirection { get; set; } = "";
        public ParticleTrajectory? Trajectory { get; set; }
    }

    /// <summary>
    /// 地理边界
    /// </summary>
    public class GeographicBounds
    {
        public double LonMin { get; set; }
        public double LonMax { get; set; }
        public double LatMin { get; set; }
        public double LatMax { get; set; }
    }

    #endregion
}
