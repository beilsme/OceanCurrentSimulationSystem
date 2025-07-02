// =====================================
// 文件: PythonEngineClient.cs
// 功能: Python引擎客户端 - 负责调用Python数据处理和可视化模块
// =====================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.Interfaces;
using OceanSimulation.Domain.ValueObjects;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// Python引擎客户端 - 调用Python的data_processor和rendering_engine
    /// </summary>
    public class PythonEngineClient : IPythonMLEngine
    {
        private readonly ILogger<PythonEngineClient> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private readonly Dictionary<string, object> _configuration;
        private bool _isInitialized = false;

        public PythonEngineClient(ILogger<PythonEngineClient> logger,
                                 Dictionary<string, object> configuration)
        {
            _logger = logger;
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // 从配置中获取路径
            _pythonExecutablePath = GetConfigValue<string>("PythonExecutablePath", "python");
            _pythonEngineRootPath = GetConfigValue<string>("PythonEngineRootPath", "./Source/PythonEngine");
            _workingDirectory = GetConfigValue<string>("WorkingDirectory", "./Data/Cache/PythonCache");

            EnsureDirectoriesExist();
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("正在初始化Python引擎客户端...");

                // 检查Python环境
                if (!await CheckPythonEnvironmentAsync())
                {
                    _logger.LogError("Python环境检查失败");
                    return false;
                }

                // 检查Python模块
                if (!await CheckPythonModulesAsync())
                {
                    _logger.LogError("Python模块检查失败");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("Python引擎客户端初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Python引擎客户端初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 生成基础海流场可视化
        /// </summary>
        public async Task<string> GenerateVectorFieldVisualizationAsync(
            VectorFieldData vectorField,
            VisualizationParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成海流场可视化...");

                // 准备输入数据
                var inputData = new
                {
                    u_data = vectorField.U,
                    v_data = vectorField.V,
                    lat_data = vectorField.Latitude,
                    lon_data = vectorField.Longitude,
                    depth = vectorField.Depth,
                    time_info = vectorField.TimeInfo,
                    visualization_params = new
                    {
                        skip = parameters.Skip,
                        lon_min = parameters.LonMin,
                        lon_max = parameters.LonMax,
                        lat_min = parameters.LatMin,
                        lat_max = parameters.LatMax,
                        font_size = parameters.FontSize,
                        dpi = parameters.DPI,
                        save_path = Path.Combine(_workingDirectory, $"vector_field_{DateTime.Now:yyyyMMdd_HHmmss}.png")
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "vector_field_input");
                var outputFile = await ExecutePythonScriptAsync("data_processor_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<VisualizationResult>(outputFile);

                _logger.LogInformation($"海流场可视化生成完成: {result.ImagePath}");
                return result.ImagePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成海流场可视化失败");
                throw;
            }
        }

        /// <summary>
        /// 生成3D海洋数据可视化
        /// </summary>
        public async Task<string> Generate3DVelocityVisualizationAsync(
            VelocityField3D velocityField,
            Visualization3DParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成3D海洋数据可视化...");

                var inputData = new
                {
                    velocity_data = new
                    {
                        u = velocityField.U,
                        v = velocityField.V,
                        w = velocityField.W
                    },
                    grid_config = new
                    {
                        nx = velocityField.NX,
                        ny = velocityField.NY,
                        nz = velocityField.NZ,
                        dx = velocityField.DX,
                        dy = velocityField.DY,
                        dz = velocityField.DZ
                    },
                    render_params = new
                    {
                        title = parameters.Title,
                        slice_positions = parameters.SlicePositions,
                        save_path = Path.Combine(_workingDirectory, $"3d_velocity_{DateTime.Now:yyyyMMdd_HHmmss}.png")
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "3d_velocity_input");
                var outputFile = await ExecutePythonScriptAsync("rendering_engine_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<VisualizationResult>(outputFile);

                _logger.LogInformation($"3D可视化生成完成: {result.ImagePath}");
                return result.ImagePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成3D可视化失败");
                throw;
            }
        }

        /// <summary>
        /// 生成综合仪表板
        /// </summary>
        public async Task<string> GenerateComprehensiveDashboardAsync(
            DashboardData dashboardData,
            DashboardParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成综合仪表板...");

                var inputData = new
                {
                    velocity_data = dashboardData.VelocityData,
                    concentration_data = dashboardData.ConcentrationData,
                    particle_positions = dashboardData.ParticlePositions,
                    time_info = dashboardData.TimeInfo,
                    statistics = dashboardData.Statistics,
                    dashboard_params = new
                    {
                        title = parameters.Title,
                        save_path = Path.Combine(_workingDirectory, $"dashboard_{DateTime.Now:yyyyMMdd_HHmmss}.png")
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "dashboard_input");
                var outputFile = await ExecutePythonScriptAsync("dashboard_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<VisualizationResult>(outputFile);

                _logger.LogInformation($"综合仪表板生成完成: {result.ImagePath}");
                return result.ImagePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成综合仪表板失败");
                throw;
            }
        }

        /// <summary>
        /// 处理NetCDF数据
        /// </summary>
        public async Task<OceanDataResult> ProcessNetCDFDataAsync(string filePath, DataProcessingParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation($"正在处理NetCDF数据: {filePath}");

                var inputData = new
                {
                    netcdf_path = filePath,
                    processing_params = new
                    {
                        time_idx = parameters.TimeIndex,
                        depth_idx = parameters.DepthIndex,
                        lon_min = parameters.LonMin,
                        lon_max = parameters.LonMax,
                        lat_min = parameters.LatMin,
                        lat_max = parameters.LatMax
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "netcdf_processing_input");
                var outputFile = await ExecutePythonScriptAsync("netcdf_processor_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<OceanDataResult>(outputFile);

                _logger.LogInformation("NetCDF数据处理完成");
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "处理NetCDF数据失败");
                throw;
            }
        }

        /// <summary>
        /// 获取性能指标
        /// </summary>
        public async Task<Dictionary<string, object>> GetPerformanceMetricsAsync()
        {
            EnsureInitialized();

            try
            {
                var inputData = new { request_type = "performance_metrics" };
                var inputFile = await SaveInputDataAsync(inputData, "performance_input");
                var outputFile = await ExecutePythonScriptAsync("performance_monitor.py", inputFile);

                var result = await ReadOutputDataAsync<Dictionary<string, object>>(outputFile);
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "获取性能指标失败");
                return new Dictionary<string, object> { ["error"] = ex.Message };
            }
        }

        public void Dispose()
        {
            _logger.LogInformation("Python引擎客户端正在关闭...");
            // 清理临时文件
            CleanupTempFiles();
        }

        #region 私有辅助方法

        private async Task<bool> CheckPythonEnvironmentAsync()
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
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    _logger.LogInformation($"Python环境检查通过: {output.Trim()}");
                    return true;
                }
                else
                {
                    _logger.LogError($"Python环境检查失败, 退出码: {process.ExitCode}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "检查Python环境时发生异常");
                return false;
            }
        }

        private async Task<bool> CheckPythonModulesAsync()
        {
            try
            {
                var requiredModules = new[]
                {
                    "numpy", "matplotlib", "cartopy", "netCDF4",
                    "geopandas", "shapely", "scipy"
                };

                foreach (var module in requiredModules)
                {
                    var process = new Process
                    {
                        StartInfo = new ProcessStartInfo
                        {
                            FileName = _pythonExecutablePath,
                            Arguments = $"-c \"import {module}; print('{module} OK')\"",
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };

                    process.Start();
                    await process.WaitForExitAsync();

                    if (process.ExitCode != 0)
                    {
                        var error = await process.StandardError.ReadToEndAsync();
                        _logger.LogError($"Python模块 {module} 检查失败: {error}");
                        return false;
                    }
                }

                _logger.LogInformation("所有必需的Python模块检查通过");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "检查Python模块时发生异常");
                return false;
            }
        }

        private async Task<string> ExecutePythonScriptAsync(string scriptName, string inputFile)
        {
            var scriptPath = Path.Combine(_pythonEngineRootPath, "wrappers", scriptName);
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

            _logger.LogDebug($"执行Python脚本: {scriptPath}");

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                var errorMessage = $"Python脚本执行失败 (退出码: {process.ExitCode})\n错误信息: {error}\n输出: {output}";
                _logger.LogError(errorMessage);
                throw new InvalidOperationException(errorMessage);
            }

            if (!string.IsNullOrWhiteSpace(error))
            {
                _logger.LogWarning($"Python脚本警告信息: {error}");
            }

            _logger.LogDebug($"Python脚本执行成功: {output}");
            return outputFile;
        }

        private async Task<string> SaveInputDataAsync(object data, string prefix)
        {
            var inputFile = Path.Combine(_workingDirectory, $"{prefix}_{Guid.NewGuid():N}.json");
            var jsonString = JsonSerializer.Serialize(data, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });

            await File.WriteAllTextAsync(inputFile, jsonString);
            return inputFile;
        }

        private async Task<T> ReadOutputDataAsync<T>(string outputFile)
        {
            if (!File.Exists(outputFile))
            {
                throw new FileNotFoundException($"输出文件不存在: {outputFile}");
            }

            var jsonString = await File.ReadAllTextAsync(outputFile);
            var result = JsonSerializer.Deserialize<T>(jsonString, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            // 清理输出文件
            File.Delete(outputFile);

            return result;
        }

        private void EnsureDirectoriesExist()
        {
            Directory.CreateDirectory(_workingDirectory);
            Directory.CreateDirectory(Path.Combine(_workingDirectory, "temp"));
        }

        private void EnsureInitialized()
        {
            if (!_isInitialized)
                throw new InvalidOperationException("Python引擎客户端尚未初始化，请先调用InitializeAsync()");
        }

        private void CleanupTempFiles()
        {
            try
            {
                var tempDir = Path.Combine(_workingDirectory, "temp");
                if (Directory.Exists(tempDir))
                {
                    Directory.Delete(tempDir, true);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "清理临时文件失败");
            }
        }

        private T GetConfigValue<T>(string key, T defaultValue)
        {
            if (_configuration.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }
            return defaultValue;
        }

        #endregion
    }

    #region 数据传输对象定义

    /// <summary>
    /// 矢量场数据
    /// </summary>
    public class VectorFieldData
    {
        public double[,] U { get; set; }
        public double[,] V { get; set; }
        public double[] Latitude { get; set; }
        public double[] Longitude { get; set; }
        public double? Depth { get; set; }
        public string TimeInfo { get; set; }
    }

    /// <summary>
    /// 3D速度场数据
    /// </summary>
    public class VelocityField3D
    {
        public double[,,] U { get; set; }
        public double[,,] V { get; set; }
        public double[,,] W { get; set; }
        public int NX { get; set; }
        public int NY { get; set; }
        public int NZ { get; set; }
        public double DX { get; set; }
        public double DY { get; set; }
        public double DZ { get; set; }
    }

    /// <summary>
    /// 仪表板数据
    /// </summary>
    public class DashboardData
    {
        public object VelocityData { get; set; }
        public object ConcentrationData { get; set; }
        public double[,] ParticlePositions { get; set; }
        public Dictionary<string, object> TimeInfo { get; set; }
        public Dictionary<string, object> Statistics { get; set; }
    }

    /// <summary>
    /// 可视化参数
    /// </summary>
    public class VisualizationParameters
    {
        public int Skip { get; set; } = 3;
        public double? LonMin { get; set; }
        public double? LonMax { get; set; }
        public double? LatMin { get; set; }
        public double? LatMax { get; set; }
        public int FontSize { get; set; } = 14;
        public int DPI { get; set; } = 120;
    }

    /// <summary>
    /// 3D可视化参数
    /// </summary>
    public class Visualization3DParameters
    {
        public string Title { get; set; } = "3D海洋数据可视化";
        public Dictionary<string, int> SlicePositions { get; set; } = new();
    }

    /// <summary>
    /// 仪表板参数
    /// </summary>
    public class DashboardParameters
    {
        public string Title { get; set; } = "海洋模拟综合仪表板";
    }

    /// <summary>
    /// 数据处理参数
    /// </summary>
    public class DataProcessingParameters
    {
        public int TimeIndex { get; set; } = 0;
        public int DepthIndex { get; set; } = 0;
        public double? LonMin { get; set; }
        public double? LonMax { get; set; }
        public double? LatMin { get; set; }
        public double? LatMax { get; set; }
    }

    /// <summary>
    /// 可视化结果
    /// </summary>
    public class VisualizationResult
    {
        public string ImagePath { get; set; }
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> Metadata { get; set; }
    }

    /// <summary>
    /// 海洋数据结果
    /// </summary>
    public class OceanDataResult
    {
        public double[,] U { get; set; }
        public double[,] V { get; set; }
        public double[] Latitude { get; set; }
        public double[] Longitude { get; set; }
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> Metadata { get; set; }
    }

    #endregion
}
