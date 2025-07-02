// =====================================
// 文件: OceanDataInterface.cs
// 功能: 基于data_processor.py的海洋数据接口
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 海洋数据处理接口 - 调用Python data_processor.py
    /// </summary>
    public class OceanDataInterface : IDisposable
    {
        private readonly ILogger<OceanDataInterface> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private readonly Dictionary<string, object> _configuration;
        private bool _isInitialized = false;

        public OceanDataInterface(ILogger<OceanDataInterface> logger,
                                 Dictionary<string, object> configuration)
        {
            _logger = logger;
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // 从配置中获取路径
            _pythonExecutablePath = GetConfigValue<string>("PythonExecutablePath", GetDefaultPythonPath());
            _pythonEngineRootPath = GetConfigValue<string>("PythonEngineRootPath", "../../../PythonEngine");
            _workingDirectory = GetConfigValue<string>("WorkingDirectory", "./OceanData_Output");

            EnsureDirectoriesExist();
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("正在初始化海洋数据处理接口...");

                // 检查Python环境
                if (!await CheckPythonEnvironmentAsync())
                {
                    _logger.LogError("Python环境检查失败");
                    return false;
                }

                // 检查Python模块
                if (!await CheckRequiredModulesAsync())
                {
                    _logger.LogError("必需的Python模块检查失败");
                    return false;
                }

                // 检查data_processor.py是否存在
                var dataProcessorPath = Path.Combine(_pythonEngineRootPath, "core", "data_processor.py");
                if (!File.Exists(dataProcessorPath))
                {
                    _logger.LogError($"未找到data_processor.py: {dataProcessorPath}");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("海洋数据处理接口初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "海洋数据处理接口初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 加载并处理NetCDF文件
        /// </summary>
        public async Task<OceanDataLoadResult> LoadNetCDFDataAsync(string ncFilePath, NetCDFLoadParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation($"正在加载NetCDF文件: {ncFilePath}");

                if (!File.Exists(ncFilePath))
                {
                    throw new FileNotFoundException($"NetCDF文件不存在: {ncFilePath}");
                }

                var inputData = new
                {
                    action = "load_netcdf",
                    netcdf_path = Path.GetFullPath(ncFilePath),
                    parameters = new
                    {
                        time_idx = parameters.TimeIndex,
                        depth_idx = parameters.DepthIndex,
                        lon_min = parameters.LonMin,
                        lon_max = parameters.LonMax,
                        lat_min = parameters.LatMin,
                        lat_max = parameters.LatMax
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "load_netcdf");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<OceanDataLoadResult>(outputFile);

                _logger.LogInformation($"NetCDF文件加载完成: {result.Message}");
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "加载NetCDF文件失败");
                return new OceanDataLoadResult
                {
                    Success = false,
                    Message = ex.Message,
                    DataInfo = new Dictionary<string, object>()
                };
            }
        }

        /// <summary>
        /// 生成专业的海流矢量场可视化
        /// </summary>
        public async Task<VisualizationResult> GenerateVectorFieldAsync(OceanDataSet dataSet, VectorFieldParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成海流矢量场可视化...");

                var outputPath = Path.Combine(_workingDirectory, $"vector_field_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var inputData = new
                {
                    action = "plot_vector_field",
                    data = new
                    {
                        u = dataSet.U,
                        v = dataSet.V,
                        lat = dataSet.Latitude,
                        lon = dataSet.Longitude,
                        depth = dataSet.Depth,
                        time_info = dataSet.TimeInfo
                    },
                    parameters = new
                    {
                        skip = parameters.Skip,
                        show = false, // 不显示，只保存
                        save_path = outputPath,
                        lon_min = parameters.LonMin,
                        lon_max = parameters.LonMax,
                        lat_min = parameters.LatMin,
                        lat_max = parameters.LatMax,
                        contourf_levels = parameters.ContourLevels,
                        contourf_cmap = parameters.ColorMap,
                        quiver_scale = parameters.QuiverScale,
                        quiver_width = parameters.QuiverWidth,
                        font_size = parameters.FontSize,
                        dpi = parameters.DPI
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "vector_field");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<VisualizationResult>(outputFile);

                if (result.Success && File.Exists(outputPath))
                {
                    result.ImagePath = outputPath;
                    _logger.LogInformation($"矢量场可视化生成成功: {outputPath}");
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成矢量场可视化失败");
                return new VisualizationResult
                {
                    Success = false,
                    Message = ex.Message,
                    ImagePath = ""
                };
            }
        }

        /// <summary>
        /// 导出矢量场为Shapefile
        /// </summary>
        public async Task<ExportResult> ExportVectorShapefileAsync(OceanDataSet dataSet, ShapefileExportParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在导出矢量场为Shapefile...");

                var outputPath = Path.Combine(_workingDirectory, $"vector_export_{DateTime.Now:yyyyMMdd_HHmmss}");

                var inputData = new
                {
                    action = "export_vector_shapefile",
                    data = new
                    {
                        u = dataSet.U,
                        v = dataSet.V,
                        lat = dataSet.Latitude,
                        lon = dataSet.Longitude
                    },
                    parameters = new
                    {
                        out_path = outputPath,
                        skip = parameters.Skip,
                        file_type = parameters.FileType // "shp" 或 "geojson"
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "export_shapefile");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<ExportResult>(outputFile);

                _logger.LogInformation($"Shapefile导出完成: {result.Message}");
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "导出Shapefile失败");
                return new ExportResult
                {
                    Success = false,
                    Message = ex.Message,
                    OutputPath = ""
                };
            }
        }

        /// <summary>
        /// 获取处理后的数据统计信息
        /// </summary>
        public async Task<DataStatistics> GetDataStatisticsAsync(OceanDataSet dataSet)
        {
            EnsureInitialized();

            try
            {
                var inputData = new
                {
                    action = "get_statistics",
                    data = new
                    {
                        u = dataSet.U,
                        v = dataSet.V,
                        lat = dataSet.Latitude,
                        lon = dataSet.Longitude
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "statistics");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<DataStatistics>(outputFile);
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "获取数据统计失败");
                return new DataStatistics
                {
                    Success = false,
                    Message = ex.Message
                };
            }
        }

        public void Dispose()
        {
            _logger.LogInformation("海洋数据处理接口正在关闭...");
            CleanupTempFiles();
        }

        #region 私有方法

        private string GetDefaultPythonPath()
        {
            var possiblePaths = new[]
            {
                "../../../PythonEngine/venv_oceansim/Scripts/python.exe",  // Windows
                "../../../PythonEngine/venv_oceansim/bin/python",          // Linux/Mac
                "python",
                "python3"
            };

            foreach (var path in possiblePaths)
            {
                try
                {
                    if (File.Exists(path))
                        return Path.GetFullPath(path);
                }
                catch { }
            }

            return "python";
        }

        private async Task<bool> CheckPythonEnvironmentAsync()
        {
            try
            {
                var result = await RunPythonCommandAsync("--version");
                if (result.Success)
                {
                    _logger.LogInformation($"Python环境检查通过: {result.Output}");
                    return true;
                }
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "检查Python环境失败");
                return false;
            }
        }

        private async Task<bool> CheckRequiredModulesAsync()
        {
            var requiredModules = new[]
            {
                "numpy", "matplotlib", "cartopy", "netCDF4",
                "geopandas", "shapely", "Source.PythonEngine.utils.chinese_config"
            };

            foreach (var module in requiredModules)
            {
                var result = await RunPythonCommandAsync($"-c \"import {module}; print('{module} OK')\"");
                if (!result.Success)
                {
                    _logger.LogError($"缺少必需模块: {module}");
                    return false;
                }
            }

            _logger.LogInformation("所有必需的Python模块检查通过");
            return true;
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

            return outputFile;
        }

        private async Task<PythonResult> RunPythonCommandAsync(string arguments)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = _pythonExecutablePath,
                        Arguments = arguments,
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

                return new PythonResult
                {
                    Success = process.ExitCode == 0,
                    Output = output.Trim(),
                    Error = error.Trim()
                };
            }
            catch (Exception ex)
            {
                return new PythonResult
                {
                    Success = false,
                    Output = "",
                    Error = ex.Message
                };
            }
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
        }

        private void EnsureInitialized()
        {
            if (!_isInitialized)
                throw new InvalidOperationException("海洋数据处理接口尚未初始化，请先调用InitializeAsync()");
        }

        private void CleanupTempFiles()
        {
            try
            {
                var tempFiles = Directory.GetFiles(_workingDirectory, "*.json");
                foreach (var file in tempFiles)
                {
                    File.Delete(file);
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

    #region 数据模型定义

    /// <summary>
    /// 海洋数据集
    /// </summary>
    public class OceanDataSet
    {
        public double[,] U { get; set; }          // 东向速度分量
        public double[,] V { get; set; }          // 北向速度分量
        public double[] Latitude { get; set; }    // 纬度数组
        public double[] Longitude { get; set; }   // 经度数组
        public double? Depth { get; set; }        // 深度
        public string TimeInfo { get; set; }      // 时间信息
    }

    /// <summary>
    /// NetCDF加载参数
    /// </summary>
    public class NetCDFLoadParameters
    {
        public int TimeIndex { get; set; } = 0;
        public int DepthIndex { get; set; } = 0;
        public double? LonMin { get; set; }
        public double? LonMax { get; set; }
        public double? LatMin { get; set; }
        public double? LatMax { get; set; }
    }

    /// <summary>
    /// 矢量场可视化参数
    /// </summary>
    public class VectorFieldParameters
    {
        public int Skip { get; set; } = 3;                    // 矢量跳过间隔
        public double? LonMin { get; set; }                   // 经度最小值
        public double? LonMax { get; set; }                   // 经度最大值
        public double? LatMin { get; set; }                   // 纬度最小值
        public double? LatMax { get; set; }                   // 纬度最大值
        public int ContourLevels { get; set; } = 100;         // 等值线级数
        public string ColorMap { get; set; } = "coolwarm";    // 颜色映射
        public double QuiverScale { get; set; } = 30.0;       // 箭头缩放
        public double QuiverWidth { get; set; } = 0.001;      // 箭头宽度
        public int FontSize { get; set; } = 14;               // 字体大小
        public int DPI { get; set; } = 120;                   // 分辨率
    }

    /// <summary>
    /// Shapefile导出参数
    /// </summary>
    public class ShapefileExportParameters
    {
        public int Skip { get; set; } = 5;                    // 数据跳过间隔
        public string FileType { get; set; } = "shp";        // 文件类型: "shp" 或 "geojson"
    }

    /// <summary>
    /// 数据加载结果
    /// </summary>
    public class OceanDataLoadResult
    {
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> DataInfo { get; set; }
        public OceanDataSet DataSet { get; set; }
    }

    /// <summary>
    /// 可视化结果
    /// </summary>
    public class VisualizationResult
    {
        public bool Success { get; set; }
        public string Message { get; set; }
        public string ImagePath { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// 导出结果
    /// </summary>
    public class ExportResult
    {
        public bool Success { get; set; }
        public string Message { get; set; }
        public string OutputPath { get; set; }
    }

    /// <summary>
    /// 数据统计信息
    /// </summary>
    public class DataStatistics
    {
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> Statistics { get; set; } = new();
    }

    /// <summary>
    /// Python执行结果
    /// </summary>
    public class PythonResult
    {
        public bool Success { get; set; }
        public string Output { get; set; } = "";
        public string Error { get; set; } = "";
    }

    #endregion
}
