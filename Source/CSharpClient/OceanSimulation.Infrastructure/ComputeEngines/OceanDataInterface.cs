// =====================================
// 文件: OceanDataInterface.cs
// 功能: 基于IPythonMLEngine接口的海洋数据处理实现
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.Interfaces;
using OceanSimulation.Domain.ValueObjects;
using OceanSimulation.Domain.Entities;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 海洋数据处理接口 - 实现IPythonMLEngine接口，调用Python ocean_data_wrapper.py
    /// </summary>
    public class OceanDataInterface : IPythonMLEngine, IDisposable
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

        #region IPythonMLEngine接口实现

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

                // 检查ocean_data_wrapper.py是否存在
                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到ocean_data_wrapper.py: {wrapperPath}");
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

        public async Task<string> GenerateVectorFieldVisualizationAsync(VectorFieldData vectorField, VisualizationParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成矢量场可视化...");

                var outputPath = Path.Combine(_workingDirectory, $"vector_field_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var inputData = new
                {
                    action = "plot_vector_field",
                    data = new
                    {
                        u = ConvertArray2DToJsonArray(vectorField.U ?? new double[0,0]),
                        v = ConvertArray2DToJsonArray(vectorField.V ?? new double[0,0]),
                        lat = vectorField.Latitude ?? Array.Empty<double>(),
                        lon = vectorField.Longitude ?? Array.Empty<double>(),
                        depth = vectorField.Depth,
                        time_info = vectorField.TimeInfo ?? ""
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
                        font_size = parameters.FontSize,
                        dpi = parameters.DPI
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "vector_field");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    _logger.LogInformation($"矢量场可视化生成成功: {outputPath}");
                    return outputPath;
                }

                var errorMessage = result.TryGetProperty("message", out var msgElement) ? msgElement.GetString() : "未知错误";
                _logger.LogError($"矢量场可视化生成失败: {errorMessage}");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成矢量场可视化失败");
                return "";
            }
        }

        public async Task<string> Generate3DVelocityVisualizationAsync(VelocityField3D velocityField, Visualization3DParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成3D速度场可视化...");

                var outputPath = Path.Combine(_workingDirectory, $"velocity_3d_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var inputData = new
                {
                    action = "plot_3d_velocity",
                    data = new
                    {
                        u = ConvertArray3DToJsonArray(velocityField.U ?? new double[0,0,0]),
                        v = ConvertArray3DToJsonArray(velocityField.V ?? new double[0,0,0]),
                        w = ConvertArray3DToJsonArray(velocityField.W ?? new double[0,0,0]),
                        nx = velocityField.NX,
                        ny = velocityField.NY,
                        nz = velocityField.NZ,
                        dx = velocityField.DX,
                        dy = velocityField.DY,
                        dz = velocityField.DZ
                    },
                    parameters = new
                    {
                        title = parameters.Title ?? "3D海洋数据可视化",
                        slice_positions = parameters.SlicePositions ?? new Dictionary<string, int>(),
                        save_path = outputPath,
                        show = false
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "velocity_3d");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    _logger.LogInformation($"3D速度场可视化生成成功: {outputPath}");
                    return outputPath;
                }

                var errorMessage = result.TryGetProperty("message", out var msgElement) ? msgElement.GetString() : "未知错误";
                _logger.LogError($"3D速度场可视化生成失败: {errorMessage}");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成3D速度场可视化失败");
                return "";
            }
        }

        public async Task<string> GenerateComprehensiveDashboardAsync(DashboardData dashboardData, DashboardParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation("正在生成综合仪表板...");

                var outputPath = Path.Combine(_workingDirectory, $"dashboard_{DateTime.Now:yyyyMMdd_HHmmss}.html");

                var inputData = new
                {
                    action = "generate_dashboard",
                    data = new
                    {
                        velocity_data = dashboardData.VelocityData,
                        concentration_data = dashboardData.ConcentrationData,
                        particle_positions = ConvertArray2DToJsonArray(dashboardData.ParticlePositions ?? new double[0,0]),
                        time_info = dashboardData.TimeInfo ?? new Dictionary<string, object>(),
                        statistics = dashboardData.Statistics ?? new Dictionary<string, object>()
                    },
                    parameters = new
                    {
                        title = parameters.Title ?? "海洋模拟综合仪表板",
                        save_path = outputPath
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "dashboard");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    _logger.LogInformation($"综合仪表板生成成功: {outputPath}");
                    return outputPath;
                }

                var errorMessage = result.TryGetProperty("message", out var msgElement) ? msgElement.GetString() : "未知错误";
                _logger.LogError($"综合仪表板生成失败: {errorMessage}");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成综合仪表板失败");
                return "";
            }
        }

        public async Task<OceanDataResult> ProcessNetCDFDataAsync(string filePath, DataProcessingParameters parameters)
        {
            EnsureInitialized();

            try
            {
                _logger.LogInformation($"正在处理NetCDF文件: {filePath}");

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"NetCDF文件不存在: {filePath}");
                }

                var inputData = new
                {
                    action = "load_netcdf",
                    netcdf_path = Path.GetFullPath(filePath),
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

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                var oceanDataResult = new OceanDataResult
                {
                    Success = result.GetProperty("success").GetBoolean(),
                    Message = result.GetProperty("message").GetString() ?? "",
                    U = new double[0, 0],
                    V = new double[0, 0],
                    Latitude = Array.Empty<double>(),
                    Longitude = Array.Empty<double>(),
                    Metadata = new Dictionary<string, object>()
                };

                // 如果成功，解析数据
                if (oceanDataResult.Success && result.TryGetProperty("data_set", out var dataSetElement))
                {
                    oceanDataResult.U = ParseArray2D(dataSetElement.GetProperty("u"));
                    oceanDataResult.V = ParseArray2D(dataSetElement.GetProperty("v"));
                    oceanDataResult.Latitude = ParseArray1D(dataSetElement.GetProperty("lat"));
                    oceanDataResult.Longitude = ParseArray1D(dataSetElement.GetProperty("lon"));
                }

                // 解析元数据
                if (result.TryGetProperty("data_info", out var dataInfoElement))
                {
                    oceanDataResult.Metadata = ParseDictionaryFromJson(dataInfoElement);
                }

                _logger.LogInformation($"NetCDF文件处理完成: {oceanDataResult.Message}");
                return oceanDataResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "处理NetCDF文件失败");
                return new OceanDataResult
                {
                    Success = false,
                    Message = ex.Message,
                    U = new double[0, 0],
                    V = new double[0, 0],
                    Latitude = Array.Empty<double>(),
                    Longitude = Array.Empty<double>(),
                    Metadata = new Dictionary<string, object>()
                };
            }
        }

        public async Task<Dictionary<string, object>> GetPerformanceMetricsAsync()
        {
            EnsureInitialized();

            try
            {
                var inputData = new
                {
                    action = "get_performance_metrics"
                };

                var inputFile = await SaveInputDataAsync(inputData, "performance");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                if (result.GetProperty("success").GetBoolean() && result.TryGetProperty("metrics", out var metricsElement))
                {
                    return ParseDictionaryFromJson(metricsElement);
                }

                return new Dictionary<string, object>
                {
                    ["error"] = result.TryGetProperty("message", out var msgElement)
                        ? (msgElement.GetString() ?? "获取性能指标失败")
                        : "获取性能指标失败"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "获取性能指标失败");
                return new Dictionary<string, object>
                {
                    ["error"] = ex.Message
                };
            }
        }

        #endregion

        #region 扩展方法（兼容原有功能）

        /// <summary>
        /// 导出矢量场为Shapefile
        /// </summary>
        public async Task<ExportResult> ExportVectorShapefileAsync(VectorFieldData vectorField, ShapefileExportParameters parameters)
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
                        u = ConvertArray2DToJsonArray(vectorField.U),
                        v = ConvertArray2DToJsonArray(vectorField.V),
                        lat = vectorField.Latitude,
                        lon = vectorField.Longitude
                    },
                    parameters = new
                    {
                        out_path = outputPath,
                        skip = parameters.Skip,
                        file_type = parameters.FileType
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "export_shapefile");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                var exportResult = new ExportResult
                {
                    Success = result.GetProperty("success").GetBoolean(),
                    Message = result.GetProperty("message").GetString() ?? "",
                    OutputPath = result.TryGetProperty("output_path", out var pathElement)
                                ? pathElement.GetString() ?? ""
                                : ""
                };

                _logger.LogInformation($"Shapefile导出完成: {exportResult.Message}");
                return exportResult;
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
        /// 获取数据统计信息
        /// </summary>
        public async Task<DataStatistics> GetDataStatisticsAsync(VectorFieldData vectorField)
        {
            EnsureInitialized();

            try
            {
                var inputData = new
                {
                    action = "get_statistics",
                    data = new
                    {
                        u = ConvertArray2DToJsonArray(vectorField.U),
                        v = ConvertArray2DToJsonArray(vectorField.V),
                        lat = vectorField.Latitude,
                        lon = vectorField.Longitude
                    }
                };

                var inputFile = await SaveInputDataAsync(inputData, "statistics");
                var outputFile = await ExecutePythonScriptAsync("ocean_data_wrapper.py", inputFile);

                var result = await ReadOutputDataAsync<dynamic>(outputFile);

                var statistics = new DataStatistics
                {
                    Success = result.GetProperty("success").GetBoolean(),
                    Message = result.GetProperty("message").GetString() ?? "",
                    Statistics = new Dictionary<string, object>()
                };

                if (result.TryGetProperty("statistics", out var statsElement))
                {
                    statistics.Statistics = ParseDictionaryFromJson(statsElement);
                }

                return statistics;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "获取数据统计失败");
                return new DataStatistics
                {
                    Success = false,
                    Message = ex.Message,
                    Statistics = new Dictionary<string, object>()
                };
            }
        }

        #endregion

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
                "geopandas", "shapely", "json", "pathlib"
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

        private async Task<JsonElement> ReadOutputDataAsync<T>(string outputFile)
        {
            if (!File.Exists(outputFile))
            {
                throw new FileNotFoundException($"输出文件不存在: {outputFile}");
            }

            var jsonString = await File.ReadAllTextAsync(outputFile);
            var document = JsonDocument.Parse(jsonString);
            var result = document.RootElement;

            // 清理输出文件
            File.Delete(outputFile);

            return result;
        }

        private double[,] ParseArray2D(JsonElement element)
        {
            var arrays = element.EnumerateArray().ToArray();
            if (arrays.Length == 0) return new double[0, 0];

            var firstArray = arrays[0].EnumerateArray().ToArray();
            var result = new double[arrays.Length, firstArray.Length];

            for (int i = 0; i < arrays.Length; i++)
            {
                var row = arrays[i].EnumerateArray().ToArray();
                for (int j = 0; j < row.Length; j++)
                {
                    result[i, j] = row[j].GetDouble();
                }
            }

            return result;
        }

        private double[] ParseArray1D(JsonElement element)
        {
            return element.EnumerateArray()
                         .Select(e => e.GetDouble())
                         .ToArray();
        }

        private Dictionary<string, object> ParseDictionaryFromJson(JsonElement element)
        {
            var result = new Dictionary<string, object>();

            foreach (var property in element.EnumerateObject())
            {
                object? value = property.Value.ValueKind switch
                {
                    JsonValueKind.String => property.Value.GetString() ?? string.Empty,
                    JsonValueKind.Number => property.Value.TryGetDouble(out var dbl) ? dbl : 0.0,
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.Null => null, // 这里允许 null
                    JsonValueKind.Array => property.Value.EnumerateArray().Select(ParseJsonValue).ToArray(),
                    JsonValueKind.Object => ParseDictionaryFromJson(property.Value),
                    _ => property.Value.GetRawText() ?? string.Empty
                };


                result[property.Name] = value ?? DBNull.Value;
            }

            return result;
        }

        private object ParseJsonValue(JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString() ?? string.Empty,
                JsonValueKind.Number => element.TryGetDouble(out var dbl) ? dbl : 0.0,
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => DBNull.Value, // 避免 null 传到 object
                JsonValueKind.Array => element.EnumerateArray().Select(ParseJsonValue).ToArray(),
                JsonValueKind.Object => ParseDictionaryFromJson(element),
                _ => element.GetRawText() ?? string.Empty
            };
        }


        private double[][] ConvertArray2DToJsonArray(double[,] array)
        {
            var rows = array.GetLength(0);
            var cols = array.GetLength(1);
            var result = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                result[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = array[i, j];
                }
            }

            return result;
        }

        private double[][][] ConvertArray3DToJsonArray(double[,,] array)
        {
            var depth = array.GetLength(0);
            var rows = array.GetLength(1);
            var cols = array.GetLength(2);
            var result = new double[depth][][];

            for (int d = 0; d < depth; d++)
            {
                result[d] = new double[rows][];
                for (int r = 0; r < rows; r++)
                {
                    result[d][r] = new double[cols];
                    for (int c = 0; c < cols; c++)
                    {
                        result[d][r][c] = array[d, r, c];
                    }
                }
            }

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

    #region 扩展数据模型定义（原有兼容性）

    /// <summary>
    /// Shapefile导出参数
    /// </summary>
    public class ShapefileExportParameters
    {
        public int Skip { get; set; } = 5;                    // 数据跳过间隔
        public string FileType { get; set; } = "shp";        // 文件类型: "shp" 或 "geojson"
    }

    /// <summary>
    /// 导出结果
    /// </summary>
    public class ExportResult
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public string OutputPath { get; set; } = "";
    }

    /// <summary>
    /// 数据统计信息
    /// </summary>
    public class DataStatistics
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
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
