// =====================================
// 文件: OceanDataInterface.cs - 极简版
// 功能: 只传文件路径生成可视化，其他全交给Python
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 极简海洋数据处理接口 - 只传文件路径，生成PNG图像
    /// </summary>
    public class OceanDataInterface : IDisposable
    {
        private readonly ILogger<OceanDataInterface> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private bool _isInitialized = false;

        public OceanDataInterface(ILogger<OceanDataInterface> logger,
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
        /// 初始化Python环境
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("初始化Python环境...");

                // 检查Python
                var result = await RunPythonAsync("--version");
                if (!result.Success)
                {
                    _logger.LogError("Python环境检查失败");
                    return false;
                }

                // 检查必要文件
                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到ocean_data_wrapper.py: {wrapperPath}");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("Python环境初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 从NetCDF文件生成可视化图像 - 核心功能
        /// </summary>
        public async Task<string> GenerateVisualizationFromFileAsync(string netcdfPath, string outputPath = "")
        {
            if (!_isInitialized)
                throw new InvalidOperationException("尚未初始化，请先调用InitializeAsync()");

            try
            {
                _logger.LogInformation($"生成可视化: {netcdfPath}");

                if (!File.Exists(netcdfPath))
                    throw new FileNotFoundException($"NetCDF文件不存在: {netcdfPath}");

                if (string.IsNullOrEmpty(outputPath))
                    outputPath = Path.Combine(_workingDirectory, $"ocean_viz_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                // 准备输入数据 - 让Python自己处理所有参数
                var inputData = new
                {
                    action = "plot_vector_field",

                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        save_path = outputPath
                    }
                };

                // 执行Python脚本
                var inputFile = await SaveJsonAsync(inputData, "viz_input");
                var outputFile = await ExecutePythonScriptAsync(inputFile);
                var result = await ReadJsonAsync(outputFile);

                // 检查结果
                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    _logger.LogInformation($"可视化生成成功: {outputPath}");
                    return outputPath;
                }

                var errorMsg = result.TryGetProperty("message", out var msgElement)
                              ? msgElement.GetString() ?? "未知错误"
                              : "生成失败";
                _logger.LogError($"可视化生成失败: {errorMsg}");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成可视化失败");
                return "";
            }
        }

        public void Dispose()
        {
            _logger.LogInformation("清理资源...");
            try
            {
                var tempFiles = Directory.GetFiles(_workingDirectory, "*.json");
                foreach (var file in tempFiles)
                    File.Delete(file);
            }
            catch { }
        }

        #region 私有方法

        private async Task<PythonResult> RunPythonAsync(string arguments)
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
                return new PythonResult { Success = false, Error = ex.Message };
            }
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
                var errorMessage = $"Python脚本执行失败 (退出码: {process.ExitCode})\n错误: {error}\n输出: {output}";
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

            File.Delete(outputFile); // 清理
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

    /// <summary>
    /// Python执行结果
    /// </summary>
    public class PythonResult
    {
        public bool Success { get; set; }
        public string Output { get; set; } = "";
        public string Error { get; set; } = "";
    }
}
