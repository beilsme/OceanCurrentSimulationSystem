// =====================================
// 文件: OceanAnimationInterface.cs
// 功能: 简洁的洋流时间序列动画生成
// 位置: Source/CharpClient/OceanSimulation.Infrastructure/ComputeEngines/
// =====================================
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.ValueObjects;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 海洋数据动画生成接口 - 从NetCDF时间序列生成GIF动画
    /// </summary>
    public class OceanAnimationInterface : IDisposable
    {
        private readonly ILogger<OceanAnimationInterface> _logger;
        private readonly string _pythonExecutablePath;
        private readonly string _pythonEngineRootPath;
        private readonly string _workingDirectory;
        private bool _isInitialized = false;

        public OceanAnimationInterface(ILogger<OceanAnimationInterface> logger,
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
                _logger.LogInformation("初始化动画生成环境...");

                var result = await RunPythonAsync("--version");
                if (!result.Success)
                {
                    _logger.LogError("Python环境检查失败");
                    return false;
                }

                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "ocean_data_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到ocean_data_wrapper.py: {wrapperPath}");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("动画生成环境初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 生成洋流时间序列GIF动画
        /// </summary>
        /// <param name="netcdfPath">NetCDF文件路径</param>
        /// <param name="outputPath">输出GIF路径（可选）</param>
        /// <param name="maxFrames">最大帧数（默认20）</param>
        /// <param name="frameDelay">帧延迟毫秒（默认500）</param>
        /// <returns>生成的GIF文件路径</returns>
        public async Task<string> GenerateOceanAnimationAsync(
            string netcdfPath,
            string outputPath = "",
            int maxFrames = 20,
            int frameDelay = 500)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("尚未初始化，请先调用InitializeAsync()");

            try
            {
                _logger.LogInformation($"生成洋流动画: {netcdfPath}");

                if (!File.Exists(netcdfPath))
                    throw new FileNotFoundException($"NetCDF文件不存在: {netcdfPath}");

                if (string.IsNullOrEmpty(outputPath))
                    outputPath = Path.Combine(_workingDirectory, $"ocean_animation_{DateTime.Now:yyyyMMdd_HHmmss}.gif");

                var inputData = new
                {
                    action = "create_ocean_animation",
                    parameters = new
                    {
                        netcdf_path = Path.GetFullPath(netcdfPath),
                        output_path = outputPath,
                        max_frames = maxFrames,
                        frame_delay = frameDelay
                    }
                };

                var inputFile = await SaveJsonAsync(inputData, "animation_input");
                var outputFile = await ExecutePythonScriptAsync(inputFile);
                var result = await ReadJsonAsync(outputFile);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    var metadata = result.GetProperty("metadata");
                    var frameCount = metadata.GetProperty("frame_count").GetInt32();
                    var fileSizeMb = metadata.GetProperty("file_size_mb").GetDouble();

                    _logger.LogInformation($"动画生成成功: {outputPath} ({frameCount}帧, {fileSizeMb:F1}MB)");
                    return outputPath;
                }

                var errorMsg = result.TryGetProperty("message", out var msgElement)
                              ? msgElement.GetString() ?? "未知错误"
                              : "动画生成失败";
                _logger.LogError($"动画生成失败: {errorMsg}");
                return "";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "生成洋流动画失败");
                return "";
            }
        }

        public void Dispose()
        {
            _logger.LogInformation("清理动画生成资源...");
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
