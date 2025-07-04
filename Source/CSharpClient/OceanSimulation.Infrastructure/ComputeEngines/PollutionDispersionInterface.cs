// =======================================
// 文件: PollutionDispersionInterface.cs
// 接口: PollutionDispersionInterface
// 作者: beilsm
// 版本: v1.2.0
// 功能: 污染物扩散模拟C#接口
// 较上一版:
//   * 路径改成全部绝对路径
//   * 避免拼写错误导致路径乱串
//   * 增强日志输出
//   * 保证运行可重复可测
// 最后更新日期: 2025-07-05
// =======================================

using Microsoft.Extensions.Logging;
using OceanSimulation.Domain.ValueObjects;
using System.Diagnostics;
using System.Text.Json;
using OceanSimulation.Infrastructure.Utils;

namespace OceanSimulation.Infrastructure.ComputeEngines
{
    /// <summary>
    /// 污染物扩散模拟接口
    /// </summary>
    public class PollutionDispersionInterface : IDisposable
    {
        private readonly ILogger<PollutionDispersionInterface> _logger;

        // 绝对路径
        private readonly string _pythonExecutablePath = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/.venv/bin/python";
        private readonly string _pythonEngineRootPath = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine";
        private readonly string _workingDirectory = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/Temp";

        private bool _isInitialized = false;

        public PollutionDispersionInterface(ILogger<PollutionDispersionInterface> logger,
            Dictionary<string, object> configuration)
        {
            _logger = logger;

            // 显式使用绝对路径
            Directory.CreateDirectory(_workingDirectory);
            _logger.LogInformation($@"
污染物扩散接口启动:
  PythonEngineRootPath: {_pythonEngineRootPath}
  WorkingDirectory: {_workingDirectory}
  PythonExecutable: {_pythonExecutablePath}
");
        }

        /// <summary>
        /// 初始化Python环境
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("初始化污染物扩散环境...");
                var result = await RunPythonAsync("--version");
                if (!result.Success)
                {
                    _logger.LogError("Python环境检查失败");
                    return false;
                }

                var wrapperPath = Path.Combine(_pythonEngineRootPath, "wrappers", "pollution_dispersion_wrapper.py");
                if (!File.Exists(wrapperPath))
                {
                    _logger.LogError($"未找到Python包装器: {wrapperPath}");
                    return false;
                }

                _isInitialized = true;
                _logger.LogInformation("污染物扩散环境初始化成功");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "初始化失败");
                return false;
            }
        }

        /// <summary>
        /// 运行默认的污染物扩散模拟
        /// </summary>
        public async Task<PollutionDispersionResult?> RunSimpleSimulationAsync(string? outputPath = null, string? netcdfPath = null)
        {
            if (!_isInitialized)
                throw new InvalidOperationException("尚未初始化，请先调用 InitializeAsync()");

            try
            {
                outputPath ??= Path.Combine(_workingDirectory, $"pollution_{DateTime.Now:yyyyMMdd_HHmmss}.png");

                var parameters = new Dictionary<string, object?>
                {
                    ["output_path"] = outputPath
                };
                if (!string.IsNullOrEmpty(netcdfPath))
                    parameters["netcdf_path"] = netcdfPath;

                var inputData = new
                {
                    action = "run_pollution_dispersion",
                    parameters
                };

                var inputFile = await SaveJsonAsync(inputData, "pollution_input");
                var outputFile = await ExecutePythonScriptAsync(inputFile);
                var result = await ReadJsonAsync(outputFile);

                if (result.GetProperty("success").GetBoolean() && File.Exists(outputPath))
                {
                    var stats = result.GetProperty("statistics");
                    return new PollutionDispersionResult
                    {
                        Success = true,
                        OutputPath = outputPath,
                        MaxConcentration = stats.GetProperty("max_concentration").GetDouble(),
                        MeanConcentration = stats.GetProperty("mean_concentration").GetDouble(),
                        TotalMass = stats.GetProperty("total_mass").GetDouble()
                    };
                }

                var errorMsg = result.TryGetProperty("message", out var msg) ? msg.GetString() ?? "模拟失败" : "模拟失败";
                _logger.LogError($"污染物扩散模拟失败: {errorMsg}");
                return new PollutionDispersionResult { Success = false, ErrorMessage = errorMsg };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "运行污染物扩散模拟失败");
                return new PollutionDispersionResult { Success = false, ErrorMessage = ex.Message };
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

        #region Private helpers

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
            if (!File.Exists(inputFile))
                throw new FileNotFoundException($"输入文件不存在: {inputFile}");

            var scriptPath = Path.Combine(_pythonEngineRootPath, "wrappers", "pollution_dispersion_wrapper.py");
            var outputFile = Path.Combine(_workingDirectory, $"output_{Guid.NewGuid():N}.json");

            _logger.LogInformation($@"
即将执行Python脚本:
  ScriptPath: {scriptPath}
  InputFile: {inputFile}
  OutputFile: {outputFile}
  WorkingDirectory: {_workingDirectory}
");

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
        #endregion
    }
}
