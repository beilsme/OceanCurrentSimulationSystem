using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace PythonTest
{
    class Program
    {
        // 修改Python路径为你的虚拟环境
        private static readonly string PythonExe = GetPythonPath();
        private static readonly string WorkingDir = "./test_output";

        static async Task Main(string[] args)
        {
            Console.WriteLine("=== 独立Python调用测试 ===");
            Console.WriteLine($"Python路径: {PythonExe}");

            // 确保输出目录存在
            Directory.CreateDirectory(WorkingDir);

            try
            {
                // 1. 检查Python环境
                Console.WriteLine("\n1. 检查Python环境...");
                if (!await CheckPythonEnvironment())
                {
                    Console.WriteLine("❌ Python环境检查失败");
                    return;
                }
                Console.WriteLine("✅ Python环境正常");

                // 2. 测试简单的Python脚本调用
                Console.WriteLine("\n2. 测试基础Python调用...");
                await TestBasicPythonCall();

                // 3. 测试数据可视化生成
                Console.WriteLine("\n3. 测试数据可视化生成...");
                await TestDataVisualization();

                // 4. 如果有NetCDF文件，测试文件处理
                Console.WriteLine("\n4. 查找并测试NetCDF文件处理...");
                await TestNetCDFProcessing();

                Console.WriteLine("\n🎉 所有测试完成！");
                Console.WriteLine($"📁 查看生成的图像文件在: {Path.GetFullPath(WorkingDir)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 测试过程中发生错误: {ex.Message}");
                Console.WriteLine($"详细信息: {ex}");
            }

            Console.WriteLine("\n按任意键退出...");
            Console.ReadKey();
        }

        static string GetPythonPath()
        {
            // 根据你的环境路径配置Python可执行文件路径
            var possiblePaths = new[]
            {
                // 你的虚拟环境路径
                "../../Source/PythonEngine/venv_oceansim/Scripts/python.exe",  // Windows
                "../../Source/PythonEngine/venv_oceansim/bin/python",          // Linux/Mac

                // 备用路径
                "../../../Source/PythonEngine/venv_oceansim/Scripts/python.exe",
                "../../../Source/PythonEngine/venv_oceansim/bin/python",

                // 系统Python
                "python",
                "python3"
            };

            foreach (var path in possiblePaths)
            {
                try
                {
                    if (File.Exists(path))
                    {
                        return Path.GetFullPath(path);
                    }
                }
                catch { }
            }

            return "python"; // 默认使用系统Python
        }

        static async Task<bool> CheckPythonEnvironment()
        {
            try
            {
                var result = await RunPythonCommand("--version");
                Console.WriteLine($"   Python版本: {result.Output}");

                if (!result.Success)
                {
                    Console.WriteLine($"   错误信息: {result.Error}");
                    return false;
                }

                // 检查必要的Python包
                Console.WriteLine("   检查Python包...");
                var packages = new[] { "numpy", "matplotlib" };

                foreach (var package in packages)
                {
                    var packageResult = await RunPythonCommand($"-c \"import {package}; print('{package} 可用')\"");
                    if (packageResult.Success)
                    {
                        Console.WriteLine($"   ✅ {package}: {packageResult.Output}");
                    }
                    else
                    {
                        Console.WriteLine($"   ❌ {package}: 未安装或不可用");
                        Console.WriteLine($"   建议运行: pip install {package}");
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   错误: {ex.Message}");
                return false;
            }
        }

        static async Task TestBasicPythonCall()
        {
            var pythonCode = @"
import json
import sys
import os

# 设置matplotlib使用非交互式后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    # 生成简单的测试图
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.title('Python调用测试 - 正弦函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = sys.argv[1] if len(sys.argv) > 1 else 'test_output.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    # 返回结果
    result = {
        'success': True,
        'message': '基础Python调用测试成功',
        'image_path': output_path,
        'data_points': len(x),
        'file_exists': os.path.exists(output_path)
    }

    print(json.dumps(result, indent=2))

except Exception as e:
    result = {
        'success': False,
        'message': f'基础Python调用失败: {str(e)}',
        'error_type': type(e).__name__
    }
    print(json.dumps(result, indent=2))
";

            var scriptPath = Path.Combine(WorkingDir, "test_basic.py");
            var imagePath = Path.Combine(WorkingDir, "basic_test.png");

            await File.WriteAllTextAsync(scriptPath, pythonCode);

            var result = await RunPythonScript(scriptPath, imagePath);

            if (result.Success)
            {
                Console.WriteLine("   ✅ 基础Python调用成功");
                if (File.Exists(imagePath))
                {
                    Console.WriteLine($"   📊 生成图像: {imagePath}");
                    Console.WriteLine($"   📏 文件大小: {new FileInfo(imagePath).Length / 1024} KB");
                }
                else
                {
                    Console.WriteLine("   ⚠️  图像文件未生成");
                }

                // 显示Python返回的详细信息
                try
                {
                    var jsonResult = JsonSerializer.Deserialize<Dictionary<string, object>>(result.Output);
                    if (jsonResult.ContainsKey("data_points"))
                        Console.WriteLine($"   📈 数据点数: {jsonResult["data_points"]}");
                }
                catch { }
            }
            else
            {
                Console.WriteLine($"   ❌ 基础Python调用失败: {result.Error}");
                Console.WriteLine($"   输出: {result.Output}");
            }
        }

        static async Task TestDataVisualization()
        {
            var pythonCode = @"
import json
import sys
import os
import numpy as np

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    # 生成模拟的海流数据
    def generate_ocean_current_data():
        # 创建网格 - 模拟南海区域
        lat = np.linspace(21, 26.5, 20)
        lon = np.linspace(118, 124, 25)
        LAT, LON = np.meshgrid(lat, lon, indexing='ij')

        # 生成涡旋流场
        center_lat, center_lon = 23.75, 121.0
        dx = LON - center_lon
        dy = LAT - center_lat
        r = np.sqrt(dx**2 + dy**2)

        # 避免除零
        r = np.where(r < 0.01, 0.01, r)

        # 涡旋强度
        strength = 0.5 * np.exp(-r * 2)

        # 速度分量
        u = -dy / r * strength
        v = dx / r * strength

        return lat, lon, u, v

    # 生成数据
    lat, lon, u, v = generate_ocean_current_data()

    # 创建可视化
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算速度大小
    speed = np.sqrt(u**2 + v**2)

    # 绘制速度场
    skip = 2
    X, Y = np.meshgrid(lon[::skip], lat[::skip])
    U = u[::skip, ::skip]
    V = v[::skip, ::skip]

    # 背景颜色图
    im = ax.contourf(X, Y, speed[::skip, ::skip], levels=20, cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(im, ax=ax, label='流速 (m/s)')

    # 矢量箭头
    ax.quiver(X, Y, U, V, scale=10, width=0.003, color='black', alpha=0.8)

    ax.set_xlabel('经度 (°E)', fontsize=12)
    ax.set_ylabel('纬度 (°N)', fontsize=12)
    ax.set_title('模拟海流场可视化测试', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加地理信息
    ax.text(0.02, 0.98, '南海区域', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存图像
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'ocean_current_test.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    # 返回结果
    result = {
        'success': True,
        'message': '海流数据可视化测试成功',
        'image_path': output_path,
        'data_shape': f'{u.shape}',
        'lon_range': f'{lon.min():.2f} - {lon.max():.2f}',
        'lat_range': f'{lat.min():.2f} - {lat.max():.2f}',
        'max_speed': f'{speed.max():.3f}',
        'min_speed': f'{speed.min():.3f}',
        'file_exists': os.path.exists(output_path)
    }

    print(json.dumps(result, indent=2))

except Exception as e:
    result = {
        'success': False,
        'message': f'海流可视化失败: {str(e)}',
        'error_type': type(e).__name__
    }
    print(json.dumps(result, indent=2))
";

            var scriptPath = Path.Combine(WorkingDir, "test_ocean_viz.py");
            var imagePath = Path.Combine(WorkingDir, "ocean_current_test.png");

            await File.WriteAllTextAsync(scriptPath, pythonCode);

            var result = await RunPythonScript(scriptPath, imagePath);

            if (result.Success)
            {
                Console.WriteLine("   ✅ 海流可视化测试成功");
                if (File.Exists(imagePath))
                {
                    Console.WriteLine($"   🌊 生成海流图: {imagePath}");

                    // 尝试解析Python返回的JSON结果
                    try
                    {
                        var jsonResult = JsonSerializer.Deserialize<Dictionary<string, object>>(result.Output);
                        if (jsonResult.ContainsKey("data_shape"))
                            Console.WriteLine($"   📐 数据形状: {jsonResult["data_shape"]}");
                        if (jsonResult.ContainsKey("max_speed"))
                            Console.WriteLine($"   💨 最大流速: {jsonResult["max_speed"]} m/s");
                        if (jsonResult.ContainsKey("lon_range"))
                            Console.WriteLine($"   🌐 经度范围: {jsonResult["lon_range"]}");
                        if (jsonResult.ContainsKey("lat_range"))
                            Console.WriteLine($"   🌐 纬度范围: {jsonResult["lat_range"]}");
                    }
                    catch { /* 忽略JSON解析错误 */ }
                }
            }
            else
            {
                Console.WriteLine($"   ❌ 海流可视化测试失败: {result.Error}");
                Console.WriteLine($"   输出: {result.Output}");
            }
        }

        static async Task TestNetCDFProcessing()
        {
            // 查找可能的NetCDF文件
            var possiblePaths = new[]
            {
                "../../Data/NetCDF/test_data.nc",
                "../../Data/NetCDF/merged_data.nc",
                "../../../Data/NetCDF/merged_data.nc",
                "../../Source/PythonEngine/data/test_data.nc",
                "./test_data.nc"
            };

            string foundFile = null;
            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    foundFile = path;
                    break;
                }
            }

            if (foundFile == null)
            {
                Console.WriteLine("   ⚠️  未找到NetCDF测试文件，跳过此测试");
                Console.WriteLine("   💡 可以手动放置.nc文件到以下位置进行测试:");
                foreach (var path in possiblePaths)
                {
                    Console.WriteLine($"      - {path}");
                }
                return;
            }

            Console.WriteLine($"   📁 找到NetCDF文件: {foundFile}");

            // 生成NetCDF处理脚本
            var pythonCode = @"
import json
import sys
import os

def test_netcdf_processing(nc_file_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(nc_file_path):
            return {
                'success': False,
                'message': f'NetCDF文件不存在: {nc_file_path}'
            }

        # 尝试导入netCDF4
        try:
            import netCDF4 as nc
        except ImportError:
            return {
                'success': False,
                'message': '缺少netCDF4模块，请安装: pip install netCDF4'
            }

        # 读取NetCDF文件信息
        with nc.Dataset(nc_file_path, 'r') as dataset:
            # 获取基本信息
            dimensions = dict(dataset.dimensions)
            variables = list(dataset.variables.keys())

            # 获取文件大小
            file_size = os.path.getsize(nc_file_path)

            result = {
                'success': True,
                'message': 'NetCDF文件读取成功',
                'file_path': nc_file_path,
                'file_size_mb': round(file_size / (1024*1024), 2),
                'dimensions': {name: size.size for name, size in dimensions.items()},
                'variables': variables[:10],  # 只显示前10个变量
                'total_variables': len(variables)
            }

            # 如果有常见的海洋变量，尝试读取一些数据
            if 'u' in variables and 'v' in variables:
                u_var = dataset.variables['u']
                v_var = dataset.variables['v']
                result['has_velocity_data'] = True
                result['u_shape'] = list(u_var.shape)
                result['v_shape'] = list(v_var.shape)

                # 尝试读取一小部分数据
                if len(u_var.shape) >= 2:
                    u_sample = u_var[0, 0] if len(u_var.shape) == 2 else u_var[0, 0, 0]
                    v_sample = v_var[0, 0] if len(v_var.shape) == 2 else v_var[0, 0, 0]
                    result['sample_u'] = float(u_sample) if not hasattr(u_sample, 'mask') or not u_sample.mask else 'masked'
                    result['sample_v'] = float(v_sample) if not hasattr(v_sample, 'mask') or not v_sample.mask else 'masked'

            return result

    except Exception as e:
        return {
            'success': False,
            'message': f'NetCDF处理失败: {str(e)}',
            'error_type': type(e).__name__
        }

# 主程序
if __name__ == '__main__':
    nc_file = sys.argv[1] if len(sys.argv) > 1 else 'test.nc'
    result = test_netcdf_processing(nc_file)
    print(json.dumps(result, indent=2))
";

            var scriptPath = Path.Combine(WorkingDir, "test_netcdf.py");
            await File.WriteAllTextAsync(scriptPath, pythonCode);

            var result = await RunPythonScript(scriptPath, foundFile);

            if (result.Success)
            {
                Console.WriteLine("   ✅ NetCDF文件处理成功");

                // 解析并显示结果
                try
                {
                    var jsonResult = JsonSerializer.Deserialize<Dictionary<string, object>>(result.Output);

                    if (jsonResult.ContainsKey("file_size_mb"))
                        Console.WriteLine($"   📊 文件大小: {jsonResult["file_size_mb"]} MB");

                    if (jsonResult.ContainsKey("total_variables"))
                        Console.WriteLine($"   🔢 变量数量: {jsonResult["total_variables"]}");

                    if (jsonResult.ContainsKey("has_velocity_data") &&
                        jsonResult["has_velocity_data"] is JsonElement elem && elem.GetBoolean())
                        Console.WriteLine("   🌊 包含速度场数据 (u, v)");

                    if (jsonResult.ContainsKey("dimensions"))
                    {
                        Console.WriteLine("   📐 数据维度:");
                        var dims = jsonResult["dimensions"].ToString();
                        Console.WriteLine($"      {dims}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️  结果解析错误: {ex.Message}");
                    Console.WriteLine($"   原始输出: {result.Output}");
                }
            }
            else
            {
                Console.WriteLine($"   ❌ NetCDF文件处理失败: {result.Error}");
                if (result.Output.Contains("netCDF4"))
                {
                    Console.WriteLine($"   💡 请在虚拟环境中安装: pip install netCDF4");
                }
            }
        }

        #region 辅助方法

        static async Task<PythonResult> RunPythonCommand(string arguments)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = PythonExe,
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

        static async Task<PythonResult> RunPythonScript(string scriptPath, string argument = "")
        {
            var args = string.IsNullOrEmpty(argument) ? $"\"{scriptPath}\"" : $"\"{scriptPath}\" \"{argument}\"";
            return await RunPythonCommand(args);
        }

        #endregion
    }

    #region 数据结构

    public class PythonResult
    {
        public bool Success { get; set; }
        public string Output { get; set; } = "";
        public string Error { get; set; } = "";
    }

    #endregion
}
