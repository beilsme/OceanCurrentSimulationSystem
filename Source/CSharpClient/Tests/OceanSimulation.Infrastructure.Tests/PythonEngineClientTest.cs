// =====================================
// 文件: PythonEngineClientTest.cs
// 功能: PythonEngineClient独立测试程序
// =====================================
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using OceanSimulation.Infrastructure.ComputeEngines;

namespace OceanSimulation.Tests.Manual
{
    /// <summary>
    /// PythonEngineClient独立测试程序
    /// 可以单独运行，测试读取.nc文件 -> Python生成图 -> 返回结果
    /// </summary>
    class PythonEngineClientTest
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== PythonEngineClient 独立测试 ===");

            // 1. 创建服务容器
            var services = CreateServiceCollection();
            var serviceProvider = services.BuildServiceProvider();

            // 2. 获取PythonEngineClient
            var pythonClient = serviceProvider.GetRequiredService<PythonEngineClient>();
            var logger = serviceProvider.GetRequiredService<ILogger<PythonEngineClientTest>>();

            try
            {
                // 3. 初始化Python客户端
                logger.LogInformation("正在初始化Python客户端...");
                var initSuccess = await pythonClient.InitializeAsync();

                if (!initSuccess)
                {
                    logger.LogError("Python客户端初始化失败");
                    return;
                }

                // 4. 测试NetCDF文件处理
                await TestNetCDFProcessing(pythonClient, logger);

                // 5. 测试矢量场可视化
                await TestVectorFieldVisualization(pythonClient, logger);

                // 6. 测试性能指标
                await TestPerformanceMetrics(pythonClient, logger);

            }
            catch (Exception ex)
            {
                logger.LogError(ex, "测试过程中发生错误");
            }
            finally
            {
                pythonClient.Dispose();
                serviceProvider.Dispose();
            }

            Console.WriteLine("测试完成，按任意键退出...");
            Console.ReadKey();
        }

        /// <summary>
        /// 创建服务容器
        /// </summary>
        private static IServiceCollection CreateServiceCollection()
        {
            var services = new ServiceCollection();

            // 添加日志服务
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information);

            });

            // 添加PythonEngineClient配置
            var pythonConfig = new Dictionary<string, object>
            {
                ["PythonExecutablePath"] = "python",  // 或者指定具体路径
                ["PythonEngineRootPath"] = "./Source/PythonEngine",
                ["WorkingDirectory"] = "./Data/Cache/PythonCache"
            };

            // 注册PythonEngineClient
            services.AddSingleton<PythonEngineClient>(provider =>
            {
                var logger = provider.GetRequiredService<ILogger<PythonEngineClient>>();
                return new PythonEngineClient(logger, pythonConfig);
            });

            return services;
        }

        /// <summary>
        /// 测试NetCDF文件处理
        /// </summary>
        private static async Task TestNetCDFProcessing(PythonEngineClient client, ILogger logger)
        {
            logger.LogInformation("=== 测试NetCDF文件处理 ===");

            try
            {
                // 查找测试用的NetCDF文件
                string[] possiblePaths = {
                    "./Data/NetCDF/test_data.nc",
                    "./Data/NetCDF/merged_data.nc",
                    "../data/raw_data/merged_data.nc"
                };

                string ncFilePath = null;
                foreach (var path in possiblePaths)
                {
                    if (File.Exists(path))
                    {
                        ncFilePath = path;
                        break;
                    }
                }

                if (ncFilePath == null)
                {
                    logger.LogWarning("未找到NetCDF测试文件，跳过此测试");
                    return;
                }

                logger.LogInformation($"使用NetCDF文件: {ncFilePath}");

                var parameters = new DataProcessingParameters
                {
                    TimeIndex = 0,
                    DepthIndex = 0,
                    LonMin = 118.0,
                    LonMax = 124.0,
                    LatMin = 21.0,
                    LatMax = 26.5
                };

                var result = await client.ProcessNetCDFDataAsync(ncFilePath, parameters);

                if (result.Success)
                {
                    logger.LogInformation($"NetCDF处理成功!");
                    logger.LogInformation($"数据维度: U={result.U.GetLength(0)}x{result.U.GetLength(1)}");
                    logger.LogInformation($"经度范围: {result.Longitude[0]} - {result.Longitude[^1]}");
                    logger.LogInformation($"纬度范围: {result.Latitude[0]} - {result.Latitude[^1]}");
                }
                else
                {
                    logger.LogError($"NetCDF处理失败: {result.Message}");
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "NetCDF处理测试失败");
            }
        }

        /// <summary>
        /// 测试矢量场可视化
        /// </summary>
        private static async Task TestVectorFieldVisualization(PythonEngineClient client, ILogger logger)
        {
            logger.LogInformation("=== 测试矢量场可视化 ===");

            try
            {
                // 创建测试数据
                var testData = CreateTestVectorFieldData();

                var visualParams = new VisualizationParameters
                {
                    Skip = 3,
                    LonMin = 118.0,
                    LonMax = 124.0,
                    LatMin = 21.0,
                    LatMax = 26.5,
                    FontSize = 14,
                    DPI = 120
                };

                var imagePath = await client.GenerateVectorFieldVisualizationAsync(testData, visualParams);

                if (File.Exists(imagePath))
                {
                    logger.LogInformation($"矢量场可视化成功生成: {imagePath}");
                    logger.LogInformation($"图像文件大小: {new FileInfo(imagePath).Length / 1024} KB");
                }
                else
                {
                    logger.LogError("矢量场可视化生成失败");
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "矢量场可视化测试失败");
            }
        }

        /// <summary>
        /// 测试性能指标
        /// </summary>
        private static async Task TestPerformanceMetrics(PythonEngineClient client, ILogger logger)
        {
            logger.LogInformation("=== 测试性能指标获取 ===");

            try
            {
                var metrics = await client.GetPerformanceMetricsAsync();

                logger.LogInformation("性能指标:");
                foreach (var kvp in metrics)
                {
                    logger.LogInformation($"  {kvp.Key}: {kvp.Value}");
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "性能指标测试失败");
            }
        }

        /// <summary>
        /// 创建测试用的矢量场数据
        /// </summary>
        private static VectorFieldData CreateTestVectorFieldData()
        {
            // 创建简单的测试数据
            int latPoints = 20;
            int lonPoints = 25;

            var lat = new double[latPoints];
            var lon = new double[lonPoints];
            var u = new double[latPoints, lonPoints];
            var v = new double[latPoints, lonPoints];

            // 生成纬度经度网格
            for (int i = 0; i < latPoints; i++)
            {
                lat[i] = 21.0 + (26.5 - 21.0) * i / (latPoints - 1);
            }

            for (int j = 0; j < lonPoints; j++)
            {
                lon[j] = 118.0 + (124.0 - 118.0) * j / (lonPoints - 1);
            }

            // 生成简单的涡旋流场
            double centerLat = 23.75;
            double centerLon = 121.0;

            for (int i = 0; i < latPoints; i++)
            {
                for (int j = 0; j < lonPoints; j++)
                {
                    double dx = lon[j] - centerLon;
                    double dy = lat[i] - centerLat;
                    double r = Math.Sqrt(dx * dx + dy * dy);

                    if (r > 0.01)
                    {
                        double strength = 0.5 * Math.Exp(-r * 2);
                        u[i, j] = -dy / r * strength;
                        v[i, j] = dx / r * strength;
                    }
                    else
                    {
                        u[i, j] = 0;
                        v[i, j] = 0;
                    }
                }
            }

            return new VectorFieldData
            {
                U = u,
                V = v,
                Latitude = lat,
                Longitude = lon,
                Depth = 0.0,
                TimeInfo = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
            };
        }
    }
}

// =====================================
// 文件: python_wrappers/data_processor_wrapper.py
// 功能: data_processor的Python包装器脚本
// =====================================
#!/usr/bin/env python3
"""
C#调用data_processor的包装器脚本
用法: python data_processor_wrapper.py input.json output.json
"""

import sys
import json
import numpy as np
from pathlib import Path
import traceback

# 添加Python引擎路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_processor import DataProcessor
from core.netcdf_handler import NetCDFHandler

def process_vector_field_visualization(input_data):
    """处理矢量场可视化请求"""
    try:
        # 提取数据
        u_data = np.array(input_data['u_data'])
        v_data = np.array(input_data['v_data'])
        lat_data = np.array(input_data['lat_data'])
        lon_data = np.array(input_data['lon_data'])
        depth = input_data.get('depth', 0.0)
        time_info = input_data.get('time_info', '')

        # 提取可视化参数
        viz_params = input_data['visualization_params']

        # 创建数据处理器
        processor = DataProcessor(u_data, v_data, lat_data, lon_data, depth, time_info)

        # 生成可视化
        save_path = viz_params['save_path']
        processor.plot_vector_field(
            skip=viz_params.get('skip', 3),
            show=False,
            save_path=save_path,
            lon_min=viz_params.get('lon_min'),
            lon_max=viz_params.get('lon_max'),
            lat_min=viz_params.get('lat_min'),
            lat_max=viz_params.get('lat_max'),
            font_size=viz_params.get('font_size', 14),
            dpi=viz_params.get('dpi', 120)
        )

        return {
            "success": True,
            "image_path": save_path,
            "message": "矢量场可视化生成成功",
            "metadata": {
                "data_shape": f"{u_data.shape}",
                "lon_range": f"{lon_data.min():.2f} - {lon_data.max():.2f}",
                "lat_range": f"{lat_data.min():.2f} - {lat_data.max():.2f}"
            }
        }

    except Exception as e:
        return {
            "success": False,
            "image_path": "",
            "message": f"矢量场可视化失败: {str(e)}",
            "metadata": {"error_trace": traceback.format_exc()}
        }

def process_netcdf_data(input_data):
    """处理NetCDF数据"""
    try:
        netcdf_path = input_data['netcdf_path']
        params = input_data['processing_params']

        # 打开NetCDF文件
        handler = NetCDFHandler(netcdf_path)

        # 读取数据
        u, v, lat, lon = handler.get_uv(
            time_idx=params.get('time_idx', 0),
            depth_idx=params.get('depth_idx', 0)
        )

        # 应用地理范围过滤
        if params.get('lon_min') is not None:
            lon_mask = (lon >= params['lon_min']) & (lon <= params['lon_max'])
            lon = lon[lon_mask]
            u = u[:, lon_mask]
            v = v[:, lon_mask]

        if params.get('lat_min') is not None:
            lat_mask = (lat >= params['lat_min']) & (lat <= params['lat_max'])
            lat = lat[lat_mask]
            u = u[lat_mask, :]
            v = v[lat_mask, :]

        handler.close()

        return {
            "success": True,
            "u": u.tolist(),
            "v": v.tolist(),
            "latitude": lat.tolist(),
            "longitude": lon.tolist(),
            "message": "NetCDF数据处理成功",
            "metadata": {
                "source_file": netcdf_path,
                "data_shape": f"{u.shape}",
                "time_index": params.get('time_idx', 0),
                "depth_index": params.get('depth_idx', 0)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "u": [],
            "v": [],
            "latitude": [],
            "longitude": [],
            "message": f"NetCDF数据处理失败: {str(e)}",
            "metadata": {"error_trace": traceback.format_exc()}
        }

def main():
    if len(sys.argv) != 3:
        print("用法: python data_processor_wrapper.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # 读取输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 根据请求类型处理
        if 'u_data' in input_data:
            # 矢量场可视化请求
            result = process_vector_field_visualization(input_data)
        elif 'netcdf_path' in input_data:
            # NetCDF数据处理请求
            result = process_netcdf_data(input_data)
        else:
            result = {
                "success": False,
                "message": "未知的请求类型"
            }

        # 写入输出结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"处理完成: {result['message']}")

    except Exception as e:
        error_result = {
            "success": False,
            "message": f"包装器执行失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)

        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
