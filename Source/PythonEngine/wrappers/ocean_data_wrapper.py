#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: ocean_data_wrapper.py
位置: Source/PythonEngine/wrappers/ocean_data_wrapper.py
功能: C#调用data_processor.py的包装器脚本
用法: python ocean_data_wrapper.py input.json output.json
"""

import sys
import json
import numpy as np
from pathlib import Path
import traceback
import os
import math


# 添加Python引擎路径到sys.path
current_dir = Path(__file__).parent
python_engine_root = current_dir.parent
sys.path.insert(0, str(python_engine_root.parent))


try:
    from PythonEngine.core.data_processor import DataProcessor
    from PythonEngine.core.netcdf_handler import NetCDFHandler
except ImportError as e:
    print(f"导入模块失败: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    sys.exit(1)

def nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(x) for x in obj]
    return obj


def load_netcdf_data(input_data):
    """加载NetCDF数据"""
    try:
        netcdf_path = input_data['netcdf_path']
        params = input_data['parameters']

        print(f"[INFO] 正在加载NetCDF文件: {netcdf_path}")

        # 检查文件是否存在
        if not os.path.exists(netcdf_path):
            return {
                "success": False,
                "message": f"NetCDF文件不存在: {netcdf_path}",
                "data_info": {}
            }

        # 打开NetCDF文件
        handler = NetCDFHandler(netcdf_path)

        try:
            # 读取数据
            u, v, lat, lon = handler.get_uv(
                time_idx=params.get('time_idx', 0),
                depth_idx=params.get('depth_idx', 0)
            )

            print(f"[INFO] 原始数据形状: u={u.shape}, v={v.shape}, lat={len(lat)}, lon={len(lon)}")

            # 应用地理范围过滤
            original_u, original_v = u.copy(), v.copy()
            original_lat, original_lon = lat.copy(), lon.copy()

            if params.get('lon_min') is not None and params.get('lon_max') is not None:
                lon_mask = (lon >= params['lon_min']) & (lon <= params['lon_max'])
                lon = lon[lon_mask]
                u = u[:, lon_mask]
                v = v[:, lon_mask]
                print(f"[INFO] 应用经度过滤后: lon={len(lon)}, u.shape={u.shape}")

            if params.get('lat_min') is not None and params.get('lat_max') is not None:
                lat_mask = (lat >= params['lat_min']) & (lat <= params['lat_max'])
                lat = lat[lat_mask]
                u = u[lat_mask, :]
                v = v[lat_mask, :]
                print(f"[INFO] 应用纬度过滤后: lat={len(lat)}, u.shape={u.shape}")

            # 数据统计
            u_valid = u[np.isfinite(u)]
            v_valid = v[np.isfinite(v)]
            speed = np.sqrt(u**2 + v**2)
            speed_valid = speed[np.isfinite(speed)]

            return {
                "success": True,
                "message": "NetCDF数据加载成功",
                "data_info": {
                    "source_file": netcdf_path,
                    "time_index": params.get('time_idx', 0),
                    "depth_index": params.get('depth_idx', 0),
                    "original_shape": f"u={original_u.shape}, v={original_v.shape}",
                    "filtered_shape": f"u={u.shape}, v={v.shape}",
                    "lat_range": f"{lat.min():.3f} - {lat.max():.3f}",
                    "lon_range": f"{lon.min():.3f} - {lon.max():.3f}",
                    "u_range": f"{u_valid.min():.3f} - {u_valid.max():.3f}" if len(u_valid) > 0 else "无有效数据",
                    "v_range": f"{v_valid.min():.3f} - {v_valid.max():.3f}" if len(v_valid) > 0 else "无有效数据",
                    "max_speed": f"{speed_valid.max():.3f}" if len(speed_valid) > 0 else "0.000",
                    "valid_points": int(np.sum(np.isfinite(u) & np.isfinite(v))),
                    "total_points": int(u.size)
                },
                "dataset": {
                    "u": u.tolist(),
                    "v": v.tolist(),
                    "latitude": lat.tolist(),
                    "longitude": lon.tolist(),
                    "depth": params.get('depth_idx', 0),
                    "time_info": f"时间索引: {params.get('time_idx', 0)}"
                }
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"NetCDF数据加载失败: {str(e)}",
            "data_info": {},
            "error_trace": traceback.format_exc()
        }

def plot_vector_field(input_data):
    """生成矢量场可视化"""
    try:
        data = input_data['data']
        params = input_data['parameters']

        print(f"[INFO] 正在生成矢量场可视化...")

        # 从输入数据重建numpy数组
        u = np.array(data['u'])
        v = np.array(data['v'])
        lat = np.array(data['lat'])
        lon = np.array(data['lon'])

        print(f"[INFO] 数据形状: u={u.shape}, v={v.shape}, lat={len(lat)}, lon={len(lon)}")

        # 创建数据处理器
        processor = DataProcessor(
            u=u,
            v=v,
            lat=lat,
            lon=lon,
            depth=data.get('depth', 0.0),
            time_info=data.get('time_info', '')
        )

        # 生成可视化
        save_path = params['save_path']

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"[INFO] 保存路径: {save_path}")
        print(f"[INFO] 可视化参数: skip={params.get('skip', 3)}, 范围=({params.get('lon_min')}, {params.get('lon_max')}, {params.get('lat_min')}, {params.get('lat_max')})")

        processor.plot_vector_field(
            skip=params.get('skip', 3),
            show=params.get('show', False),
            save_path=save_path,
            lon_min=params.get('lon_min'),
            lon_max=params.get('lon_max'),
            lat_min=params.get('lat_min'),
            lat_max=params.get('lat_max'),
            contourf_levels=params.get('contourf_levels', 100),
            contourf_cmap=params.get('contourf_cmap', 'coolwarm'),
            quiver_scale=params.get('quiver_scale', 30),
            quiver_width=params.get('quiver_width', 0.001),
            font_size=params.get('font_size', 14),
            dpi=params.get('dpi', 120)
        )

        # 检查文件是否成功生成
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"[INFO] 矢量场图像生成成功，文件大小: {file_size / 1024:.1f} KB")

            return {
                "success": True,
                "message": "矢量场可视化生成成功",
                "image_path": save_path,
                "metadata": {
                    "file_size_kb": round(file_size / 1024, 1),
                    "data_shape": f"{u.shape}",
                    "parameter_skip": params.get('skip', 3),
                    "lon_range": f"{lon.min():.3f} - {lon.max():.3f}",
                    "lat_range": f"{lat.min():.3f} - {lat.max():.3f}",
                    "generation_time": "just_now"
                }
            }
        else:
            return {
                "success": False,
                "message": "矢量场图像文件未能生成",
                "image_path": save_path
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"矢量场可视化失败: {str(e)}",
            "image_path": "",
            "error_trace": traceback.format_exc()
        }

def export_vector_shapefile(input_data):
    """导出矢量场为Shapefile"""
    try:
        data = input_data['data']
        params = input_data['parameters']

        print(f"[INFO] 正在导出矢量场为Shapefile...")

        # 从输入数据重建numpy数组
        u = np.array(data['u'])
        v = np.array(data['v'])
        lat = np.array(data['lat'])
        lon = np.array(data['lon'])

        # 创建数据处理器
        processor = DataProcessor(u=u, v=v, lat=lat, lon=lon)

        # 导出Shapefile
        out_path = params['out_path']
        skip = params.get('skip', 5)
        file_type = params.get('file_type', 'shp')

        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        processor.export_vector_shapefile(
            out_path=out_path,
            skip=skip,
            file_type=file_type
        )

        # 检查输出文件
        expected_file = f"{out_path}.{file_type}"
        if os.path.exists(expected_file):
            file_size = os.path.getsize(expected_file)
            return {
                "success": True,
                "message": f"矢量场已成功导出为{file_type.upper()}格式",
                "output_path": expected_file,
                "metadata": {
                    "file_size_kb": round(file_size / 1024, 1),
                    "file_type": file_type,
                    "skip_interval": skip,
                    "data_points_exported": f"约{(u.size // (skip * skip))}个"
                }
            }
        else:
            return {
                "success": False,
                "message": f"导出文件未能生成: {expected_file}",
                "output_path": ""
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Shapefile导出失败: {str(e)}",
            "output_path": "",
            "error_trace": traceback.format_exc()
        }

def get_statistics(input_data):
    """获取数据统计信息"""
    try:
        data = input_data['data']

        # 从输入数据重建numpy数组
        u = np.array(data['u'])
        v = np.array(data['v'])
        lat = np.array(data['lat'])
        lon = np.array(data['lon'])

        # 计算统计信息
        u_valid = u[np.isfinite(u)]
        v_valid = v[np.isfinite(v)]
        speed = np.sqrt(u**2 + v**2)
        speed_valid = speed[np.isfinite(speed)]

        # 计算方向
        direction = np.arctan2(v, u)
        direction_valid = direction[np.isfinite(direction)]
        direction_deg = np.degrees(direction_valid)

        statistics = {
            "data_dimensions": {
                "u_shape": list(u.shape),
                "v_shape": list(v.shape),
                "lat_points": len(lat),
                "lon_points": len(lon)
            },
            "geographic_extent": {
                "lat_min": float(lat.min()),
                "lat_max": float(lat.max()),
                "lon_min": float(lon.min()),
                "lon_max": float(lon.max()),
                "lat_span": float(lat.max() - lat.min()),
                "lon_span": float(lon.max() - lon.min())
            },
            "velocity_statistics": {
                "u_component": {
                    "min": float(u_valid.min()) if len(u_valid) > 0 else None,
                    "max": float(u_valid.max()) if len(u_valid) > 0 else None,
                    "mean": float(u_valid.mean()) if len(u_valid) > 0 else None,
                    "std": float(u_valid.std()) if len(u_valid) > 0 else None
                },
                "v_component": {
                    "min": float(v_valid.min()) if len(v_valid) > 0 else None,
                    "max": float(v_valid.max()) if len(v_valid) > 0 else None,
                    "mean": float(v_valid.mean()) if len(v_valid) > 0 else None,
                    "std": float(v_valid.std()) if len(v_valid) > 0 else None
                },
                "speed": {
                    "min": float(speed_valid.min()) if len(speed_valid) > 0 else None,
                    "max": float(speed_valid.max()) if len(speed_valid) > 0 else None,
                    "mean": float(speed_valid.mean()) if len(speed_valid) > 0 else None,
                    "std": float(speed_valid.std()) if len(speed_valid) > 0 else None
                }
            },
            "data_quality": {
                "total_points": int(u.size),
                "valid_points": int(np.sum(np.isfinite(u) & np.isfinite(v))),
                "invalid_points": int(np.sum(~(np.isfinite(u) & np.isfinite(v)))),
                "valid_percentage": float(np.sum(np.isfinite(u) & np.isfinite(v)) / u.size * 100)
            },
            "flow_characteristics": {
                "dominant_direction_deg": float(np.median(direction_deg)) if len(direction_deg) > 0 else None,
                "direction_variability": float(np.std(direction_deg)) if len(direction_deg) > 0 else None,
                "high_speed_points": int(np.sum(speed_valid > np.percentile(speed_valid, 90))) if len(speed_valid) > 0 else 0
            }
        }

        return {
            "success": True,
            "message": "数据统计计算完成",
            "statistics": statistics
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"数据统计计算失败: {str(e)}",
            "statistics": {},
            "error_trace": traceback.format_exc()
        }

def main():
    if len(sys.argv) != 3:
        print("用法: python ocean_data_wrapper.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # 读取输入数据
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        print(f"[INFO] 处理请求类型: {input_data.get('action', '未知')}")

        # 根据请求类型处理
        action = input_data.get('action', '')

        if action == 'load_netcdf':
            result = load_netcdf_data(input_data)
        elif action == 'plot_vector_field':
            result = plot_vector_field(input_data)
        elif action == 'export_vector_shapefile':
            result = export_vector_shapefile(input_data)
        elif action == 'get_statistics':
            result = get_statistics(input_data)
        else:
            result = {
                "success": False,
                "message": f"未知的请求类型: {action}"
            }

        # 写入输出结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nan_to_none(result), f, ensure_ascii=False, indent=2)

        print(f"[INFO] 处理完成: {result.get('message', '未知结果')}")

        # 如果成功，返回0；否则返回1
        sys.exit(0 if result.get('success', False) else 1)

    except Exception as e:
        error_result = {
            "success": False,
            "message": f"包装器执行失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, ensure_ascii=False, indent=2)
        except:
            pass  # 如果连输出文件都写不了，就算了

        print(f"[ERROR] 错误: {str(e)}")
        print(f"[ERROR] 详细信息: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()