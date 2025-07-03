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
import oceansim


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
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        if not netcdf_path:
            raise ValueError("netcdf_path 未提供，无法加载 NetCDF")


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
                    "lat": lat.tolist(),
                    "lon": lon.tolist(),
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
    """简化版: 仅传入 netcdf_path，自动读取并绘图"""
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        if not netcdf_path or not os.path.exists(netcdf_path):
            raise ValueError(f"netcdf_path 未提供或文件不存在: {netcdf_path}")

        print(f"[INFO] 自动加载NetCDF并生成矢量场: {netcdf_path}")

        # 先读取数据
        handler = NetCDFHandler(netcdf_path)
        u, v, lat, lon = handler.get_uv(
            time_idx=params.get('time_idx', 0),
            depth_idx=params.get('depth_idx', 0)
        )

        handler.close()

        # 读取时间信息
        selected_time_idx = params.get('time_idx', 0)
        time_value = handler.get_time(index=selected_time_idx)
        if not time_value:
            time_value = "未知"

        # 读取深度信息
        selected_depth_idx = params.get('depth_idx', 0)
        depth_value = handler.get_depth(index=selected_depth_idx)
        if depth_value is None:
            depth_value = 0.0


        # 创建数据处理器
        processor = DataProcessor(
            u=u,
            v=v,
            lat=lat,
            lon=lon,
            depth=depth_value,
            time_info=time_value
        )

        save_path = params.get('save_path', "auto_plot.png")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"[INFO] 输出图像: {save_path}")

        processor.plot_vector_field(
            save_path=save_path,
            skip=params.get('skip', 3),
            show=params.get('show', False),
            contourf_levels=params.get('contourf_levels', 100),
            contourf_cmap=params.get('contourf_cmap', 'coolwarm'),
            quiver_scale=params.get('quiver_scale', 30),
            quiver_width=params.get('quiver_width', 0.001),
            font_size=params.get('font_size', 14),
            dpi=params.get('dpi', 120)
        )

        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            return {
                "success": True,
                "message": "矢量场绘制完成",
                "image_path": save_path,
                "metadata": {
                    "file_size_kb": round(file_size / 1024, 1),
                    "shape": str(u.shape)
                }
            }
        else:
            return {
                "success": False,
                "message": "图像文件未能生成",
                "image_path": save_path
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"矢量场绘制失败: {str(e)}",
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

def create_ocean_animation(input_data):
    """
    创建洋流时间序列GIF动画，复用 plot_vector_field，确保和单帧渲染一致
    """
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        output_path = params.get('output_path')
        frame_delay = params.get('frame_delay', 500)

        if not netcdf_path or not os.path.exists(netcdf_path):
            raise ValueError(f"NetCDF文件不存在: {netcdf_path}")

        print(f"[INFO] 创建洋流动画: {netcdf_path}")
        print(f"[INFO] 帧延迟: {frame_delay}ms")

        # 打开NetCDF文件
        handler = NetCDFHandler(netcdf_path)
        try:
            ds = handler.ds
            if 'time' not in ds.dims:
                raise ValueError("NetCDF文件中没有时间维度")

            total_time_steps = ds.sizes['time']
            frame_stride = params.get("frame_stride", 1)
            time_indices = list(range(0, total_time_steps, max(1, frame_stride)))

            print(f"[INFO] NetCDF文件包含 {total_time_steps} 个时间步")
            print(f"[INFO] 使用帧步长 frame_stride={frame_stride}")
            print(f"[INFO] 将生成 {len(time_indices)} 帧动画")


            # 临时帧目录
            temp_dir = os.path.join(os.path.dirname(output_path), f"temp_frames_{os.getpid()}")
            os.makedirs(temp_dir, exist_ok=True)

            frame_files = []

            try:
                # 复用 plot_vector_field
                for i, time_idx in enumerate(time_indices):
                    print(f"[INFO] 渲染帧 {i+1}/{len(time_indices)} (时间索引: {time_idx})")

                    frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")

                    # 继承参数，动态修改
                    plot_params = params.copy()
                    plot_params.update({
                        "time_idx": time_idx,
                        "save_path": frame_path,
                        "show": False
                    })

                    # 直接调用 plot_vector_field
                    plot_result = plot_vector_field({
                        "parameters": plot_params
                    })

                    if plot_result.get("success") and os.path.exists(frame_path):
                        frame_files.append(frame_path)
                    else:
                        print(f"[WARNING] 第 {i+1} 帧生成失败")

                if not frame_files:
                    raise ValueError("没有成功生成任何帧")

                print(f"[INFO] 合成GIF动画，帧数: {len(frame_files)}")
                _create_gif_from_frames(frame_files, output_path, frame_delay)

                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"[INFO] GIF动画生成成功: {output_path}")
                    return {
                        "success": True,
                        "message": "洋流动画生成成功",
                        "output_path": output_path,
                        "metadata": {
                            "frame_count": len(frame_files),
                            "file_size_mb": round(file_size / (1024 * 1024), 2),
                            "frame_delay_ms": frame_delay,
                            "total_time_steps": total_time_steps
                        }
                    }
                else:
                    return {
                        "success": False,
                        "message": "GIF文件未生成",
                        "output_path": output_path
                    }
            finally:
                # 清理
                try:
                    for f in frame_files:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except:
                    pass

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"洋流动画生成失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }



def _create_gif_from_frames(frame_files, output_path, frame_delay):
    """从帧图片创建GIF动画"""
    try:
        from PIL import Image

        # 读取所有帧
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            # 转换为RGB（GIF需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)

        if not images:
            raise ValueError("没有有效的帧图片")

        # 保存为GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_delay,
            loop=0,  # 无限循环
            optimize=True
        )

        print(f"[INFO] GIF保存成功: {output_path}")

    except ImportError:
        # 如果没有PIL，尝试使用matplotlib
        print("[INFO] PIL不可用，尝试使用matplotlib生成GIF")
        _create_gif_with_matplotlib(frame_files, output_path, frame_delay)
    except Exception as e:
        print(f"[ERROR] GIF创建失败: {str(e)}")
        raise


def _create_gif_with_matplotlib(frame_files, output_path, frame_delay):
    """使用matplotlib创建GIF（备用方案）"""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # 读取第一帧获取尺寸
    first_img = plt.imread(frame_files[0])

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # 初始化显示
    im = ax.imshow(first_img)

    def animate(frame_idx):
        img = plt.imread(frame_files[frame_idx])
        im.set_array(img)
        return [im]

    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frame_files),
        interval=frame_delay, blit=True, repeat=True
    )

    # 保存GIF
    try:
        anim.save(output_path, writer='pillow', fps=1000//frame_delay)
    except:
        # 如果pillow writer不可用，尝试默认writer
        anim.save(output_path, fps=1000//frame_delay)

    plt.close(fig)


def check_grid_data_formats(u, v, lat, lon):
    """测试不同的数据格式以找到与oceansim.GridDataStructure兼容的正确方法"""
    print(f"[TEST] 开始测试网格数据格式兼容性")
    print(f"[TEST] 原始数据形状: u={u.shape}, v={v.shape}")
    print(f"[TEST] 坐标数组长度: lat={len(lat)}, lon={len(lon)}")

    nx, ny = len(lon), len(lat)

    # 测试不同的网格构造参数和数据格式组合
    # 首先尝试与坐标维度一致的常规布局
    try_configs = [
        {
            "name": "lon_lat",
            "grid": (len(lon), len(lat), 1),
            "u": u,
            "v": v,
        },
        {
            "name": "lat_lon",
            "grid": (len(lat), len(lon), 1),
            "u": u.T,
            "v": v.T,
        },
    ]

    for cfg in try_configs:
        config_name = cfg["name"]
        grid_params = cfg["grid"]
        try:
            print(f"[TEST] 测试配置: 网格{grid_params}, 数据格式matrix")

            grid = oceansim.GridDataStructure(*grid_params, oceansim.CoordinateSystem.CARTESIAN)
            grid_dims = grid.get_dimensions()
            print(f"[TEST] 网格维度: {grid_dims}")

            u_data = cfg["u"].astype(np.float64)
            v_data = cfg["v"].astype(np.float64)

            grid.add_field2d("test_u", u_data)
            grid.add_field2d("test_v", v_data)

            print(f"[TEST] ✅ 成功找到兼容配置: {config_name}")
            return {
                "success": True,
                "grid_params": grid_params,
                "data_format": config_name,
                "grid": grid,
                "u_data": u_data,
                "v_data": v_data,
            }
        except Exception as e:
            print(f"[TEST] ❌ 配置失败: {str(e)}")
            continue

    return {"success": False, "error": "未找到兼容配置"}


def _validate_grid_dimensions(u, v, lat, lon, grid):
    """验证数据维度与网格维度的匹配性"""
    try:
        data_shape = u.shape
        ny_data, nx_data = data_shape

        grid_dims = grid.get_dimensions()
        ny_grid, nx_grid = grid_dims[0], grid_dims[1]

        if ny_data != ny_grid or nx_data != nx_grid:
            raise ValueError(
                f"数据维度不匹配: 数据形状为 ({ny_data}, {nx_data}), "
                f"网格期望为 ({ny_grid}, {nx_grid})"
            )

        if len(lat) != ny_data:
            raise ValueError(f"纬度数组长度 {len(lat)} 与数据纬度维度 {ny_data} 不匹配")

        if len(lon) != nx_data:
            raise ValueError(f"经度数组长度 {len(lon)} 与数据经度维度 {nx_data} 不匹配")

        if u.shape != v.shape:
            raise ValueError(f"U和V分量维度不匹配: U形状为 {u.shape}, V形状为 {v.shape}")

        print(f"[DEBUG] 维度验证通过: 数据 {data_shape}, 网格 ({ny_grid}, {nx_grid})")
        return True

    except Exception as e:
        print(f"[ERROR] 维度验证失败: {str(e)}")
        raise


def calculate_vorticity_divergence(input_data):
    """计算涡度场和散度场 - 使用oceansim.CurrentFieldSolver"""
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        output_path = params.get('output_path')
        time_index = params.get('time_index', 0)
        depth_index = params.get('depth_index', 0)

        print(f"[INFO] 使用oceansim.CurrentFieldSolver计算散度场: 时间{time_index}, 深度{depth_index}")

        handler = NetCDFHandler(netcdf_path)
        try:
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=depth_index)

            # 自动测试并找到兼容的配置
            test_result = check_grid_data_formats(u, v, lat, lon)
            if not test_result["success"]:
                raise ValueError(f"无法找到兼容的网格数据格式: {test_result.get('error', '未知错误')}")

            # 使用测试成功的配置重新创建网格
            grid_params = test_result["grid_params"]
            data_format = test_result["data_format"]

            print(f"[INFO] 使用配置: 网格{grid_params}, 数据格式{data_format}")

            grid = oceansim.GridDataStructure(*grid_params, oceansim.CoordinateSystem.CARTESIAN)

            # 设置网格间距
            dx = abs(lon[1] - lon[0]) * 111000
            dy = abs(lat[1] - lat[0]) * 111000

            # 创建物理参数和洋流场求解器
            params_obj = oceansim.PhysicalParameters()
            current_solver = oceansim.CurrentFieldSolver(grid, params_obj)

            # 根据测试结果准备数据
            u_data = test_result.get("u_data")
            v_data = test_result.get("v_data")
            if u_data is None or v_data is None:
                u_data = u.astype(np.float64)
                v_data = v.astype(np.float64)

            # 设置速度场数据到网格
            grid.add_field2d("u_velocity", u_data)
            grid.add_field2d("v_velocity", v_data)

            # 使用C++计算散度
            divergence_result = current_solver.compute_divergence()

            # 根据网格配置正确重塑散度结果
            if grid_params[0] == len(lat):  # ny, nx, 1
                divergence = np.array(divergence_result).reshape(len(lat), len(lon))
            else:  # nx, ny, 1
                divergence = np.array(divergence_result).reshape(len(lon), len(lat)).T

            # 计算涡度
            vorticity = _calculate_vorticity_python(u, v, dx, dy)

            # 可视化
            _plot_vorticity_divergence(lon, lat, vorticity, divergence, output_path)

            # 统计计算
            vort_stats = _compute_vorticity_stats(vorticity)
            div_stats = _compute_divergence_stats(divergence)

            return {
                "success": True,
                "message": "涡度散度场计算完成（oceansim后端）",
                "output_path": output_path,
                "statistics": {
                    "vorticity": vort_stats,
                    "divergence": div_stats
                }
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"涡度散度场计算失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }


def calculate_flow_statistics(input_data):
    """计算流速统计分布 - 使用oceansim增强功能"""
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        time_index = params.get('time_index', 0)
        depth_index = params.get('depth_index', 0)

        print(f"[INFO] 使用oceansim计算流速统计: 时间{time_index}, 深度{depth_index}")

        handler = NetCDFHandler(netcdf_path)
        try:
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=depth_index)

            print(f"[DEBUG] u.shape = {u.shape}, 坐标长度: lat={len(lat)}, lon={len(lon)}")

            # 基础流速计算
            speed = np.sqrt(u**2 + v**2)
            direction = np.arctan2(v, u)
            valid_mask = np.isfinite(speed)
            valid_speed = speed[valid_mask]
            valid_direction = direction[valid_mask]

            # 自动测试并找到兼容的配置
            test_result = check_grid_data_formats(u, v, lat, lon)
            if not test_result["success"]:
                raise ValueError(f"无法找到兼容的网格数据格式: {test_result.get('error', '未知错误')}")

            # 使用测试成功的配置
            grid_params = test_result["grid_params"]
            data_format = test_result["data_format"]

            print(f"[INFO] 使用配置: 网格{grid_params}, 数据格式{data_format}")

            grid = oceansim.GridDataStructure(*grid_params)
            params_obj = oceansim.PhysicalParameters()
            current_solver = oceansim.CurrentFieldSolver(grid, params_obj)

            # 根据测试结果准备数据
            u_data = test_result.get("u_data")
            v_data = test_result.get("v_data")
            if u_data is None or v_data is None:
                u_data = u.astype(np.float64)
                v_data = v.astype(np.float64)

            # 设置速度场
            grid.add_field2d("u_velocity", u_data)
            grid.add_field2d("v_velocity", v_data)

            # 使用C++计算动能和其他指标
            try:
                kinetic_energy_field = current_solver.compute_kinetic_energy()
                total_energy = current_solver.compute_total_energy()
                mass_imbalance = current_solver.compute_mass_imbalance()
                kinetic_energy = float(np.mean(kinetic_energy_field))
            except Exception as e:
                print(f"[WARNING] C++高级计算失败: {e}")
                kinetic_energy = float(0.5 * 1025 * np.mean(valid_speed**2))
                total_energy = kinetic_energy
                mass_imbalance = 0.0

            # 基础统计
            flow_stats = {
                "mean_speed": float(np.mean(valid_speed)),
                "max_speed": float(np.max(valid_speed)),
                "speed_standard_deviation": float(np.std(valid_speed)),
                "dominant_direction": float(np.degrees(np.median(valid_direction))),
                "kinetic_energy_density": kinetic_energy
            }

            # 高级海洋学指标
            oceanographic_metrics = {
                "total_energy": total_energy,
                "mass_conservation": {
                    "mass_imbalance": mass_imbalance,
                    "conservation_quality": "良好" if abs(mass_imbalance) < 1e-6 else "需要关注"
                },
                "geophysical_parameters": _compute_geophysical_parameters(lat, valid_speed)
            }

            return {
                "success": True,
                "message": "流速统计计算完成（oceansim增强）",
                "statistics": {
                    "flow_statistics": flow_stats,
                    "oceanographic_metrics": oceanographic_metrics
                }
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"流速统计计算失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }

def _calculate_vorticity_python(u, v, dx, dy):
    """Python实现的涡度计算（因C++接口中未找到涡度函数）"""
    vorticity = np.zeros_like(u)

    for i in range(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            dvdx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
            dudy = (u[i+1, j] - u[i-1, j]) / (2 * dy)
            vorticity[i, j] = dvdx - dudy

    return vorticity


def _compute_geophysical_parameters(lat, valid_speed):
    """计算地球物理参数"""
    try:
        # 创建物理参数对象
        params = oceansim.PhysicalParameters()

        # 计算中心纬度的科里奥利参数
        center_lat = lat[len(lat)//2]
        # 注意：需要确认PhysicalParameters是否有计算科里奥利参数的方法
        # 从接口文档看到有coriolis_f属性，但可能需要设置纬度

        # 暂时使用经典公式计算科里奥利参数
        omega_earth = 7.2921e-5  # 地球自转角速度
        coriolis_f = 2 * omega_earth * np.sin(np.radians(center_lat))

        # 计算罗斯贝数（特征速度/科里奥利参数/特征长度）
        characteristic_speed = np.mean(valid_speed)
        characteristic_length = 100000  # 假设特征长度100km
        rossby_number = characteristic_speed / (abs(coriolis_f) * characteristic_length) if abs(coriolis_f) > 1e-10 else 0.0

        return {
            "latitude": float(center_lat),
            "coriolis_parameter": float(coriolis_f),
            "rossby_number": float(rossby_number),
            "characteristic_speed": float(characteristic_speed)
        }

    except Exception as e:
        print(f"[WARNING] 地球物理参数计算失败: {e}")
        return {
            "latitude": float(lat[len(lat)//2]),
            "coriolis_parameter": 0.0,
            "rossby_number": 0.0,
            "characteristic_speed": 0.0
        }


def _generate_levels(field, num_levels=21):
    """根据数据生成稳定的等值线级别，避免级别非递增导致绘图错误"""
    finite_vals = field[np.isfinite(field)]
    if finite_vals.size == 0:
        return np.linspace(-1, 1, num_levels)

    low = np.nanpercentile(finite_vals, 5)
    high = np.nanpercentile(finite_vals, 95)

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        span = abs(low) if np.isfinite(low) else 1.0
        low -= span * 0.1
        high += span * 0.1
        if low == high:
            low -= 1e-6
            high += 1e-6

    if high < low:
        low, high = high, low
    if high == low:
        high += 1e-6

    return np.linspace(low, high, num_levels)


def _plot_vorticity_divergence(lon, lat, vorticity, divergence, output_path):
    """绘制涡度和散度场"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from Source.PythonEngine.utils.chinese_config import setup_chinese_all

    setup_chinese_all(font_size=12, dpi=120)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                   subplot_kw={'projection': ccrs.PlateCarree()})

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 涡度场
    vort_levels = _generate_levels(vorticity)
    cs1 = ax1.contourf(lon_grid, lat_grid, vorticity,
                       levels=vort_levels, cmap='RdBu_r',
                       transform=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.set_title('相对涡度场 (s^-1)')
    plt.colorbar(cs1, ax=ax1, orientation='horizontal', shrink=0.8)

    # 散度场（oceansim计算）
    # 为散度场使用更细致的级别，放大小数值的差异
    div_finite = divergence[np.isfinite(divergence)]
    if len(div_finite) > 0:
        # 使用更小的百分位数范围来突出细微差异
        div_low = np.percentile(div_finite, 1)   # 从5%改为1%
        div_high = np.percentile(div_finite, 99) # 从95%改为99%
        # 如果值域太小，人为扩大到合理范围
        if abs(div_high - div_low) < 1e-7:
            center = (div_high + div_low) / 2
            div_low = center - 5e-8
            div_high = center + 5e-8
        div_levels = np.linspace(div_low, div_high, 31)  # 增加级别数量
    else:
        div_levels = np.linspace(-5e-8, 5e-8, 31)

    # 临时水体掩膜
    valid_ocean_mask = ~np.isnan(vorticity)
    
    # 把散度场只在水体区域显示
    divergence_masked = np.where(valid_ocean_mask, divergence, np.nan)
    
    cs2 = ax2.contourf(lon_grid, lat_grid, divergence_masked,
                       levels=div_levels, cmap='RdYlBu_r',
                       transform=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.set_title('散度场 (s^-1) - oceansim计算')
    plt.colorbar(cs2, ax=ax2, orientation='horizontal', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _compute_vorticity_stats(vorticity):
    """计算涡度统计"""
    valid_vort = vorticity[np.isfinite(vorticity)]

    if len(valid_vort) == 0:
        return {"mean_vorticity": 0, "max_vorticity": 0, "min_vorticity": 0,
                "vorticity_variance": 0, "cyclone_count": 0, "anticyclone_count": 0}

    cyclone_threshold = np.percentile(valid_vort, 90)
    anticyclone_threshold = np.percentile(valid_vort, 10)

    return {
        "mean_vorticity": float(np.mean(valid_vort)),
        "max_vorticity": float(np.max(valid_vort)),
        "min_vorticity": float(np.min(valid_vort)),
        "vorticity_variance": float(np.var(valid_vort)),
        "cyclone_count": int(np.sum(vorticity > cyclone_threshold)),
        "anticyclone_count": int(np.sum(vorticity < anticyclone_threshold))
    }


def _compute_divergence_stats(divergence):
    """计算散度统计"""
    valid_div = divergence[np.isfinite(divergence)]

    if len(valid_div) == 0:
        return {"mean_divergence": 0, "max_divergence": 0, "min_divergence": 0,
                "convergence_zones": 0, "divergence_zones": 0}

    return {
        "mean_divergence": float(np.mean(valid_div)),
        "max_divergence": float(np.max(valid_div)),
        "min_divergence": float(np.min(valid_div)),
        "convergence_zones": int(np.sum(divergence < -1e-5)),
        "divergence_zones": int(np.sum(divergence > 1e-5))
    }


def validate_particle_positions_and_time(netcdf_path, initial_positions, time_index=0, simulation_days=1):
    """
    验证粒子位置是否在水域内，并检查时间范围
    """
    try:
        handler = NetCDFHandler(netcdf_path)
        try:
            # 获取时间信息
            ds = handler.ds
            total_time_steps = ds.sizes.get('time', 1)

            # 验证时间范围
            max_available_days = total_time_steps - time_index
            if time_index >= total_time_steps:
                return {
                    "success": False,
                    "error": f"起始时间索引{time_index}超出数据范围(0-{total_time_steps-1})"
                }

            if simulation_days > max_available_days:
                return {
                    "success": False,
                    "error": f"模拟天数{simulation_days}超出可用数据范围，最多可模拟{max_available_days}天",
                    "max_days": max_available_days,
                    "time_info": {
                        "total_time_steps": total_time_steps,
                        "start_index": time_index,
                        "available_days": max_available_days
                    }
                }

            # 获取速度场创建水域掩膜
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=0)
            water_mask = ~np.isnan(u) & ~np.isnan(v) & np.isfinite(u) & np.isfinite(v)

            # 验证粒子位置
            valid_positions = []
            invalid_positions = []

            for i, pos in enumerate(initial_positions):
                lon_val, lat_val = float(pos[0]), float(pos[1])

                # 检查是否在地理范围内
                if (lon_val < lon.min() or lon_val > lon.max() or
                        lat_val < lat.min() or lat_val > lat.max()):
                    invalid_positions.append({
                        "index": i,
                        "position": [lon_val, lat_val],
                        "reason": "超出数据地理范围"
                    })
                    continue

                # 转换为网格索引
                lon_idx = np.argmin(np.abs(lon - lon_val))
                lat_idx = np.argmin(np.abs(lat - lat_val))

                # 检查是否在水域
                if water_mask[lat_idx, lon_idx]:
                    valid_positions.append([lon_val, lat_val])
                else:
                    # 尝试寻找附近的水域点
                    found_water = False
                    search_radius = 3  # 搜索半径（网格点）

                    for di in range(-search_radius, search_radius + 1):
                        for dj in range(-search_radius, search_radius + 1):
                            new_lat_idx = lat_idx + di
                            new_lon_idx = lon_idx + dj

                            if (0 <= new_lat_idx < len(lat) and
                                    0 <= new_lon_idx < len(lon) and
                                    water_mask[new_lat_idx, new_lon_idx]):

                                suggested_pos = [float(lon[new_lon_idx]), float(lat[new_lat_idx])]
                                invalid_positions.append({
                                    "index": i,
                                    "position": [lon_val, lat_val],
                                    "reason": "位于陆地区域",
                                    "suggested_position": suggested_pos,
                                    "distance_km": np.sqrt(
                                        ((lon[new_lon_idx] - lon_val) * 111.32 * np.cos(np.radians(lat_val)))**2 +
                                        ((lat[new_lat_idx] - lat_val) * 111.32)**2
                                    )
                                })
                                found_water = True
                                break
                        if found_water:
                            break

                    if not found_water:
                        invalid_positions.append({
                            "index": i,
                            "position": [lon_val, lat_val],
                            "reason": "位于陆地区域且附近无水域"
                        })

            return {
                "success": len(invalid_positions) == 0,
                "valid_positions": valid_positions,
                "invalid_positions": invalid_positions,
                "time_info": {
                    "total_time_steps": total_time_steps,
                    "max_available_days": max_available_days,
                    "start_index": time_index
                },
                "message": f"验证完成: {len(valid_positions)}个有效位置, {len(invalid_positions)}个无效位置"
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "error": f"位置和时间验证失败: {str(e)}"
        }

def simulate_particle_tracking(input_data):
    """拉格朗日粒子追踪模拟"""
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        time_index = params.get('time_index', 0)
        depth_index = params.get('depth_index', 0)
        dt = params.get('dt', 3600.0)
        steps = params.get('steps', 24)
        initial_positions = np.array(params.get('initial_positions', []), dtype=float)
        output_path = params.get('output_path', 'particle_tracks.png')

        # 计算模拟天数
        simulation_days = (steps * dt) / (24 * 3600)

        if initial_positions.size == 0:
            raise ValueError('initial_positions 不能为空')

        

        if initial_positions.size == 0:
            raise ValueError('initial_positions 不能为空')

        print(f"[INFO] 验证粒子位置和时间范围...")

        # 新增：验证粒子位置和时间范围
        validation_result = validate_particle_positions_and_time(
            netcdf_path, initial_positions, time_index, simulation_days
        )

        if not validation_result["success"]:
            return {
                "success": False,
                "message": validation_result.get("error", "验证失败"),
                "validation_details": validation_result,
                "suggested_alternatives": validation_result.get("invalid_positions", [])
            }

        # 如果有无效位置，返回详细信息
        if validation_result.get("invalid_positions"):
            return {
                "success": False,
                "message": "部分粒子位置无效",
                "validation_details": validation_result,
                "invalid_positions": validation_result["invalid_positions"]
            }



        print(f"[INFO] 位置验证通过，运行粒子追踪模拟: 时间索引{time_index}, 深度索引{depth_index}")

        handler = NetCDFHandler(netcdf_path)
        try:
            u, v, lat, lon = handler.get_uv(time_idx=time_index, depth_idx=depth_index)

            test_result = check_grid_data_formats(u, v, lat, lon)
            if not test_result["success"]:
                raise ValueError(f"无法找到兼容的网格数据格式: {test_result.get('error', '未知错误')}")

            grid_params = test_result["grid_params"]
            u_data = test_result.get("u_data")
            v_data = test_result.get("v_data")
            if u_data is None or v_data is None:
                u_data = u.astype(np.float64)
                v_data = v.astype(np.float64)

            grid = oceansim.GridDataStructure(*grid_params)
            grid.add_field2d("u_velocity", u_data)
            grid.add_field2d("v_velocity", v_data)
            try:
                grid.add_vector_field("velocity", [u_data, v_data, np.zeros_like(u_data)])
            except Exception:
                pass

            rk_solver = oceansim.RungeKuttaSolver()
            simulator = oceansim.ParticleSimulator(grid, rk_solver)

            lon0, lat0 = float(lon.min()), float(lat.min())
            dx = float(lon[1]-lon[0]) if len(lon) > 1 else 1.0
            dy = float(lat[1]-lat[0]) if len(lat) > 1 else 1.0

            init_particles = []
            for p in initial_positions:
                ix = (p[0] - lon0) / dx
                iy = (p[1] - lat0) / dy
                init_particles.append([ix, iy, 0.0])

            simulator.initialize_particles(init_particles)

            trajectories = []
            for _ in range(steps):
                simulator.step_forward(dt)
                parts = simulator.get_particles()
                frame = []
                for pt in parts:
                    x = lon0 + pt.position[0] * dx
                    y = lat0 + pt.position[1] * dy
                    frame.append([x, y])
                trajectories.append(frame)

            _plot_particle_tracks(trajectories, lon, lat, output_path)

            return {
                "success": True,
                "message": "粒子追踪模拟完成",
                "output_path": output_path,
                "trajectories": trajectories,
            }
        finally:
            handler.close()
    except Exception as e:
        return {
            "success": False,
            "message": f"粒子追踪模拟失败: {str(e)}",
            "error_trace": traceback.format_exc(),
        }


def _plot_particle_tracks(trajectories, lon, lat, output_path):
    """绘制粒子轨迹"""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from Source.PythonEngine.utils.chinese_config import setup_chinese_all

    setup_chinese_all(font_size=12, dpi=120)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())])

    for traj in trajectories:
        arr = np.array(traj)
        ax.plot(arr[:,0], arr[:,1], '-', transform=ccrs.PlateCarree(), linewidth=1)
        ax.plot(arr[0,0], arr[0,1], 'go', markersize=3, transform=ccrs.PlateCarree())
        ax.plot(arr[-1,0], arr[-1,1], 'ro', markersize=3, transform=ccrs.PlateCarree())

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("用法: python ocean_data_wrapper.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        print(f"[INFO] 处理请求类型: {input_data.get('action', '未知')}")

        action = input_data.get('action', '')

        if action == 'load_netcdf':
            result = load_netcdf_data(input_data)
        elif action == 'plot_vector_field':
            result = plot_vector_field(input_data)
        elif action == 'export_vector_shapefile':
            result = export_vector_shapefile(input_data)
        elif action == 'get_statistics':
            result = get_statistics(input_data)
        elif action == 'create_ocean_animation':
            result = create_ocean_animation(input_data)
        elif action == 'calculate_vorticity_divergence':  # 新增统计分析功能
            result = calculate_vorticity_divergence(input_data)
        elif action == 'calculate_flow_statistics':  # 新增流速统计功能
            result = calculate_flow_statistics(input_data)
        elif action == 'validate_particle_setup':
            # 新增：验证粒子设置和时间范围
            result = validate_particle_positions_and_time(
                input_data['parameters']['netcdf_path'],
                input_data['parameters']['initial_positions'],
                input_data['parameters'].get('time_index', 0),
                input_data['parameters'].get('simulation_days', 1)
            )
        elif action == 'get_time_range':
            # 新增：获取NetCDF时间范围信息
            try:
                netcdf_path = input_data['parameters']['netcdf_path']
                handler = NetCDFHandler(netcdf_path)
                try:
                    ds = handler.ds
                    total_time_steps = ds.sizes.get('time', 1)

                    result = {
                        "success": True,
                        "time_info": {
                            "total_time_steps": total_time_steps,
                            "max_simulation_days": total_time_steps,
                            "time_step_hours": 24  # 假设每步为一天
                        }
                    }
                finally:
                    handler.close()
            except Exception as e:
                result = {"success": False, "error": str(e)}
        else:
            result = {
                "success": False,
                "message": f"未知的请求类型: {action}"
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nan_to_none(result), f, ensure_ascii=False, indent=2)

        print(f"[INFO] 处理完成: {result.get('message', '未知结果')}")
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
            pass

        print(f"[ERROR] 错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import json
    import os

    print("=" * 60)


    print("🌊 海洋数据统计分析模块测试")
    print("=" * 60)

    # 测试数据路径
    test_netcdf_path = "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/data/raw_data/merged_data.nc"

    # 检查测试文件是否存在
    if not os.path.exists(test_netcdf_path):
        print(f"❌ 错误: 测试文件不存在 - {test_netcdf_path}")
        exit(1)

    print(f"📁 测试数据文件: {os.path.basename(test_netcdf_path)}")
    print(f"📏 文件大小: {os.path.getsize(test_netcdf_path) / (1024*1024):.1f} MB")
    print()

    # ========== 测试1: 涡度散度场计算 ==========
    print("🔄 测试1: 涡度散度场计算")
    print("-" * 40)

    test_input_vort = {
        "action": "calculate_vorticity_divergence",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "output_path": "test_outputs/vorticity_divergence_analysis.png",
            "time_index": 0,
            "depth_index": 0
        }
    }

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    print(f"⚙️  测试参数:")
    print(f"   时间索引: {test_input_vort['parameters']['time_index']}")
    print(f"   深度索引: {test_input_vort['parameters']['depth_index']}")
    print(f"   输出路径: {test_input_vort['parameters']['output_path']}")
    print()

    result_vort = calculate_vorticity_divergence(test_input_vort)

    print("📊 涡度散度场计算结果:")
    if result_vort["success"]:
        print("✅ 计算成功")
        print(f"📈 输出图像: {result_vort.get('output_path', '未生成')}")

        # 显示统计信息
        stats = result_vort.get("statistics", {})
        if "vorticity" in stats:
            vort_stats = stats["vorticity"]
            print(f"🌀 涡度统计:")
            print(f"   平均涡度: {vort_stats.get('mean_vorticity', 0):.6e} s⁻¹")
            print(f"   最大涡度: {vort_stats.get('max_vorticity', 0):.6e} s⁻¹")
            print(f"   最小涡度: {vort_stats.get('min_vorticity', 0):.6e} s⁻¹")
            print(f"   气旋区域: {vort_stats.get('cyclone_count', 0)} 个")
            print(f"   反气旋区域: {vort_stats.get('anticyclone_count', 0)} 个")

        if "divergence" in stats:
            div_stats = stats["divergence"]
            print(f"📐 散度统计:")
            print(f"   平均散度: {div_stats.get('mean_divergence', 0):.6e} s⁻¹")
            print(f"   最大散度: {div_stats.get('max_divergence', 0):.6e} s⁻¹")
            print(f"   最小散度: {div_stats.get('min_divergence', 0):.6e} s⁻¹")
            print(f"   辐合区域: {div_stats.get('convergence_zones', 0)} 个")
            print(f"   辐散区域: {div_stats.get('divergence_zones', 0)} 个")

        # 检查输出文件
        output_path = result_vort.get('output_path')
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"💾 生成文件大小: {file_size / 1024:.1f} KB")
    else:
        print("❌ 计算失败")
        print(f"错误信息: {result_vort.get('message', '未知错误')}")

    print("\n" + "=" * 60)

    # ========== 测试2: 流速统计分析 ==========
    print("🔄 测试2: 流速统计分析")
    print("-" * 40)

    test_input_flow = {
        "action": "calculate_flow_statistics",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "time_index": 0,
            "depth_index": 0
        }
    }

    print(f"⚙️  测试参数:")
    print(f"   时间索引: {test_input_flow['parameters']['time_index']}")
    print(f"   深度索引: {test_input_flow['parameters']['depth_index']}")
    print()

    result_flow = calculate_flow_statistics(test_input_flow)

    print("📊 流速统计分析结果:")
    if result_flow["success"]:
        print("✅ 计算成功")

        # 显示基础流速统计
        stats = result_flow.get("statistics", {})
        if "flow_statistics" in stats:
            flow_stats = stats["flow_statistics"]
            print(f"🌊 基础流速统计:")
            print(f"   平均流速: {flow_stats.get('mean_speed', 0):.4f} m/s")
            print(f"   最大流速: {flow_stats.get('max_speed', 0):.4f} m/s")
            print(f"   流速标准差: {flow_stats.get('speed_standard_deviation', 0):.4f} m/s")
            print(f"   主导方向: {flow_stats.get('dominant_direction', 0):.1f}°")
            print(f"   动能密度: {flow_stats.get('kinetic_energy_density', 0):.2f} J/m³")

        # 显示高级海洋学指标
        if "oceanographic_metrics" in stats:
            ocean_metrics = stats["oceanographic_metrics"]
            print(f"🌍 高级海洋学指标:")
            print(f"   总能量: {ocean_metrics.get('total_energy', 0):.2f} J/m³")

            # 质量守恒分析
            mass_conservation = ocean_metrics.get("mass_conservation", {})
            print(f"⚖️  质量守恒分析:")
            print(f"   质量不平衡: {mass_conservation.get('mass_imbalance', 0):.2e}")
            print(f"   守恒质量: {mass_conservation.get('conservation_quality', '未知')}")

            # 地球物理参数
            geo_params = ocean_metrics.get("geophysical_parameters", {})
            print(f"🌐 地球物理参数:")
            print(f"   纬度: {geo_params.get('latitude', 0):.2f}°")
            print(f"   科里奥利参数: {geo_params.get('coriolis_parameter', 0):.2e} s⁻¹")
            print(f"   罗斯贝数: {geo_params.get('rossby_number', 0):.4f}")
            print(f"   特征流速: {geo_params.get('characteristic_speed', 0):.4f} m/s")
    else:
        print("❌ 计算失败")
        print(f"错误信息: {result_flow.get('message', '未知错误')}")

    print("\n" + "=" * 60)

    # ========== 测试3: 拉格朗日粒子追踪 ==========
    print("🔄 测试3: 拉格朗日粒子追踪")
    print("-" * 40)

    # 选择两个初始粒子位置用于示例
    handler = NetCDFHandler(test_netcdf_path)
    u_tmp, v_tmp, lat_tmp, lon_tmp = handler.get_uv(time_idx=0, depth_idx=0)
    handler.close()
    init_positions = [
        [float(lon_tmp[len(lon_tmp)//2]), float(lat_tmp[len(lat_tmp)//2])],
        [float(lon_tmp[len(lon_tmp)//3]), float(lat_tmp[len(lat_tmp)//3])]
    ]

    test_input_particles = {
        "action": "simulate_particle_tracking",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "time_index": 0,
            "depth_index": 0,
            "initial_positions": init_positions,
            "dt": 3600.0,
            "steps": 12,
            "output_path": "test_outputs/particle_tracks.png"
        }
    }

    print(f"⚙️  粒子数: {len(init_positions)}, 步数: {test_input_particles['parameters']['steps']}")

    result_particles = simulate_particle_tracking(test_input_particles)

    print("📊 粒子追踪结果:")
    if result_particles["success"]:
        print("✅ 模拟成功")
        print(f"📈 轨迹图: {result_particles.get('output_path', '未生成')}")
    else:
        print("❌ 模拟失败")
        print(f"错误信息: {result_particles.get('message', '未知错误')}")

    print("\n" + "=" * 60)
    print("🎯 测试完成总结")
    print("-" * 40)

    # 测试结果总结
    vort_success = result_vort.get("success", False)
    flow_success = result_flow.get("success", False)
    particle_success = result_particles.get("success", False)

    print(f"涡度散度场计算: {'✅ 成功' if vort_success else '❌ 失败'}")
    print(f"流速统计分析: {'✅ 成功' if flow_success else '❌ 失败'}")
    print(f"粒子追踪模拟: {'✅ 成功' if particle_success else '❌ 失败'}")

    if vort_success and flow_success and particle_success:
        print("\n🎉 所有测试通过！海洋统计分析模块运行正常。")

        # 显示生成的文件
        print("\n📁 生成的输出文件:")
        output_dir = "test_outputs"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size_kb = os.path.getsize(file_path) / 1024
                    print(f"   {file} ({size_kb:.1f} KB)")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复相关问题。")

    print("=" * 60)