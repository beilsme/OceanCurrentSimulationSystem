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

            total_time_steps = ds.dims['time']
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
        elif action == 'create_ocean_animation':  # 新增的动画功能
            result = create_ocean_animation(input_data)
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
    main()