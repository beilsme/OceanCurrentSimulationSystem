# ==============================================================================
# wrappers/netcdf_particle_wrapper.py
# ==============================================================================
"""
NetCDF粒子轨迹包装器 - 面向C#的完整封装
基于您提供的台湾海峡粒子漂移代码改进
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import os
import sys


# matplotlib 中文设置
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = [
    'PingFang SC', 'Hiragino Sans GB', 'STHeiti',  # Mac字体
    'SimHei', 'Microsoft YaHei',  # Windows字体
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC',  # Linux字体
    'Arial Unicode MS', 'DejaVu Sans'  # 备用字体
]
plt.rcParams['axes.unicode_minus'] = False

class NetCDFParticleTracker:
    """NetCDF粒子追踪器 - 完整的C#友好接口"""

    def __init__(self, netcdf_path: str):
        """
        初始化NetCDF粒子追踪器
        
        Args:
            netcdf_path: NetCDF文件路径
        """
        self.netcdf_path = netcdf_path
        self.nc_data = None
        self.lat = None
        self.lon = None
        self.time = None
        self.water_u = None
        self.water_v = None
        self.time_readable = None
        self.is_initialized = False

    def initialize(self, lon_range: Tuple[float, float] = (118, 124),
                   lat_range: Tuple[float, float] = (21, 26.5)) -> Dict[str, Any]:
        """
        初始化数据加载
        
        Args:
            lon_range: 经度范围 (min, max)
            lat_range: 纬度范围 (min, max)
            
        Returns:
            初始化结果字典
        """
        try:
            logging.info(f"初始化NetCDF数据: {self.netcdf_path}")

            # 加载NetCDF文件
            self.nc_data = Dataset(self.netcdf_path, mode='r')

            # 提取基础变量
            self.lat = self.nc_data.variables['lat'][:]
            self.lon = self.nc_data.variables['lon'][:]
            self.time = self.nc_data.variables['time'][:]
            self.water_u = self.nc_data.variables['water_u'][:]
            self.water_v = self.nc_data.variables['water_v'][:]

            # 时间处理
            time_units = self.nc_data.variables['time'].units
            calendar = (self.nc_data.variables['time'].calendar
                        if 'calendar' in self.nc_data.variables['time'].ncattrs()
                        else 'standard')
            self.time_readable = num2date(self.time, units=time_units, calendar=calendar)

            # 应用地理范围过滤
            lon_min, lon_max = lon_range
            lat_min, lat_max = lat_range

            self.lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
            self.lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)

            self.lon_filtered = self.lon[self.lon_mask]
            self.lat_filtered = self.lat[self.lat_mask]

            # 过滤速度场数据
            self.water_u_filtered = self.water_u[:, :, self.lat_mask, :][:, :, :, self.lon_mask]
            self.water_v_filtered = self.water_v[:, :, self.lat_mask, :][:, :, :, self.lon_mask]

            self.is_initialized = True

            logging.info("NetCDF数据初始化成功")

            return {
                "success": True,
                "message": "NetCDF数据初始化成功",
                "data_info": {
                    "time_steps": len(self.time),
                    "lat_points": len(self.lat_filtered),
                    "lon_points": len(self.lon_filtered),
                    "depth_levels": self.water_u.shape[1],
                    "time_range": [str(self.time_readable[0]), str(self.time_readable[-1])],
                    "geographic_range": {
                        "lat_min": float(self.lat_filtered.min()),
                        "lat_max": float(self.lat_filtered.max()),
                        "lon_min": float(self.lon_filtered.min()),
                        "lon_max": float(self.lon_filtered.max())
                    }
                }
            }

        except Exception as e:
            logging.error(f"NetCDF数据初始化失败: {e}")
            return {
                "success": False,
                "message": f"NetCDF数据初始化失败: {str(e)}",
                "error_trace": traceback.format_exc()
            }

    def create_interpolator(self, data: np.ndarray, method: str = 'linear') -> RegularGridInterpolator:
        """
        创建插值器
        
        Args:
            data: 2D数据数组
            method: 插值方法
            
        Returns:
            插值器对象
        """
        return RegularGridInterpolator(
            (self.lat_filtered, self.lon_filtered),
            data,
            bounds_error=False,
            fill_value=0,
            method=method
        )

    def track_single_particle(self, start_lat: float, start_lon: float,
                              time_step_hours: float = 3.0,
                              max_time_steps: Optional[int] = None,
                              depth_level: int = 0) -> Dict[str, Any]:
        """
        追踪单个粒子轨迹
        
        Args:
            start_lat: 起始纬度
            start_lon: 起始经度
            time_step_hours: 时间步长（小时）
            max_time_steps: 最大时间步数
            depth_level: 深度层级
            
        Returns:
            追踪结果字典
        """
        if not self.is_initialized:
            return {"success": False, "message": "数据未初始化"}

        try:
            logging.info(f"开始追踪粒子: 起点({start_lat:.3f}, {start_lon:.3f})")

            # 初始化轨迹数组
            track_lat = [start_lat]
            track_lon = [start_lon]
            track_time = [self.time_readable[0]]
            velocities = []

            current_lat = start_lat
            current_lon = start_lon

            # 地理边界
            lat_min, lat_max = self.lat_filtered.min(), self.lat_filtered.max()
            lon_min, lon_max = self.lon_filtered.min(), self.lon_filtered.max()

            # 确定最大时间步数
            if max_time_steps is None:
                max_time_steps = len(self.time)
            else:
                max_time_steps = min(max_time_steps, len(self.time))

            # 时间步进模拟
            for time_idx in range(max_time_steps):
                try:
                    # 获取当前时间步的速度场
                    u = self.water_u_filtered[time_idx, depth_level, :, :].copy()
                    v = self.water_v_filtered[time_idx, depth_level, :, :].copy()

                    # 处理掩码数组
                    if isinstance(u, np.ma.MaskedArray):
                        u = u.filled(0)
                    if isinstance(v, np.ma.MaskedArray):
                        v = v.filled(0)

                    # 创建插值器
                    u_interp = self.create_interpolator(u, method='linear')
                    v_interp = self.create_interpolator(v, method='linear')

                    # 获取当前位置的流速
                    current_u = float(u_interp([[current_lat, current_lon]])[0])
                    current_v = float(v_interp([[current_lat, current_lon]])[0])

                    # 记录速度
                    velocities.append({
                        "u": current_u,
                        "v": current_v,
                        "speed": np.sqrt(current_u**2 + current_v**2)
                    })

                    # 计算位置变化
                    time_step_seconds = time_step_hours * 3600
                    delta_lat = current_v * time_step_seconds / 111000  # 转换为度
                    delta_lon = current_u * time_step_seconds / (111000 * np.cos(np.radians(current_lat)))

                    # 更新位置
                    new_lat = current_lat + delta_lat
                    new_lon = current_lon + delta_lon

                    # 边界检查
                    if (new_lat < lat_min or new_lat > lat_max or
                            new_lon < lon_min or new_lon > lon_max):
                        logging.info(f"粒子到达边界，时间步 {time_idx}，终止模拟")
                        break

                    # 更新位置
                    current_lat = new_lat
                    current_lon = new_lon
                    track_lat.append(current_lat)
                    track_lon.append(current_lon)

                    # 计算对应的时间
                    if time_idx + 1 < len(self.time_readable):
                        track_time.append(self.time_readable[time_idx + 1])

                except Exception as e:
                    logging.warning(f"时间步 {time_idx} 计算错误: {e}")
                    break

            # 计算统计信息
            total_distance = self._calculate_total_distance(track_lat, track_lon)
            direct_distance = self._haversine_distance(
                track_lat[0], track_lon[0], track_lat[-1], track_lon[-1]
            )

            avg_speed = np.mean([v["speed"] for v in velocities]) if velocities else 0
            max_speed = np.max([v["speed"] for v in velocities]) if velocities else 0

            logging.info(f"粒子追踪完成: {len(track_lat)} 个时间步")

            return {
                "success": True,
                "message": "粒子追踪完成",
                "trajectory": {
                    "latitudes": track_lat,
                    "longitudes": track_lon,
                    "times": [str(t) for t in track_time],
                    "velocities": velocities
                },
                "statistics": {
                    "total_points": len(track_lat),
                    "total_distance_km": total_distance,
                    "direct_distance_km": direct_distance,
                    "avg_speed_ms": avg_speed,
                    "max_speed_ms": max_speed,
                    "simulation_hours": len(track_lat) * time_step_hours
                },
                "start_position": {"lat": start_lat, "lon": start_lon},
                "end_position": {"lat": track_lat[-1], "lon": track_lon[-1]}
            }

        except Exception as e:
            logging.error(f"粒子追踪失败: {e}")
            return {
                "success": False,
                "message": f"粒子追踪失败: {str(e)}",
                "error_trace": traceback.format_exc()
            }

    def track_multiple_particles(self, start_positions: List[Tuple[float, float]],
                                 time_step_hours: float = 3.0,
                                 max_time_steps: Optional[int] = None,
                                 depth_level: int = 0) -> Dict[str, Any]:
        """
        追踪多个粒子轨迹
        
        Args:
            start_positions: 起始位置列表 [(lat, lon), ...]
            time_step_hours: 时间步长（小时）
            max_time_steps: 最大时间步数
            depth_level: 深度层级
            
        Returns:
            多粒子追踪结果
        """
        try:
            logging.info(f"开始追踪 {len(start_positions)} 个粒子")

            all_trajectories = []
            all_statistics = []
            success_count = 0

            for i, (start_lat, start_lon) in enumerate(start_positions):
                logging.info(f"追踪粒子 {i+1}/{len(start_positions)}")

                result = self.track_single_particle(
                    start_lat=start_lat,
                    start_lon=start_lon,
                    time_step_hours=time_step_hours,
                    max_time_steps=max_time_steps,
                    depth_level=depth_level
                )

                if result["success"]:
                    all_trajectories.append(result["trajectory"])
                    all_statistics.append(result["statistics"])
                    success_count += 1
                else:
                    logging.warning(f"粒子 {i} 追踪失败: {result['message']}")
                    all_trajectories.append(None)
                    all_statistics.append(None)

            return {
                "success": success_count > 0,
                "message": f"多粒子追踪完成: {success_count}/{len(start_positions)} 成功",
                "trajectories": all_trajectories,
                "statistics": all_statistics,
                "summary": {
                    "total_particles": len(start_positions),
                    "successful_particles": success_count,
                    "failed_particles": len(start_positions) - success_count
                }
            }

        except Exception as e:
            logging.error(f"多粒子追踪失败: {e}")
            return {
                "success": False,
                "message": f"多粒子追踪失败: {str(e)}",
                "error_trace": traceback.format_exc()
            }

    def create_trajectory_animation(self, trajectories: List[Dict],
                                    output_path: str,
                                    title: str = "粒子轨迹动画",
                                    show_velocities: bool = False) -> Dict[str, Any]:
        """
        创建轨迹动画
        
        Args:
            trajectories: 轨迹数据列表
            output_path: 输出路径
            title: 动画标题
            show_velocities: 是否显示速度场
            
        Returns:
            动画创建结果
        """
        try:
            logging.info("开始创建轨迹动画")

            # 过滤有效轨迹
            valid_trajectories = [t for t in trajectories if t is not None]
            if not valid_trajectories:
                return {"success": False, "message": "没有有效的轨迹数据"}

            # 计算显示范围
            all_lats = []
            all_lons = []
            for traj in valid_trajectories:
                all_lats.extend(traj["latitudes"])
                all_lons.extend(traj["longitudes"])

            lat_margin = (max(all_lats) - min(all_lats)) * 0.1
            lon_margin = (max(all_lons) - min(all_lons)) * 0.1

            extent = [
                min(all_lons) - lon_margin,
                max(all_lons) + lon_margin,
                min(all_lats) - lat_margin,
                max(all_lats) + lat_margin
            ]

            # 创建静态图像
            fig = plt.figure(figsize=(15, 12))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent)

            # 添加地图要素
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.5)
            ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

            # 绘制轨迹
            colors = plt.cm.Set1(np.linspace(0, 1, len(valid_trajectories)))

            for i, (traj, color) in enumerate(zip(valid_trajectories, colors)):
                lats = traj["latitudes"]
                lons = traj["longitudes"]

                # 绘制轨迹线
                ax.plot(lons, lats, color=color, linewidth=2,
                        label=f'粒子 {i+1}', transform=ccrs.PlateCarree())

                # 起点标记
                ax.plot(lons[0], lats[0], 'o', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1,
                        transform=ccrs.PlateCarree())

                # 终点标记
                ax.plot(lons[-1], lats[-1], 's', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1,
                        transform=ccrs.PlateCarree())

            # 设置标题和图例
            plt.title(title, pad=20, fontsize=14)
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))

            # 保存图像
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"轨迹动画保存至: {output_path}")

            return {
                "success": True,
                "message": "轨迹动画创建成功",
                "output_path": output_path,
                "animation_info": {
                    "particle_count": len(valid_trajectories),
                    "geographic_extent": extent
                }
            }

        except Exception as e:
            logging.error(f"创建轨迹动画失败: {e}")
            return {
                "success": False,
                "message": f"创建轨迹动画失败: {str(e)}",
                "error_trace": traceback.format_exc()
            }

    def _haversine_distance(self, lat1: float, lon1: float,
                            lat2: float, lon2: float) -> float:
        """计算两点间的球面距离（公里）"""
        R = 6371  # 地球半径（公里）
        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        a = (np.sin(dLat / 2) * np.sin(dLat / 2) +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dLon / 2) * np.sin(dLon / 2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _calculate_total_distance(self, lats: List[float], lons: List[float]) -> float:
        """计算轨迹总距离"""
        total_distance = 0
        for i in range(1, len(lats)):
            total_distance += self._haversine_distance(
                lats[i-1], lons[i-1], lats[i], lons[i]
            )
        return total_distance

    def close(self):
        """关闭NetCDF文件"""
        if self.nc_data:
            self.nc_data.close()
            self.is_initialized = False
            logging.info("NetCDF文件已关闭")


def netcdf_particle_tracking_wrapper(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    NetCDF粒子追踪包装器 - C#接口
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        处理结果字典
    """
    tracker = None

    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        action = params.get('action', 'track_single')

        # 地理范围参数
        lon_range = params.get('lon_range', [118, 124])
        lat_range = params.get('lat_range', [21, 26.5])

        # 模拟参数
        time_step_hours = params.get('time_step_hours', 3.0)
        max_time_steps = params.get('max_time_steps', None)
        depth_level = params.get('depth_level', 0)

        # 验证输入
        if not netcdf_path or not os.path.exists(netcdf_path):
            raise FileNotFoundError(f"NetCDF文件不存在: {netcdf_path}")

        # 初始化追踪器
        tracker = NetCDFParticleTracker(netcdf_path)
        init_result = tracker.initialize(
            lon_range=tuple(lon_range),
            lat_range=tuple(lat_range)
        )

        if not init_result["success"]:
            return init_result

        # 根据动作类型执行不同操作
        if action == 'track_single':
            start_lat = params.get('start_lat')
            start_lon = params.get('start_lon')

            if start_lat is None or start_lon is None:
                raise ValueError("单粒子追踪需要起始位置参数")

            result = tracker.track_single_particle(
                start_lat=start_lat,
                start_lon=start_lon,
                time_step_hours=time_step_hours,
                max_time_steps=max_time_steps,
                depth_level=depth_level
            )

        elif action == 'track_multiple':
            start_positions = params.get('start_positions', [])

            if not start_positions:
                raise ValueError("多粒子追踪需要起始位置列表")

            result = tracker.track_multiple_particles(
                start_positions=[(pos[0], pos[1]) for pos in start_positions],
                time_step_hours=time_step_hours,
                max_time_steps=max_time_steps,
                depth_level=depth_level
            )

        elif action == 'create_animation':
            trajectories = params.get('trajectories', [])
            output_path = params.get('output_path', 'particle_animation.png')
            title = params.get('title', '粒子轨迹动画')

            if not trajectories:
                raise ValueError("创建动画需要轨迹数据")

            result = tracker.create_trajectory_animation(
                trajectories=trajectories,
                output_path=output_path,
                title=title
            )

        else:
            raise ValueError(f"未知的动作类型: {action}")

        # 添加初始化信息到结果中
        if result.get("success"):
            result["data_info"] = init_result["data_info"]

        return result

    except Exception as e:
        logging.error(f"NetCDF粒子追踪包装器失败: {e}")
        return {
            "success": False,
            "message": f"NetCDF粒子追踪失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }

    finally:
        if tracker:
            tracker.close()


# 主要处理函数
def handle_netcdf_particle_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理NetCDF粒子追踪请求的主函数
    
    Args:
        input_data: 包含action和parameters的请求数据
        
    Returns:
        处理结果
    """
    action = input_data.get("action", "")

    if action == "netcdf_particle_tracking":
        return netcdf_particle_tracking_wrapper(input_data)
    else:
        return {
            "success": False,
            "message": f"未知的动作类型: {action}",
            "available_actions": ["netcdf_particle_tracking"]
        }


def main() -> None:
    """命令行入口: python netcdf_particle_wrapper.py input.json output.json"""
    if len(sys.argv) != 3:
        print("Usage: python netcdf_particle_wrapper.py <input.json> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    result = handle_netcdf_particle_request(input_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
        sys.exit(0)
        
    # 测试NetCDF粒子追踪包装器
    import logging

    logging.basicConfig(level=logging.INFO)

    print("🌊 测试NetCDF粒子追踪包装器")
    print("-" * 50)

    # 测试配置
    test_netcdf_path = "../data/raw_data/merged_data.nc"

    if not os.path.exists(test_netcdf_path):
        print(f"❌ 测试文件不存在: {test_netcdf_path}")
        print("请确保NetCDF文件路径正确")
        exit(1)

    # 创建输出目录
    os.makedirs("test_outputs", exist_ok=True)

    # 测试1: 单粒子追踪
    print("🎯 测试1: 单粒子追踪（台湾海峡）")
    test_input_1 = {
        "action": "netcdf_particle_tracking",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "action": "track_single",
            "start_lat": 22.5,
            "start_lon": 119.5,
            "time_step_hours": 3.0,
            "max_time_steps": 240,  # 30天
            "depth_level": 0,
            "lon_range": [118, 124],
            "lat_range": [21, 26.5]
        }
    }

    result_1 = handle_netcdf_particle_request(test_input_1)
    print(f"结果: {'✅ 成功' if result_1['success'] else '❌ 失败'}")

    if result_1['success']:
        stats = result_1['statistics']
        print(f"   轨迹点数: {stats['total_points']}")
        print(f"   总距离: {stats['total_distance_km']:.2f} km")
        print(f"   直线距离: {stats['direct_distance_km']:.2f} km")
        print(f"   平均速度: {stats['avg_speed_ms']:.3f} m/s")
        print(f"   起点: ({result_1['start_position']['lat']:.3f}, {result_1['start_position']['lon']:.3f})")
        print(f"   终点: ({result_1['end_position']['lat']:.3f}, {result_1['end_position']['lon']:.3f})")
    else:
        print(f"   错误: {result_1['message']}")

    # 测试2: 多粒子追踪
    print("\n🌊 测试2: 多粒子追踪")
    test_input_2 = {
        "action": "netcdf_particle_tracking",
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "action": "track_multiple",
            "start_positions": [
                [22.5, 119.5],   # 台湾海峡中部
                [23.0, 119.0],   # 台湾海峡北部
                [22.0, 120.0],   # 台湾海峡南部
            ],
            "time_step_hours": 3.0,
            "max_time_steps": 120,  # 15天
            "depth_level": 0
        }
    }

    result_2 = handle_netcdf_particle_request(test_input_2)
    print(f"结果: {'✅ 成功' if result_2['success'] else '❌ 失败'}")

    if result_2['success']:
        summary = result_2['summary']
        print(f"   总粒子数: {summary['total_particles']}")
        print(f"   成功追踪: {summary['successful_particles']}")
        print(f"   失败数量: {summary['failed_particles']}")
    else:
        print(f"   错误: {result_2['message']}")

    # 测试3: 创建动画（如果多粒子追踪成功）
    if result_2['success'] and result_2['trajectories']:
        print("\n🎬 测试3: 创建轨迹动画")
        test_input_3 = {
            "action": "netcdf_particle_tracking",
            "parameters": {
                "netcdf_path": test_netcdf_path,
                "action": "create_animation",
                "trajectories": result_2['trajectories'],
                "output_path": "test_outputs/taiwan_strait_particles.png",
                "title": "台湾海峡粒子漂移轨迹"
            }
        }

        result_3 = handle_netcdf_particle_request(test_input_3)
        print(f"结果: {'✅ 成功' if result_3['success'] else '❌ 失败'}")

        if result_3['success']:
            print(f"   动画文件: {result_3['output_path']}")
            info = result_3['animation_info']
            print(f"   粒子数量: {info['particle_count']}")
        else:
            print(f"   错误: {result_3['message']}")

    print("\n" + "=" * 50)
    print("🎯 NetCDF粒子追踪包装器测试完成")
    print("💡 此包装器完全兼容C#调用，避免了C++模块问题")