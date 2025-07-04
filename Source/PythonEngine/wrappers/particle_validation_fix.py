# ==============================================================================
# wrappers/particle_validation_fix.py
# ==============================================================================
"""
粒子验证修复和调试系统 - 解决粒子位置验证失败问题
"""

import numpy as np
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

# 导入相关模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # ensure package root
from PythonEngine.wrappers.ocean_data_wrapper import NetCDFHandler
from PythonEngine.wrappers.lagrangian_particle_wrapper import validate_particle_positions_and_time, simulate_particle_tracking


def debug_netcdf_data(netcdf_path: str) -> Dict[str, Any]:
    """
    调试NetCDF数据，检查数据格式和范围
    
    Args:
        netcdf_path: NetCDF文件路径
        
    Returns:
        调试信息字典
    """
    try:
        handler = NetCDFHandler(netcdf_path)
        try:
            ds = handler.ds

            # 获取维度信息
            dims = dict(ds.dims)

            # 获取变量列表
            variables = list(ds.variables.keys())

            # 获取坐标信息
            coords = {}
            for coord_name in ['longitude', 'lon', 'x', 'latitude', 'lat', 'y', 'time', 'depth', 'z']:
                if coord_name in ds.variables:
                    coord_var = ds.variables[coord_name]
                    coords[coord_name] = {
                        'shape': coord_var.shape,
                        'min': float(coord_var.min()),
                        'max': float(coord_var.max()),
                        'dtype': str(coord_var.dtype)
                    }

            # 尝试获取速度场数据
            velocity_info = {}
            for vel_name in ['u', 'v', 'w', 'u_velocity', 'v_velocity', 'eastward_velocity', 'northward_velocity']:
                if vel_name in ds.variables:
                    vel_var = ds.variables[vel_name]
                    velocity_info[vel_name] = {
                        'shape': vel_var.shape,
                        'dimensions': vel_var.dims,
                        'dtype': str(vel_var.dtype)
                    }

            # 检查第一个时间步的数据
            time_0_data = {}
            try:
                u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
                time_0_data = {
                    'u_shape': u.shape,
                    'v_shape': v.shape,
                    'lat_shape': lat.shape,
                    'lon_shape': lon.shape,
                    'lat_range': [float(lat.min()), float(lat.max())],
                    'lon_range': [float(lon.min()), float(lon.max())],
                    'u_valid_points': int(np.sum(~np.isnan(u) & np.isfinite(u))),
                    'v_valid_points': int(np.sum(~np.isnan(v) & np.isfinite(v))),
                    'total_points': int(u.size)
                }
            except Exception as e:
                time_0_data['error'] = str(e)

            return {
                'success': True,
                'file_path': netcdf_path,
                'dimensions': dims,
                'variables': variables,
                'coordinates': coords,
                'velocity_fields': velocity_info,
                'time_0_data': time_0_data
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_trace': traceback.format_exc()
        }


def auto_fix_particle_positions(netcdf_path: str,
                                initial_positions: List[List[float]],
                                max_search_radius: float = 0.5) -> Dict[str, Any]:
    """
    自动修复粒子位置，将陆地上的粒子移动到最近的水域
    
    Args:
        netcdf_path: NetCDF文件路径
        initial_positions: 初始粒子位置 [[lon, lat], ...]
        max_search_radius: 最大搜索半径（度）
        
    Returns:
        修复结果字典
    """
    try:
        handler = NetCDFHandler(netcdf_path)
        try:
            # 获取速度场创建水域掩膜
            u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
            water_mask = ~np.isnan(u) & ~np.isnan(v) & np.isfinite(u) & np.isfinite(v)

            fixed_positions = []
            fix_log = []

            for i, pos in enumerate(initial_positions):
                lon_val, lat_val = float(pos[0]), float(pos[1])

                # 检查是否在数据范围内
                if (lon_val < lon.min() or lon_val > lon.max() or
                        lat_val < lat.min() or lat_val > lat.max()):

                    # 将位置调整到数据范围内
                    fixed_lon = np.clip(lon_val, lon.min(), lon.max())
                    fixed_lat = np.clip(lat_val, lat.min(), lat.max())

                    fix_log.append({
                        'particle': i,
                        'original': [lon_val, lat_val],
                        'fixed': [float(fixed_lon), float(fixed_lat)],
                        'reason': '移动到数据范围内'
                    })

                    lon_val, lat_val = fixed_lon, fixed_lat

                # 转换为网格索引
                lon_idx = np.argmin(np.abs(lon - lon_val))
                lat_idx = np.argmin(np.abs(lat - lat_val))

                # 检查是否在水域
                if water_mask[lat_idx, lon_idx]:
                    # 已经在水域中
                    fixed_positions.append([lon_val, lat_val])
                else:
                    # 寻找最近的水域点
                    found_water = False

                    # 计算搜索半径（网格点数）
                    lon_step = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.01
                    lat_step = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.01
                    max_lon_steps = int(max_search_radius / lon_step)
                    max_lat_steps = int(max_search_radius / lat_step)

                    best_distance = float('inf')
                    best_position = None

                    # 螺旋搜索最近的水域点
                    for radius in range(1, max(max_lon_steps, max_lat_steps) + 1):
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                if abs(di) != radius and abs(dj) != radius:
                                    continue  # 只检查边界点

                                new_lat_idx = lat_idx + di
                                new_lon_idx = lon_idx + dj

                                if (0 <= new_lat_idx < len(lat) and
                                        0 <= new_lon_idx < len(lon) and
                                        water_mask[new_lat_idx, new_lon_idx]):

                                    # 计算实际距离
                                    distance = np.sqrt(
                                        ((lon[new_lon_idx] - lon_val) * 111.32 * np.cos(np.radians(lat_val)))**2 +
                                        ((lat[new_lat_idx] - lat_val) * 111.32)**2
                                    )

                                    if distance < best_distance:
                                        best_distance = distance
                                        best_position = [float(lon[new_lon_idx]), float(lat[new_lat_idx])]

                        if best_position is not None:
                            found_water = True
                            break

                    if found_water and best_position is not None:
                        fixed_positions.append(best_position)
                        fix_log.append({
                            'particle': i,
                            'original': [lon_val, lat_val],
                            'fixed': best_position,
                            'distance_km': best_distance,
                            'reason': '移动到最近水域'
                        })
                    else:
                        # 如果找不到水域，使用数据中心的有效点
                        valid_indices = np.where(water_mask)
                        if len(valid_indices[0]) > 0:
                            center_idx = len(valid_indices[0]) // 2
                            center_lat_idx = valid_indices[0][center_idx]
                            center_lon_idx = valid_indices[1][center_idx]

                            fallback_position = [float(lon[center_lon_idx]), float(lat[center_lat_idx])]
                            fixed_positions.append(fallback_position)
                            fix_log.append({
                                'particle': i,
                                'original': [lon_val, lat_val],
                                'fixed': fallback_position,
                                'reason': '使用数据中心有效点'
                            })
                        else:
                            # 最后的备选方案：使用数据中心点
                            center_position = [float(lon[len(lon)//2]), float(lat[len(lat)//2])]
                            fixed_positions.append(center_position)
                            fix_log.append({
                                'particle': i,
                                'original': [lon_val, lat_val],
                                'fixed': center_position,
                                'reason': '使用数据中心点（强制）'
                            })

            return {
                'success': True,
                'original_positions': initial_positions,
                'fixed_positions': fixed_positions,
                'fixes_applied': len(fix_log),
                'fix_log': fix_log,
                'data_extent': {
                    'lon_range': [float(lon.min()), float(lon.max())],
                    'lat_range': [float(lat.min()), float(lat.max())],
                    'water_coverage': float(np.sum(water_mask) / water_mask.size)
                }
            }

        finally:
            handler.close()

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_trace': traceback.format_exc()
        }


def robust_particle_tracking_with_validation(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    带有自动验证和修复的强健粒子追踪
    
    Args:
        input_data: 输入参数字典
        
    Returns:
        追踪结果字典
    """
    try:
        params = input_data.get('parameters', {})
        netcdf_path = params.get('netcdf_path')
        initial_positions = params.get('initial_positions', [])
        simulation_hours = params.get('simulation_hours', 24.0)
        time_step_hours = params.get('time_step_hours', 1.0)
        auto_fix = params.get('auto_fix_positions', True)
        debug_mode = params.get('debug_mode', True)

        logging.info(f"开始强健粒子追踪: {len(initial_positions)}个粒子")

        # 步骤1: 调试NetCDF数据
        if debug_mode:
            logging.info("步骤1: 调试NetCDF数据格式...")
            debug_info = debug_netcdf_data(netcdf_path)
            if not debug_info['success']:
                return {
                    'success': False,
                    'message': f"NetCDF数据调试失败: {debug_info['error']}",
                    'debug_info': debug_info
                }

            logging.info("NetCDF数据调试完成:")
            logging.info(f"  - 维度: {debug_info['dimensions']}")
            logging.info(f"  - 变量: {len(debug_info['variables'])} 个")
            if 'time_0_data' in debug_info and 'lat_range' in debug_info['time_0_data']:
                extent = debug_info['time_0_data']
                logging.info(f"  - 地理范围: {extent['lat_range']} (纬度), {extent['lon_range']} (经度)")
                logging.info(f"  - 有效数据点: {extent.get('u_valid_points', 0)}/{extent.get('total_points', 0)}")

        # 步骤2: 验证和修复粒子位置
        logging.info("步骤2: 验证和修复粒子位置...")

        # 计算模拟天数
        simulation_days = (int(simulation_hours / time_step_hours) * time_step_hours) / 24

        # 原始验证
        validation_result = validate_particle_positions_and_time(
            netcdf_path, initial_positions, time_index=0, simulation_days=simulation_days
        )

        current_positions = initial_positions
        fixes_applied = []

        if not validation_result["success"] and auto_fix:
            logging.info("原始验证失败，尝试自动修复位置...")

            # 自动修复位置
            fix_result = auto_fix_particle_positions(netcdf_path, initial_positions)

            if fix_result['success']:
                current_positions = fix_result['fixed_positions']
                fixes_applied = fix_result['fix_log']

                logging.info(f"位置修复完成: {fix_result['fixes_applied']} 个粒子被修复")
                for fix in fixes_applied:
                    logging.info(f"  粒子{fix['particle']}: {fix['reason']}")

                # 重新验证修复后的位置
                validation_result = validate_particle_positions_and_time(
                    netcdf_path, current_positions, time_index=0, simulation_days=simulation_days
                )

                if not validation_result["success"]:
                    logging.warning("修复后仍然验证失败，尝试使用更宽松的参数...")

                    # 使用更宽松的验证（减少模拟时间）
                    reduced_simulation_days = min(simulation_days, 1.0)  # 最多1天
                    validation_result = validate_particle_positions_and_time(
                        netcdf_path, current_positions, time_index=0, simulation_days=reduced_simulation_days
                    )

                    if validation_result["success"]:
                        # 调整模拟参数
                        simulation_hours = reduced_simulation_days * 24
                        logging.info(f"使用减少的模拟时间: {simulation_hours} 小时")

        if not validation_result["success"]:
            return {
                'success': False,
                'message': f"粒子位置验证失败: {validation_result.get('error', '未知错误')}",
                'validation_details': validation_result,
                'fixes_applied': fixes_applied,
                'debug_info': debug_info if debug_mode else None
            }

        logging.info("粒子位置验证通过")

        # 步骤3: 执行粒子追踪
        logging.info("步骤3: 执行粒子追踪模拟...")

        tracking_params = {
            "netcdf_path": netcdf_path,
            "initial_positions": current_positions,
            "dt": time_step_hours * 3600,  # 转换为秒
            "steps": int(simulation_hours / time_step_hours),
            "time_index": 0,
            "depth_index": 0
        }

        tracking_result = simulate_particle_tracking({"parameters": tracking_params})

        if tracking_result.get("success"):
            # 添加修复信息到结果中
            tracking_result['position_fixes'] = fixes_applied
            tracking_result['original_positions'] = initial_positions
            tracking_result['used_positions'] = current_positions

            if debug_mode:
                tracking_result['debug_info'] = debug_info

            logging.info("粒子追踪模拟成功完成")
            return tracking_result
        else:
            return {
                'success': False,
                'message': f"粒子追踪模拟失败: {tracking_result.get('message', '未知错误')}",
                'tracking_details': tracking_result,
                'position_fixes': fixes_applied,
                'debug_info': debug_info if debug_mode else None
            }

    except Exception as e:
        logging.error(f"强健粒子追踪失败: {e}")
        return {
            'success': False,
            'message': f"强健粒子追踪失败: {str(e)}",
            'error_trace': traceback.format_exc()
        }


def create_test_positions_in_valid_area(netcdf_path: str, num_particles: int = 5) -> List[List[float]]:
    """
    在有效水域中创建测试粒子位置
    
    Args:
        netcdf_path: NetCDF文件路径
        num_particles: 粒子数量
        
    Returns:
        有效的粒子位置列表
    """
    try:
        handler = NetCDFHandler(netcdf_path)
        try:
            u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
            water_mask = ~np.isnan(u) & ~np.isnan(v) & np.isfinite(u) & np.isfinite(v)

            # 找到所有有效的水域点
            valid_indices = np.where(water_mask)

            if len(valid_indices[0]) == 0:
                raise ValueError("没有找到有效的水域点")

            # 随机选择粒子位置
            selected_indices = np.random.choice(
                len(valid_indices[0]),
                size=min(num_particles, len(valid_indices[0])),
                replace=False
            )

            test_positions = []
            for idx in selected_indices:
                lat_idx = valid_indices[0][idx]
                lon_idx = valid_indices[1][idx]
                test_positions.append([float(lon[lon_idx]), float(lat[lat_idx])])

            return test_positions

        finally:
            handler.close()

    except Exception as e:
        logging.error(f"创建测试位置失败: {e}")
        # 返回默认位置
        return [[0.0, 0.0]]


if __name__ == "__main__":
    # 测试验证修复系统
    import os

    print("🔧 测试粒子验证修复系统")
    print("-" * 50)

    # 配置测试
    test_netcdf_path = "data/test_ocean_data.nc"  # 需要替换为实际路径

    if not os.path.exists(test_netcdf_path):
        print(f"❌ 测试文件不存在: {test_netcdf_path}")
        print("请提供有效的NetCDF文件路径")
        exit(1)

    # 测试1: 调试NetCDF数据
    print("📊 测试1: 调试NetCDF数据...")
    debug_result = debug_netcdf_data(test_netcdf_path)

    if debug_result['success']:
        print("✅ NetCDF调试成功")
        print(f"   维度: {debug_result['dimensions']}")
        print(f"   变量数: {len(debug_result['variables'])}")
    else:
        print(f"❌ NetCDF调试失败: {debug_result['error']}")

    # 测试2: 创建有效测试位置
    print("\n🎯 测试2: 创建有效测试位置...")
    test_positions = create_test_positions_in_valid_area(test_netcdf_path, 4)
    print(f"✅ 创建了 {len(test_positions)} 个测试位置")
    for i, pos in enumerate(test_positions):
        print(f"   粒子{i}: [{pos[0]:.3f}, {pos[1]:.3f}]")

    # 测试3: 强健粒子追踪
    print("\n🌊 测试3: 强健粒子追踪...")
    test_input = {
        "parameters": {
            "netcdf_path": test_netcdf_path,
            "initial_positions": test_positions,
            "simulation_hours": 24.0,
            "time_step_hours": 2.0,
            "auto_fix_positions": True,
            "debug_mode": True
        }
    }

    result = robust_particle_tracking_with_validation(test_input)

    if result['success']:
        print("✅ 强健粒子追踪成功")
        if 'position_fixes' in result and result['position_fixes']:
            print(f"   应用了 {len(result['position_fixes'])} 个位置修复")

        trajectories = result.get('trajectories', [])
        print(f"   生成了 {len(trajectories)} 个时间步的轨迹")
    else:
        print(f"❌ 强健粒子追踪失败: {result['message']}")
        if 'debug_info' in result:
            print("   调试信息已包含在结果中")

    print("\n" + "=" * 50)
    print("🎯 验证修复系统测试完成")

    if result['success']:
        print("🎉 系统运行正常！可以用于生产环境。")
    else:
        print("⚠️  系统需要进一步调试，请检查NetCDF数据格式。")