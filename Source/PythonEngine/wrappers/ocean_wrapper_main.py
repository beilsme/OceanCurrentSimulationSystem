#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: ocean_wrapper_main.py
位置: Source/PythonEngine/wrappers/ocean_wrapper_main.py
功能: 海洋模拟系统统一包装器入口，负责C#调用的请求路由和分发
用法: python ocean_wrapper_main.py input.json output.json
"""

import sys
import json
import math
import traceback
from pathlib import Path

# 添加Python引擎路径到sys.path
current_dir = Path(__file__).parent
python_engine_root = current_dir.parent.parent  # Source directory

sys.path.insert(0, str(python_engine_root))

def nan_to_none(obj):
    """将NaN值转换为None，确保JSON序列化的兼容性"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(x) for x in obj]
    return obj

def route_request(input_data):
    """
    请求路由函数，根据action字段将请求分发到相应的处理模块
    
    Args:
        input_data: 输入数据字典，包含action和parameters字段
        
    Returns:
        dict: 处理结果字典
    """
    action = input_data.get('action', '')

    try:
        # 基础数据处理和可视化功能
        if action in ['load_netcdf', 'plot_vector_field', 'export_vector_shapefile',
                      'get_statistics', 'create_ocean_animation', 'calculate_vorticity_divergence',
                      'calculate_flow_statistics', 'get_time_range']:

            from ocean_data_wrapper import (
                load_netcdf_data, plot_vector_field, export_vector_shapefile,
                get_statistics, create_ocean_animation, calculate_vorticity_divergence,
                calculate_flow_statistics
            )
            from PythonEngine.core.netcdf_handler import NetCDFHandler

            if action == 'load_netcdf':
                return load_netcdf_data(input_data)
            elif action == 'plot_vector_field':
                return plot_vector_field(input_data)
            elif action == 'export_vector_shapefile':
                return export_vector_shapefile(input_data)
            elif action == 'get_statistics':
                return get_statistics(input_data)
            elif action == 'create_ocean_animation':
                return create_ocean_animation(input_data)
            elif action == 'calculate_vorticity_divergence':
                return calculate_vorticity_divergence(input_data)
            elif action == 'calculate_flow_statistics':
                return calculate_flow_statistics(input_data)
            elif action == 'get_time_range':
                return handle_get_time_range(input_data)

        # 拉格朗日粒子追踪相关功能
        elif action in ['simulate_particle_tracking', 'validate_particle_setup']:

            from lagrangian_particle_wrapper import (
                simulate_particle_tracking, validate_particle_setup
            )

            if action == 'simulate_particle_tracking':
                return simulate_particle_tracking(input_data)
            elif action == 'validate_particle_setup':
                return validate_particle_setup(input_data)

        # 未来扩展功能预留位置
        # elif action in ['advanced_analysis', 'machine_learning_forecast']:
        #     from advanced_analysis_wrapper import handle_advanced_analysis
        #     return handle_advanced_analysis(input_data)

        else:
            return {
                "success": False,
                "message": f"未知的请求类型: {action}",
                "supported_actions": [
                    "load_netcdf", "plot_vector_field", "export_vector_shapefile",
                    "get_statistics", "create_ocean_animation", "calculate_vorticity_divergence",
                    "calculate_flow_statistics", "get_time_range", "simulate_particle_tracking",
                    "validate_particle_setup"
                ]
            }

    except ImportError as e:
        return {
            "success": False,
            "message": f"模块导入失败: {str(e)}",
            "error_type": "import_error",
            "error_trace": traceback.format_exc()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"请求处理失败: {str(e)}",
            "error_type": "processing_error",
            "error_trace": traceback.format_exc()
        }

def handle_get_time_range(input_data):
    """
    处理获取NetCDF时间范围信息的请求
    
    Args:
        input_data: 包含netcdf_path的输入数据
        
    Returns:
        dict: 时间范围信息结果
    """
    try:
        from PythonEngine.core.netcdf_handler import NetCDFHandler

        netcdf_path = input_data['parameters']['netcdf_path']
        handler = NetCDFHandler(netcdf_path)

        try:
            ds = handler.ds
            total_time_steps = ds.sizes.get('time', 1)

            # 尝试获取时间单位信息
            time_units = "unknown"
            if 'time' in ds.variables:
                time_var = ds.variables['time']
                time_units = getattr(time_var, 'units', 'unknown')

            return {
                "success": True,
                "time_info": {
                    "total_time_steps": total_time_steps,
                    "max_simulation_days": total_time_steps,
                    "time_step_hours": 24,  # 默认假设每步为一天
                    "time_units": time_units
                },
                "message": f"时间范围信息获取成功，共{total_time_steps}个时间步"
            }
        finally:
            handler.close()

    except Exception as e:
        return {
            "success": False,
            "message": f"时间范围信息获取失败: {str(e)}",
            "error_trace": traceback.format_exc()
        }

def validate_input_data(input_data):
    """
    验证输入数据的基本格式和必要字段
    
    Args:
        input_data: 输入数据字典
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(input_data, dict):
        return False, "输入数据必须是字典格式"

    if 'action' not in input_data:
        return False, "缺少必要的'action'字段"

    if 'parameters' not in input_data:
        return False, "缺少必要的'parameters'字段"

    if not isinstance(input_data['parameters'], dict):
        return False, "'parameters'字段必须是字典格式"

    return True, None

def main():
    """
    主函数，处理命令行参数和请求分发
    """
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python ocean_wrapper_main.py input.json output.json")
        print("说明: 海洋模拟系统统一包装器入口")
        print("支持的功能:")
        print("  - 基础数据处理: load_netcdf, plot_vector_field, export_vector_shapefile")
        print("  - 统计分析: get_statistics, calculate_vorticity_divergence, calculate_flow_statistics")
        print("  - 动画生成: create_ocean_animation")
        print("  - 粒子追踪: simulate_particle_tracking, validate_particle_setup")
        print("  - 时间信息: get_time_range")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # 验证输入数据格式
        is_valid, error_message = validate_input_data(input_data)
        if not is_valid:
            result = {
                "success": False,
                "message": f"输入数据格式错误: {error_message}",
                "error_type": "validation_error"
            }
        else:
            # 记录处理开始
            action = input_data.get('action', '未知')
            print(f"[INFO] 开始处理请求: {action}")

            # 路由并处理请求
            result = route_request(input_data)

            # 记录处理结果
            if result.get('success', False):
                print(f"[INFO] 请求处理成功: {result.get('message', '操作完成')}")
            else:
                print(f"[ERROR] 请求处理失败: {result.get('message', '未知错误')}")

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(nan_to_none(result), f, ensure_ascii=False, indent=2)

        # 设置退出码
        exit_code = 0 if result.get('success', False) else 1
        sys.exit(exit_code)

    except FileNotFoundError as e:
        error_result = {
            "success": False,
            "message": f"文件未找到: {str(e)}",
            "error_type": "file_not_found",
            "error_trace": traceback.format_exc()
        }
        print(f"[ERROR] 文件错误: {str(e)}")

    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "message": f"JSON格式错误: {str(e)}",
            "error_type": "json_decode_error",
            "error_trace": traceback.format_exc()
        }
        print(f"[ERROR] JSON解析错误: {str(e)}")

    except Exception as e:
        error_result = {
            "success": False,
            "message": f"包装器执行失败: {str(e)}",
            "error_type": "execution_error",
            "error_trace": traceback.format_exc()
        }
        print(f"[ERROR] 执行错误: {str(e)}")

    # 尝试写入错误结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 静默处理文件写入失败的情况

    sys.exit(1)

if __name__ == "__main__":
    main()