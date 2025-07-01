#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 oceansim 模块中各个类的接口参数
"""

import oceansim
import inspect


def inspect_class_init(cls_name):
    """检查类的构造函数参数"""
    try:
        cls = getattr(oceansim, cls_name)
        print(f"\n=== {cls_name} ===")

        # 获取构造函数的文档字符串
        if hasattr(cls, '__init__'):
            init_doc = cls.__init__.__doc__
            if init_doc:
                print("构造函数签名:")
                print(init_doc)
            else:
                print("没有文档字符串")

        # 尝试检查类的其他方法
        methods = [method for method in dir(cls) if not method.startswith('_')]
        if methods:
            print(f"可用方法: {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''}")

    except Exception as e:
        print(f"❌ 检查 {cls_name} 失败: {e}")


def check_config_classes():
    """检查配置类"""
    print("🔍 检查配置类:")

    config_classes = ['VectorConfig', 'EngineConfig']
    for cls_name in config_classes:
        inspect_class_init(cls_name)


def check_solver_classes():
    """检查求解器类"""
    print("\n🔍 检查求解器类:")

    solver_classes = [
        'FiniteDifferenceSolver',
        'RungeKuttaSolver',
        'VectorizedOperations',
        'ParallelComputeEngine'
    ]
    for cls_name in solver_classes:
        inspect_class_init(cls_name)


def check_main_classes():
    """检查主要类"""
    print("\n🔍 检查主要类:")

    main_classes = [
        'GridDataStructure',
        'PhysicalParameters',
        'Particle',
        'ParticleSimulator',
        'AdvectionDiffusionSolver',
        'CurrentFieldSolver'
    ]
    for cls_name in main_classes:
        inspect_class_init(cls_name)


def check_enum_values():
    """检查枚举值"""
    print("\n🔍 检查枚举值:")

    enums = {
        'CoordinateSystem': ['CARTESIAN', 'SPHERICAL', 'HYBRID_SIGMA', 'ISOPYCNAL'],
        'GridType': ['REGULAR', 'CURVILINEAR', 'UNSTRUCTURED'],
        'InterpolationMethod': ['LINEAR', 'CUBIC', 'BILINEAR', 'TRILINEAR', 'CONSERVATIVE'],
        'ExecutionPolicy': [],  # 看看有什么值
        'Priority': [],  # 看看有什么值
        'SimdType': []  # 看看有什么值
    }

    for enum_name, known_values in enums.items():
        print(f"\n{enum_name}:")
        if known_values:
            for value in known_values:
                try:
                    val = getattr(oceansim, value)
                    print(f"  {value} = {val}")
                except:
                    print(f"  {value} = 未找到")
        else:
            # 尝试找到枚举的所有值
            try:
                enum_cls = getattr(oceansim, enum_name)
                if hasattr(enum_cls, '__members__'):
                    for name, value in enum_cls.__members__.items():
                        print(f"  {name} = {value}")
            except:
                print(f"  无法检查 {enum_name}")


def try_create_objects():
    """尝试创建各种对象"""
    print("\n🧪 尝试创建对象:")

    # 尝试创建 VectorConfig
    try:
        config = oceansim.VectorConfig()
        print("✅ VectorConfig() 成功")
    except Exception as e:
        print(f"❌ VectorConfig() 失败: {e}")

    # 尝试创建 EngineConfig
    try:
        config = oceansim.EngineConfig()
        print("✅ EngineConfig() 成功")
    except Exception as e:
        print(f"❌ EngineConfig() 失败: {e}")

    # 尝试用不同参数创建 FiniteDifferenceSolver
    try:
        solver = oceansim.FiniteDifferenceSolver(100, 0.01)
        print("✅ FiniteDifferenceSolver(100, 0.01) 成功")
    except Exception as e:
        print(f"❌ FiniteDifferenceSolver(100, 0.01) 失败: {e}")


def main():
    print("🔍 oceansim 模块接口检查")
    print("=" * 60)

    check_config_classes()
    check_solver_classes()
    check_main_classes()
    check_enum_values()
    try_create_objects()

    print("\n" + "=" * 60)
    print("检查完成!")


if __name__ == "__main__":
    main()