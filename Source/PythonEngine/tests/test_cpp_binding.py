#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 C++ 绑定模块的功能
"""

import oceansim
import numpy as np


def test_basic_import():
    """测试基本导入"""
    print("=== 基本导入测试 ===")
    print(f"✅ oceansim 模块导入成功")
    print(f"模块位置: {oceansim.__file__}")
    print(f"可用功能数量: {len([x for x in dir(oceansim) if not x.startswith('_')])}")


def test_hello_function():
    """测试简单函数"""
    print("\n=== 简单函数测试 ===")
    result = oceansim.hello()
    print(f"hello() 返回: {result}")
    assert isinstance(result, str)

    result = oceansim.test_add(3.14, 2.86)
    print(f"test_add(3.14, 2.86) = {result}")
    assert abs(result - 6.0) < 1e-6


def test_enums():
    """测试枚举类型"""
    print("\n=== 枚举类型测试 ===")

    # 坐标系统
    print("坐标系统:")
    print(f"  CARTESIAN = {oceansim.CARTESIAN}")
    print(f"  SPHERICAL = {oceansim.SPHERICAL}")
    print(f"  HYBRID_SIGMA = {oceansim.HYBRID_SIGMA}")

    # 网格类型
    print("网格类型:")
    print(f"  REGULAR = {oceansim.REGULAR}")
    print(f"  CURVILINEAR = {oceansim.CURVILINEAR}")

    # 插值方法
    print("插值方法:")
    print(f"  LINEAR = {oceansim.LINEAR}")
    print(f"  BILINEAR = {oceansim.BILINEAR}")
    print(f"  TRILINEAR = {oceansim.TRILINEAR}")


def test_physical_parameters():
    """测试物理参数类"""
    print("\n=== 物理参数测试 ===")
    try:
        params = oceansim.PhysicalParameters()
        print("✅ PhysicalParameters 对象创建成功")
        print(f"对象类型: {type(params)}")
    except Exception as e:
        print(f"❌ PhysicalParameters 创建失败: {e}")


def test_particle():
    """测试粒子类"""
    print("\n=== 粒子测试 ===")
    try:
        # 测试默认构造
        particle1 = oceansim.Particle()
        print("✅ Particle() 默认构造成功")

        # 测试带位置构造（如果支持 Eigen 向量）
        try:
            # 尝试用 numpy 数组
            position = np.array([1.0, 2.0, 3.0])
            particle2 = oceansim.Particle(position)
            print("✅ Particle(position) 构造成功")
        except Exception as e:
            print(f"⚠️  带参数构造失败: {e}")

        print(f"粒子对象类型: {type(particle1)}")

    except Exception as e:
        print(f"❌ Particle 创建失败: {e}")


def test_grid_data_structure():
    """测试网格数据结构"""
    print("\n=== 网格数据结构测试 ===")
    try:
        # 创建简单的 3D 网格
        nx, ny, nz = 10, 10, 5
        grid = oceansim.GridDataStructure(nx, ny, nz)
        print(f"✅ GridDataStructure({nx}, {ny}, {nz}) 创建成功")

        # 测试带参数的构造
        grid2 = oceansim.GridDataStructure(
            nx, ny, nz,
            oceansim.CARTESIAN,
            oceansim.REGULAR
        )
        print("✅ 带参数的网格创建成功")

        # 测试方法调用
        dims = grid.get_dimensions()
        print(f"网格维度: {dims}")

        print(f"网格对象类型: {type(grid)}")

    except Exception as e:
        print(f"❌ GridDataStructure 创建失败: {e}")


def test_solvers():
    """测试求解器"""
    print("\n=== 求解器测试 ===")

    # 测试有限差分求解器（需要参数）
    try:
        fd_solver = oceansim.FiniteDifferenceSolver(grid_size=100, time_step=0.01)
        print("✅ FiniteDifferenceSolver 创建成功")
    except Exception as e:
        print(f"❌ FiniteDifferenceSolver 创建失败: {e}")

    # 测试龙格库塔求解器
    try:
        rk_solver = oceansim.RungeKuttaSolver()
        print("✅ RungeKuttaSolver 创建成功")
    except Exception as e:
        print(f"❌ RungeKuttaSolver 创建失败: {e}")


def test_vectorized_operations():
    """测试向量化运算"""
    print("\n=== 向量化运算测试 ===")
    try:
        # 需要 VectorConfig
        config = oceansim.VectorConfig()  # 尝试默认构造
        vec_ops = oceansim.VectorizedOperations(config)
        print("✅ VectorizedOperations 创建成功")
    except Exception as e:
        print(f"❌ VectorizedOperations 创建失败: {e}")


def test_parallel_compute_engine():
    """测试并行计算引擎"""
    print("\n=== 并行计算引擎测试 ===")
    try:
        # 需要 EngineConfig
        config = oceansim.EngineConfig()  # 尝试默认构造
        parallel_engine = oceansim.ParallelComputeEngine(config)
        print("✅ ParallelComputeEngine 创建成功")
    except Exception as e:
        print(f"❌ ParallelComputeEngine 创建失败: {e}")


def test_ocean_simulation():
    """测试完整的海洋模拟流程"""
    print("\n=== 海洋模拟流程测试 ===")
    try:
        # 1. 创建网格
        grid = oceansim.GridDataStructure(20, 20, 10)
        print("✅ 网格创建成功")

        # 2. 创建物理参数
        params = oceansim.PhysicalParameters()
        print("✅ 物理参数创建成功")

        # 3. 创建洋流场求解器
        try:
            current_solver = oceansim.CurrentFieldSolver(grid, params)
            print("✅ 洋流场求解器创建成功")
        except Exception as e:
            print(f"⚠️  洋流场求解器创建失败: {e}")

        # 4. 创建粒子模拟器
        try:
            rk_solver = oceansim.RungeKuttaSolver()
            particle_sim = oceansim.ParticleSimulator(grid, rk_solver)
            print("✅ 粒子模拟器创建成功")
        except Exception as e:
            print(f"⚠️  粒子模拟器创建失败: {e}")

        print("🎉 基本模拟流程测试完成")

    except Exception as e:
        print(f"❌ 海洋模拟流程测试失败: {e}")


def run_all_tests():
    """运行所有测试"""
    print("🌊 海洋洋流模拟系统 C++ 绑定测试")
    print("=" * 50)

    test_basic_import()
    test_hello_function()
    test_enums()
    test_physical_parameters()
    test_particle()
    test_grid_data_structure()
    test_solvers()
    test_vectorized_operations()
    test_parallel_compute_engine()
    test_ocean_simulation()

    print("\n" + "=" * 50)
    print("🎯 测试完成！")


if __name__ == "__main__":
    run_all_tests()