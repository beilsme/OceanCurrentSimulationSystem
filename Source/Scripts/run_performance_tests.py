#!/usr/bin/env python3
"""
性能测试和基准测试脚本
"""

import subprocess
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path

def run_cpp_benchmarks():
    """运行C++性能测试"""
    print("运行C++性能基准测试...")
    
    cmd = "./CppCore/build/tests/performance_tests --json"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        print(f"C++性能测试失败: {result.stderr}")
        return None

def generate_performance_report(benchmark_data):
    """生成性能报告"""
    if not benchmark_data:
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 粒子模拟性能
    particle_times = benchmark_data.get('particle_simulation', {})
    if particle_times:
        particle_counts = list(particle_times.keys())
        times = list(particle_times.values())
        
        axes[0, 0].plot(particle_counts, times, 'b-o')
        axes[0, 0].set_title('Particle Simulation Performance')
        axes[0, 0].set_xlabel('Particle Count')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].grid(True)
    
    # 内存使用
    memory_data = benchmark_data.get('memory_usage', {})
    if memory_data:
        operations = list(memory_data.keys())
        memory = list(memory_data.values())
        
        axes[0, 1].bar(operations, memory)
        axes[0, 1].set_title('Memory Usage by Operation')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 并行效率
    parallel_data = benchmark_data.get('parallel_efficiency', {})
    if parallel_data:
        thread_counts = list(parallel_data.keys())
        efficiency = list(parallel_data.values())
        
        axes[1, 0].plot(thread_counts, efficiency, 'g-s')
        axes[1, 0].set_title('Parallel Efficiency')
        axes[1, 0].set_xlabel('Thread Count')
        axes[1, 0].set_ylabel('Efficiency (%)')
        axes[1, 0].grid(True)
    
    # 算法比较
    algorithm_data = benchmark_data.get('algorithm_comparison', {})
    if algorithm_data:
        algorithms = list(algorithm_data.keys())
        performance = list(algorithm_data.values())
        
        axes[1, 1].bar(algorithms, performance)
        axes[1, 1].set_title('Algorithm Performance Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('performance_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("性能报告已生成: performance_report.png")

if __name__ == "__main__":
    benchmark_data = run_cpp_benchmarks()
    generate_performance_report(benchmark_data)
