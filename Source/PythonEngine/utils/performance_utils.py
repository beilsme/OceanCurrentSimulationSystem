#!/usr/bin/env python3
"""
性能监控工具
提供系统性能监控、函数执行时间测量等功能
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import gc
import tracemalloc
import weakref

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0

@dataclass
class FunctionProfile:
    """函数性能分析数据"""
    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: Optional[datetime] = None
    memory_usage: List[float] = field(default_factory=list)

class PerformanceMonitor:
    """性能监控器类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能监控器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 监控设置
        self.monitoring_interval = self.config.get("monitoring_interval", 1.0)  # 秒
        self.max_history_length = self.config.get("max_history_length", 1000)
        self.enable_memory_profiling = self.config.get("enable_memory_profiling", True)
        self.enable_gpu_monitoring = self.config.get("enable_gpu_monitoring", True)

        # 数据存储
        self.metrics_history: List[PerformanceMetrics] = []
        self.function_profiles: Dict[str, FunctionProfile] = {}

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 内存分析
        self.memory_tracker_active = False

        # GPU监控（如果可用）
        self.gpu_available = False
        try:
            import GPUtil
            self.GPUtil = GPUtil
            self.gpu_available = True
        except ImportError:
            self.logger.info("GPUtil不可用，GPU监控已禁用")

        # 基准性能数据
        self.baseline_metrics: Optional[PerformanceMetrics] = None

        # 弱引用注册表（用于跟踪对象）
        self.object_registry = weakref.WeakSet()

    def start(self):
        """启动性能监控"""
        if self.is_monitoring:
            self.logger.warning("性能监控已在运行")
            return

        self.is_monitoring = True
        self.stop_event.clear()

        # 启动内存追踪
        if self.enable_memory_profiling:
            try:
                tracemalloc.start()
                self.memory_tracker_active = True
                self.logger.info("内存追踪已启动")
            except Exception as e:
                self.logger.warning(f"启动内存追踪失败: {e}")

        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

        # 记录基准性能
        self.baseline_metrics = self._collect_current_metrics()

        self.logger.info("性能监控已启动")

    def stop(self):
        """停止性能监控"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # 停止内存追踪
        if self.memory_tracker_active:
            try:
                tracemalloc.stop()
                self.memory_tracker_active = False
                self.logger.info("内存追踪已停止")
            except Exception as e:
                self.logger.warning(f"停止内存追踪失败: {e}")

        self.logger.info("性能监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_event.wait(self.monitoring_interval):
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)

                # 限制历史长度
                if len(self.metrics_history) > self.max_history_length:
                    self.metrics_history.pop(0)

                # 检查性能告警
                self._check_performance_alerts(metrics)

            except Exception as e:
                self.logger.error(f"性能监控循环错误: {e}")

    def _collect_current_metrics(self) -> PerformanceMetrics:
        """收集当前性能指标"""
        metrics = PerformanceMetrics()

        try:
            # CPU使用率
            metrics.cpu_percent = psutil.cpu_percent()

            # 内存信息
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_used_mb = memory.used / (1024 * 1024)
            metrics.memory_available_mb = memory.available / (1024 * 1024)

            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics.disk_io_read_mb = disk_io.read_bytes / (1024 * 1024)
                metrics.disk_io_write_mb = disk_io.write_bytes / (1024 * 1024)

            # 网络IO
            network_io = psutil.net_io_counters()
            if network_io:
                metrics.network_sent_mb = network_io.bytes_sent / (1024 * 1024)
                metrics.network_recv_mb = network_io.bytes_recv / (1024 * 1024)

            # 活跃线程数
            metrics.active_threads = threading.active_count()

            # GPU信息（如果可用）
            if self.gpu_available and self.enable_gpu_monitoring:
                try:
                    gpus = self.GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # 使用第一个GPU
                        metrics.gpu_utilization = gpu.load * 100
                        metrics.gpu_memory_used_mb = gpu.memoryUsed
                except Exception as e:
                    self.logger.debug(f"GPU信息收集失败: {e}")

        except Exception as e:
            self.logger.error(f"收集性能指标失败: {e}")

        return metrics

    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """检查性能告警"""
        alerts = []

        # CPU使用率告警
        cpu_threshold = self.config.get("cpu_alert_threshold", 90.0)
        if metrics.cpu_percent > cpu_threshold:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")

        # 内存使用率告警
        memory_threshold = self.config.get("memory_alert_threshold", 85.0)
        if metrics.memory_percent > memory_threshold:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")

        # GPU使用率告警
        if metrics.gpu_utilization > 0:
            gpu_threshold = self.config.get("gpu_alert_threshold", 95.0)
            if metrics.gpu_utilization > gpu_threshold:
                alerts.append(f"GPU使用率过高: {metrics.gpu_utilization:.1f}%")

        # 发送告警
        for alert in alerts:
            self.logger.warning(f"性能告警: {alert}")

    def profile_function(self, func_name: Optional[str] = None):
        """函数性能分析装饰器"""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._profile_sync_function(func, name, *args, **kwargs)

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_function(func, name, *args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper

        return decorator

    def _profile_sync_function(self, func: Callable, name: str, *args, **kwargs):
        """同步函数性能分析"""
        start_time = time.time()
        start_memory = self._get_current_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = self._get_current_memory_usage()
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0

            self._update_function_profile(name, execution_time, memory_delta)

    async def _profile_async_function(self, func: Callable, name: str, *args, **kwargs):
        """异步函数性能分析"""
        start_time = time.time()
        start_memory = self._get_current_memory_usage()

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = self._get_current_memory_usage()
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory if start_memory and end_memory else 0

            self._update_function_profile(name, execution_time, memory_delta)

    def _get_current_memory_usage(self) -> Optional[float]:
        """获取当前内存使用量"""
        if not self.memory_tracker_active:
            return None

        try:
            current, peak = tracemalloc.get_traced_memory()
            return current / (1024 * 1024)  # 转换为MB
        except Exception:
            return None

    def _update_function_profile(self, name: str, execution_time: float, memory_delta: float):
        """更新函数性能分析数据"""
        if name not in self.function_profiles:
            self.function_profiles[name] = FunctionProfile(name=name)

        profile = self.function_profiles[name]
        profile.total_calls += 1
        profile.total_time += execution_time
        profile.min_time = min(profile.min_time, execution_time)
        profile.max_time = max(profile.max_time, execution_time)
        profile.avg_time = profile.total_time / profile.total_calls
        profile.last_call_time = datetime.now()

        if memory_delta != 0:
            profile.memory_usage.append(memory_delta)
            # 限制内存使用记录长度
            if len(profile.memory_usage) > 100:
                profile.memory_usage.pop(0)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.metrics_history:
            return {"message": "没有可用的性能数据"}

        latest = self.metrics_history[-1]

        # 计算统计信息
        recent_metrics = self.metrics_history[-min(60, len(self.metrics_history)):]  # 最近60个数据点

        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)

        return {
            "current": {
                "timestamp": latest.timestamp.isoformat(),
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": latest.memory_used_mb,
                "memory_available_mb": latest.memory_available_mb,
                "active_threads": latest.active_threads,
                "gpu_utilization": latest.gpu_utilization,
                "gpu_memory_used_mb": latest.gpu_memory_used_mb
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "baseline_comparison": self._get_baseline_comparison(latest),
            "monitoring_status": {
                "is_active": self.is_monitoring,
                "history_length": len(self.metrics_history),
                "memory_tracking": self.memory_tracker_active,
                "gpu_available": self.gpu_available
            }
        }

    def get_function_profiles(self) -> Dict[str, Dict[str, Any]]:
        """获取函数性能分析报告"""
        profiles = {}

        for name, profile in self.function_profiles.items():
            avg_memory = (
                sum(profile.memory_usage) / len(profile.memory_usage)
                if profile.memory_usage else 0.0
            )

            profiles[name] = {
                "total_calls": profile.total_calls,
                "total_time": profile.total_time,
                "avg_time": profile.avg_time,
                "min_time": profile.min_time if profile.min_time != float('inf') else 0.0,
                "max_time": profile.max_time,
                "last_call": profile.last_call_time.isoformat() if profile.last_call_time else None,
                "avg_memory_delta_mb": avg_memory,
                "calls_per_second": self._calculate_calls_per_second(profile)
            }

        return profiles

    def _calculate_calls_per_second(self, profile: FunctionProfile) -> float:
        """计算函数调用频率"""
        if not profile.last_call_time or profile.total_calls < 2:
            return 0.0

        # 估算基于最近一小时的调用频率
        time_window = timedelta(hours=1)
        cutoff_time = datetime.now() - time_window

        if profile.last_call_time < cutoff_time:
            return 0.0

        # 简化计算：假设调用均匀分布
        elapsed_hours = (datetime.now() - (profile.last_call_time - timedelta(seconds=profile.total_time))).total_seconds() / 3600
        if elapsed_hours > 0:
            return profile.total_calls / elapsed_hours / 3600  # 转换为每秒

        return 0.0

    def _get_baseline_comparison(self, current: PerformanceMetrics) -> Dict[str, float]:
        """与基准性能比较"""
        if not self.baseline_metrics:
            return {}

        baseline = self.baseline_metrics

        return {
            "cpu_change": current.cpu_percent - baseline.cpu_percent,
            "memory_change": current.memory_percent - baseline.memory_percent,
            "threads_change": current.active_threads - baseline.active_threads
        }

    def get_memory_top_consumers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取内存消耗最大的函数"""
        if not self.memory_tracker_active:
            return []

        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')

            consumers = []
            for stat in top_stats[:limit]:
                consumers.append({
                    "file": stat.traceback.format()[0],
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count
                })

            return consumers
        except Exception as e:
            self.logger.error(f"获取内存消耗信息失败: {e}")
            return []

    def register_object(self, obj: Any, name: str = None):
        """注册对象用于内存跟踪"""
        self.object_registry.add(obj)
        if name:
            # 可以添加名称映射逻辑
            pass

    def get_object_count(self) -> int:
        """获取注册对象数量"""
        return len(self.object_registry)

    def force_garbage_collection(self) -> Dict[str, int]:
        """强制垃圾回收并返回统计信息"""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())

        stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_collected": collected,
            "objects_freed": before_objects - after_objects
        }

        self.logger.info(f"垃圾回收完成: {stats}")
        return stats

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu": {
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3),
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            },
            "gpu": self._get_gpu_info() if self.gpu_available else None
        }

    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """获取GPU信息"""
        try:
            gpus = self.GPUtil.getGPUs()
            if not gpus:
                return None

            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature
                })

            return {"gpus": gpu_info, "count": len(gpus)}
        except Exception as e:
            self.logger.error(f"获取GPU信息失败: {e}")
            return None

    def export_metrics_to_csv(self, filepath: str, include_functions: bool = True):
        """导出性能指标到CSV文件"""
        import csv

        with open(filepath, 'w', newline='') as csvfile:
            # 写入系统指标
            fieldnames = [
                'timestamp', 'cpu_percent', 'memory_percent', 'memory_used_mb',
                'active_threads', 'gpu_utilization', 'gpu_memory_used_mb'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for metrics in self.metrics_history:
                writer.writerow({
                    'timestamp': metrics.timestamp.isoformat(),
                    'cpu_percent': metrics.cpu_percent,
                    'memory_percent': metrics.memory_percent,
                    'memory_used_mb': metrics.memory_used_mb,
                    'active_threads': metrics.active_threads,
                    'gpu_utilization': metrics.gpu_utilization,
                    'gpu_memory_used_mb': metrics.gpu_memory_used_mb
                })

        # 写入函数性能数据
        if include_functions and self.function_profiles:
            func_filepath = filepath.replace('.csv', '_functions.csv')
            with open(func_filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'function_name', 'total_calls', 'total_time', 'avg_time',
                    'min_time', 'max_time', 'avg_memory_delta_mb'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for name, profile in self.function_profiles.items():
                    avg_memory = (
                        sum(profile.memory_usage) / len(profile.memory_usage)
                        if profile.memory_usage else 0.0
                    )

                    writer.writerow({
                        'function_name': name,
                        'total_calls': profile.total_calls,
                        'total_time': profile.total_time,
                        'avg_time': profile.avg_time,
                        'min_time': profile.min_time if profile.min_time != float('inf') else 0.0,
                        'max_time': profile.max_time,
                        'avg_memory_delta_mb': avg_memory
                    })

        self.logger.info(f"性能指标已导出到: {filepath}")

    def reset_statistics(self):
        """重置所有统计数据"""
        self.metrics_history.clear()
        self.function_profiles.clear()
        self.baseline_metrics = None
        self.logger.info("性能统计数据已重置")

    def cleanup(self):
        """清理资源"""
        self.stop()
        self.reset_statistics()
        self.logger.info("性能监控器已清理")

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()

# 装饰器函数
def profile_performance(func_name: Optional[str] = None):
    """性能分析装饰器"""
    return performance_monitor.profile_function(func_name)

def async_timer(func: Optional[Callable] = None):
    """异步函数计时装饰器"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                func_name = f"{f.__module__}.{f.__name__}"
                performance_monitor._update_function_profile(func_name, elapsed, 0)
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)

def timer(func: Optional[Callable] = None):
    """同步函数计时装饰器"""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                func_name = f"{f.__module__}.{f.__name__}"
                performance_monitor._update_function_profile(func_name, elapsed, 0)
        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)

class MemoryProfiler:
    """内存分析器上下文管理器"""

    def __init__(self, name: str = "memory_profile"):
        self.name = name
        self.start_memory = 0
        self.peak_memory = 0

    def __enter__(self):
        if tracemalloc.is_tracing():
            self.start_memory, _ = tracemalloc.get_traced_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if tracemalloc.is_tracing():
            current_memory, self.peak_memory = tracemalloc.get_traced_memory()
            memory_used = (current_memory - self.start_memory) / (1024 * 1024)
            peak_used = self.peak_memory / (1024 * 1024)

            logging.getLogger(__name__).info(
                f"内存分析 [{self.name}]: 使用 {memory_used:.2f}MB, 峰值 {peak_used:.2f}MB"
            )

class CPUProfiler:
    """CPU使用率分析器上下文管理器"""

    def __init__(self, name: str = "cpu_profile", interval: float = 0.1):
        self.name = name
        self.interval = interval
        self.start_time = 0
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None

    def __enter__(self):
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if self.cpu_samples:
            avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
            max_cpu = max(self.cpu_samples)
            elapsed = time.time() - self.start_time

            logging.getLogger(__name__).info(
                f"CPU分析 [{self.name}]: 平均 {avg_cpu:.1f}%, 峰值 {max_cpu:.1f}%, 耗时 {elapsed:.2f}s"
            )

    def _monitor_cpu(self):
        """监控CPU使用率"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                time.sleep(self.interval)
            except Exception:
                break