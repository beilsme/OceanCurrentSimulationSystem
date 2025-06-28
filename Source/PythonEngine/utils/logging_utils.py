#!/usr/bin/env python3
"""
日志工具
提供统一的日志配置和管理功能
"""

import logging
import logging.handlers
import sys
import os
import json
from typing import Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import traceback
import threading
from enum import Enum

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器"""

    def __init__(self, include_thread=True, include_process=True, colored=True):
        """
        初始化格式化器
        
        Args:
            include_thread: 是否包含线程信息
            include_process: 是否包含进程信息
            colored: 是否使用彩色输出
        """
        self.include_thread = include_thread
        self.include_process = include_process
        self.colored = colored and sys.stdout.isatty()

        # 颜色代码
        self.colors = {
            'DEBUG': '\033[36m',     # 青色
            'INFO': '\033[32m',      # 绿色
            'WARNING': '\033[33m',   # 黄色
            'ERROR': '\033[31m',     # 红色
            'CRITICAL': '\033[35m',  # 紫色
            'RESET': '\033[0m'       # 重置
        }

        # 构建格式字符串
        format_parts = ['%(asctime)s']

        if self.include_process:
            format_parts.append('[PID:%(process)d]')

        if self.include_thread:
            format_parts.append('[%(threadName)s]')

        format_parts.extend([
            '[%(name)s]',
            '[%(levelname)s]',
            '%(message)s'
        ])

        self.base_format = ' '.join(format_parts)

        super().__init__(
            fmt=self.base_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def format(self, record):
        """格式化日志记录"""
        # 基础格式化
        formatted = super().format(record)

        # 添加颜色
        if self.colored:
            level_color = self.colors.get(record.levelname, '')
            reset_color = self.colors['RESET']

            # 只给级别名称和消息添加颜色
            formatted = formatted.replace(
                f"[{record.levelname}]",
                f"{level_color}[{record.levelname}]{reset_color}"
            )

        return formatted

class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""

    def __init__(self, include_extra=True):
        """
        初始化JSON格式化器
        
        Args:
            include_extra: 是否包含额外字段
        """
        self.include_extra = include_extra
        super().__init__()

    def format(self, record):
        """格式化为JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.threadName,
            'process': record.process
        }

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in log_data and not key.startswith('_'):
                    try:
                        # 确保值可以JSON序列化
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)

class PerformanceLogHandler(logging.Handler):
    """性能日志处理器"""

    def __init__(self, performance_monitor=None):
        """
        初始化性能日志处理器
        
        Args:
            performance_monitor: 性能监控器实例
        """
        super().__init__()
        self.performance_monitor = performance_monitor
        self.log_counts = {
            'DEBUG': 0,
            'INFO': 0,
            'WARNING': 0,
            'ERROR': 0,
            'CRITICAL': 0
        }
        self.lock = threading.Lock()

    def emit(self, record):
        """处理日志记录"""
        with self.lock:
            level = record.levelname
            if level in self.log_counts:
                self.log_counts[level] += 1

            # 如果有性能监控器，记录日志事件
            if self.performance_monitor:
                try:
                    # 可以在这里添加与性能监控器的集成
                    pass
                except Exception:
                    pass  # 避免日志处理器本身出错

    def get_statistics(self) -> Dict[str, int]:
        """获取日志统计信息"""
        with self.lock:
            return self.log_counts.copy()

def setup_logging(
        name: str = "OceanSimulation",
        level: Union[str, int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        json_format: bool = False,
        include_thread_info: bool = True,
        include_process_info: bool = True,
        colored_output: bool = True,
        performance_monitor=None
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件名
        log_dir: 日志目录
        max_file_size: 单个日志文件最大大小
        backup_count: 备份文件数量
        console_output: 是否输出到控制台
        json_format: 是否使用JSON格式
        include_thread_info: 是否包含线程信息
        include_process_info: 是否包含进程信息
        colored_output: 是否使用彩色输出
        performance_monitor: 性能监控器实例
        
    Returns:
        配置好的日志器
    """
    # 转换日志级别
    if isinstance(level, LogLevel):
        level = level.value
    elif isinstance(level, str):
        level = getattr(logging, level.upper())

    # 获取或创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 选择格式化器
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = CustomFormatter(
            include_thread=include_thread_info,
            include_process=include_process_info,
            colored=colored_output
        )

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file or log_dir:
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            if not log_file:
                log_file = f"{name.lower().replace(' ', '_')}.log"
            log_file_path = log_dir_path / log_file
        else:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用旋转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_file_path),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        # 文件使用非彩色格式
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = CustomFormatter(
                include_thread=include_thread_info,
                include_process=include_process_info,
                colored=False  # 文件不使用颜色
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 性能日志处理器
    if performance_monitor:
        perf_handler = PerformanceLogHandler(performance_monitor)
        perf_handler.setLevel(logging.WARNING)  # 只记录警告及以上级别
        logger.addHandler(perf_handler)

    # 防止重复日志
    logger.propagate = False

    return logger

def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)

def log_exception(logger: logging.Logger, message: str = "发生异常"):
    """记录异常信息"""
    logger.error(f"{message}: {traceback.format_exc()}")

def log_function_call(logger: logging.Logger, func_name: str, args: tuple = (), kwargs: dict = None):
    """记录函数调用"""
    kwargs = kwargs or {}
    args_str = ', '.join(str(arg) for arg in args)
    kwargs_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())

    params = ', '.join(filter(None, [args_str, kwargs_str]))
    logger.debug(f"调用函数 {func_name}({params})")

def create_child_logger(parent_logger: logging.Logger, child_name: str) -> logging.Logger:
    """创建子日志器"""
    child_logger_name = f"{parent_logger.name}.{child_name}"
    child_logger = logging.getLogger(child_logger_name)
    child_logger.setLevel(parent_logger.level)
    return child_logger

class LogContext:
    """日志上下文管理器"""

    def __init__(self, logger: logging.Logger, context_info: Dict[str, Any]):
        """
        初始化日志上下文
        
        Args:
            logger: 日志器
            context_info: 上下文信息
        """
        self.logger = logger
        self.context_info = context_info
        self.original_factory = None

    def __enter__(self):
        """进入上下文"""
        self.original_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context_info.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        logging.setLogRecordFactory(self.original_factory)

def with_log_context(logger: logging.Logger, **context):
    """日志上下文装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LogContext(logger, context):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """自定义时间旋转文件处理器"""

    def __init__(self, filename, when='midnight', interval=1, backupCount=7, **kwargs):
        """
        初始化时间旋转文件处理器
        
        Args:
            filename: 文件名
            when: 旋转时机
            interval: 间隔
            backupCount: 备份数量
        """
        super().__init__(filename, when, interval, backupCount, **kwargs)

    def doRollover(self):
        """执行日志旋转"""
        super().doRollover()
        # 可以在这里添加额外的清理逻辑

def configure_system_logging(
        config_file: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
):
    """
    配置系统级日志
    
    Args:
        config_file: 配置文件路径
        config_dict: 配置字典
    """
    if config_file and os.path.exists(config_file):
        import logging.config
        if config_file.endswith('.json'):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        elif config_file.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    elif config_dict:
        import logging.config
        logging.config.dictConfig(config_dict)
    else:
        # 默认配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

# 默认日志配置
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [PID:%(process)d] [%(threadName)s] [%(name)s] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(asctime)s [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': 'utils.logging_utils.JSONFormatter'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/ocean_simulation.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/ocean_simulation_errors.log',
            'maxBytes': 10485760,
            'backupCount': 3,
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        'OceanSimulation': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}