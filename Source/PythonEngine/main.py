#!/usr/bin/env python3
"""
洋流模拟系统Python引擎主入口程序
提供REST API服务，协调各个模块的工作
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.data_processor import DataProcessor
from core.netcdf_handler import NetCDFHandler
from core.interpolation_engine import InterpolationEngine
from core.cpp_interface import CppInterface
from simulation.particle_tracking_wrapper import ParticleTrackingWrapper
from simulation.current_simulation_wrapper import CurrentSimulationWrapper
from machine_learning.deep_learning.lstm_current_predictor import LSTMCurrentPredictor
from machine_learning.deep_learning.pinn_ocean_models import PINNOceanModel
from api.endpoints import setup_routes
from utils.logging_utils import setup_logging
from utils.performance_utils import PerformanceMonitor

class OceanSimulationEngine:
    """洋流模拟引擎主类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # 核心组件
        self.data_processor = None
        self.netcdf_handler = None
        self.interpolation_engine = None
        self.cpp_interface = None

        # 仿真包装器
        self.particle_tracker = None
        self.current_simulator = None

        # 机器学习模型
        self.lstm_predictor = None
        self.pinn_model = None

        # 性能监控
        self.performance_monitor = PerformanceMonitor()

        # 应用状态
        self.is_initialized = False
        self.active_simulations = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 未找到，使用默认配置")
            return self._default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "log_level": "info"
            },
            "models": {
                "lstm_model_path": "./Data/Models/LSTM",
                "pinn_model_path": "./Data/Models/PINN",
                "cache_models": True
            },
            "data": {
                "netcdf_path": "./Data/NetCDF",
                "chunk_size": 1000,
                "compression": "gzip"
            },
            "cpp_interface": {
                "library_path": "./Build/Release/Cpp/libOceanSimCore.so",
                "enable_cuda": False
            },
            "performance": {
                "enable_monitoring": True,
                "log_metrics": True,
                "max_memory_gb": 8
            }
        }

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_level = self.config.get("server", {}).get("log_level", "info")
        return setup_logging("OceanSimEngine", log_level.upper())

    async def initialize_components(self):
        """初始化所有组件"""
        try:
            self.logger.info("初始化洋流模拟引擎组件...")

            # 初始化核心数据处理组件
            self.netcdf_handler = NetCDFHandler(self.config["data"])
            self.data_processor = DataProcessor(
                netcdf_handler=self.netcdf_handler,
                config=self.config["data"]
            )
            self.interpolation_engine = InterpolationEngine()

            # 初始化C++接口
            try:
                self.cpp_interface = CppInterface(self.config["cpp_interface"])
                await self.cpp_interface.initialize()
                self.logger.info("C++计算引擎初始化成功")
            except Exception as e:
                self.logger.warning(f"C++接口初始化失败: {e}")
                self.cpp_interface = None

            # 初始化仿真包装器
            self.particle_tracker = ParticleTrackingWrapper(
                cpp_interface=self.cpp_interface,
                data_processor=self.data_processor
            )

            self.current_simulator = CurrentSimulationWrapper(
                cpp_interface=self.cpp_interface,
                data_processor=self.data_processor
            )

            # 初始化机器学习模型
            if self.config["models"]["cache_models"]:
                try:
                    self.lstm_predictor = LSTMCurrentPredictor(
                        model_path=self.config["models"]["lstm_model_path"]
                    )
                    await self.lstm_predictor.load_model()

                    self.pinn_model = PINNOceanModel(
                        model_path=self.config["models"]["pinn_model_path"]
                    )
                    await self.pinn_model.load_model()

                    self.logger.info("机器学习模型加载成功")
                except Exception as e:
                    self.logger.warning(f"机器学习模型加载失败: {e}")

            # 启动性能监控
            if self.config["performance"]["enable_monitoring"]:
                self.performance_monitor.start()
                self.logger.info("性能监控已启动")

            self.is_initialized = True
            self.logger.info("洋流模拟引擎初始化完成")

        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise

    async def shutdown_components(self):
        """关闭所有组件"""
        self.logger.info("关闭洋流模拟引擎组件...")

        # 停止性能监控
        if self.performance_monitor:
            self.performance_monitor.stop()

        # 关闭C++接口
        if self.cpp_interface:
            await self.cpp_interface.cleanup()

        # 清理机器学习模型
        if self.lstm_predictor:
            self.lstm_predictor.cleanup()
        if self.pinn_model:
            self.pinn_model.cleanup()

        self.logger.info("洋流模拟引擎已关闭")

    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "initialized": self.is_initialized,
            "components": {
                "data_processor": self.data_processor is not None,
                "netcdf_handler": self.netcdf_handler is not None,
                "cpp_interface": self.cpp_interface is not None and self.cpp_interface.is_available(),
                "lstm_predictor": self.lstm_predictor is not None,
                "pinn_model": self.pinn_model is not None
            },
            "active_simulations": len(self.active_simulations),
            "performance": self.performance_monitor.get_metrics() if self.performance_monitor else None
        }

# 全局引擎实例
engine = OceanSimulationEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await engine.initialize_components()
    yield
    # 关闭时清理
    await engine.shutdown_components()

# 创建FastAPI应用
app = FastAPI(
    title="洋流模拟系统Python引擎",
    description="高性能海洋流模拟与预测系统的Python计算引擎",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    status = engine.get_status()
    if not status["initialized"]:
        raise HTTPException(status_code=503, detail="引擎未初始化")
    return {"status": "healthy", "engine": status}

@app.get("/status")
async def get_engine_status():
    """获取引擎详细状态"""
    return engine.get_status()

# 设置API路由
setup_routes(app, engine)

def main():
    """主函数"""
    try:
        # 从配置获取服务器参数
        server_config = engine.config["server"]

        # 启动服务器
        uvicorn.run(
            "main:app",
            host=server_config["host"],
            port=server_config["port"],
            log_level=server_config["log_level"],
            reload=False,  # 生产环境中禁用热重载
            workers=1  # 由于共享状态，使用单进程
        )

    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务...")
    except Exception as e:
        print(f"服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()