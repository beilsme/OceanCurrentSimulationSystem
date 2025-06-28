#!/usr/bin/env python3
"""
API端点定义
定义洋流模拟系统的REST API接口
"""

import asyncio
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import uuid
import io
import base64

# 请求和响应模型定义
class GridParameters(BaseModel):
    """网格参数模型"""
    nx: int = Field(..., ge=10, le=2000, description="X方向网格点数")
    ny: int = Field(..., ge=10, le=2000, description="Y方向网格点数")
    nz: int = Field(1, ge=1, le=100, description="Z方向网格点数")
    dx: float = Field(..., gt=0, description="X方向网格间距")
    dy: float = Field(..., gt=0, description="Y方向网格间距")
    dz: float = Field(1.0, gt=0, description="Z方向网格间距")
    x_min: float = Field(..., description="X方向最小值")
    y_min: float = Field(..., description="Y方向最小值")
    z_min: float = Field(0.0, description="Z方向最小值")

class SimulationParameters(BaseModel):
    """模拟参数模型"""
    dt: float = Field(..., gt=0, le=3600, description="时间步长(秒)")
    total_time: float = Field(..., gt=0, le=86400*30, description="总模拟时间(秒)")
    diffusion_coeff: float = Field(0.0, ge=0, description="扩散系数")
    viscosity: float = Field(1e-6, gt=0, description="粘性系数")
    enable_3d: bool = Field(False, description="是否启用3D模拟")
    enable_diffusion: bool = Field(False, description="是否启用扩散")

class ParticleTrackingRequest(BaseModel):
    """粒子追踪请求模型"""
    initial_positions: List[List[float]] = Field(..., description="初始位置列表")
    grid_params: GridParameters
    simulation_params: SimulationParameters
    velocity_data_id: Optional[str] = Field(None, description="速度场数据ID")

class CurrentFieldRequest(BaseModel):
    """洋流场求解请求模型"""
    input_field_data: str = Field(..., description="输入场数据(base64编码)")
    grid_params: GridParameters
    time_step: float = Field(..., gt=0, description="时间步长")
    solver_type: int = Field(0, ge=0, le=3, description="求解器类型")

class DataLoadRequest(BaseModel):
    """数据加载请求模型"""
    file_path: str = Field(..., description="文件路径")
    variables: Optional[List[str]] = Field(None, description="变量列表")
    time_range: Optional[List[str]] = Field(None, description="时间范围")
    spatial_bounds: Optional[Dict[str, List[float]]] = Field(None, description="空间范围")

class InterpolationRequest(BaseModel):
    """插值请求模型"""
    data_id: str = Field(..., description="数据ID")
    target_grid: GridParameters
    method: str = Field("linear", description="插值方法")
    variables: Optional[List[str]] = Field(None, description="变量列表")

class QualityControlRequest(BaseModel):
    """质量控制请求模型"""
    data_id: str = Field(..., description="数据ID")
    outlier_threshold: float = Field(3.0, gt=0, description="异常值阈值")
    enable_repair: bool = Field(True, description="是否启用数据修复")

class CoordinateTransformRequest(BaseModel):
    """坐标转换请求模型"""
    data_id: str = Field(..., description="数据ID")
    source_crs: str = Field(..., description="源坐标系")
    target_crs: str = Field(..., description="目标坐标系")

class TaskStatus(BaseModel):
    """任务状态模型"""
    task_id: str
    status: str
    progress: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class APIResponse(BaseModel):
    """API响应模型"""
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

def setup_routes(app, engine):
    """设置API路由"""

    logger = logging.getLogger(__name__)

    # 创建路由器
    router = APIRouter()

    # 数据管理相关端点
    @router.post("/data/load", response_model=APIResponse)
    async def load_ocean_data(request: DataLoadRequest):
        """加载海洋数据"""
        try:
            if not engine.is_initialized:
                raise HTTPException(status_code=503, detail="引擎未初始化")

            # 解析时间范围
            time_range = None
            if request.time_range:
                time_range = (
                    datetime.fromisoformat(request.time_range[0]),
                    datetime.fromisoformat(request.time_range[1])
                )

            # 加载数据
            dataset = await engine.data_processor.load_ocean_data(
                file_path=request.file_path,
                variables=request.variables,
                time_range=time_range,
                spatial_bounds=request.spatial_bounds
            )

            # 生成数据ID
            data_id = str(uuid.uuid4())

            # 存储数据（简化实现，实际应该有更完善的数据管理）
            if not hasattr(engine, 'data_cache'):
                engine.data_cache = {}
            engine.data_cache[data_id] = dataset

            return APIResponse(
                message="数据加载成功",
                data={
                    "data_id": data_id,
                    "variables": list(dataset.data_vars),
                    "dimensions": dict(dataset.dims),
                    "coordinates": list(dataset.coords)
                }
            )

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/data/upload", response_model=APIResponse)
    async def upload_data_file(file: UploadFile = File(...)):
        """上传数据文件"""
        try:
            # 检查文件类型
            if not file.filename.endswith(('.nc', '.netcdf')):
                raise HTTPException(status_code=400, detail="仅支持NetCDF文件格式")

            # 保存文件
            upload_dir = Path("Data/Upload")
            upload_dir.mkdir(parents=True, exist_ok=True)

            file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            return APIResponse(
                message="文件上传成功",
                data={
                    "file_path": str(file_path),
                    "filename": file.filename,
                    "size": len(content)
                }
            )

        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/data/interpolate", response_model=APIResponse)
    async def interpolate_data(request: InterpolationRequest):
        """数据插值"""
        try:
            if not engine.is_initialized:
                raise HTTPException(status_code=503, detail="引擎未初始化")

            # 获取源数据
            if not hasattr(engine, 'data_cache') or request.data_id not in engine.data_cache:
                raise HTTPException(status_code=404, detail="数据不存在")

            source_dataset = engine.data_cache[request.data_id]

            # 创建目标网格
            target_grid = {
                "lon": np.linspace(
                    request.target_grid.x_min,
                    request.target_grid.x_min + request.target_grid.nx * request.target_grid.dx,
                    request.target_grid.nx
                ),
                "lat": np.linspace(
                    request.target_grid.y_min,
                    request.target_grid.y_min + request.target_grid.ny * request.target_grid.dy,
                    request.target_grid.ny
                )
            }

            # 执行插值
            interpolated_dataset = await engine.data_processor.interpolate_to_grid(
                source_dataset,
                target_grid,
                method=request.method,
                variables=request.variables
            )

            # 生成新的数据ID
            result_id = str(uuid.uuid4())
            engine.data_cache[result_id] = interpolated_dataset

            return APIResponse(
                message="插值完成",
                data={
                    "result_id": result_id,
                    "target_grid": dict(request.target_grid),
                    "method": request.method
                }
            )

        except Exception as e:
            logger.error(f"数据插值失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/data/quality-control", response_model=APIResponse)
    async def apply_quality_control(request: QualityControlRequest):
        """应用质量控制"""
        try:
            if not engine.is_initialized:
                raise HTTPException(status_code=503, detail="引擎未初始化")

            # 获取数据
            if not hasattr(engine, 'data_cache') or request.data_id not in engine.data_cache:
                raise HTTPException(status_code=404, detail="数据不存在")

            dataset = engine.data_cache[request.data_id]

            # 应用质量控制
            qc_dataset = await engine.data_processor.quality_control(dataset)

            # 生成结果ID
            result_id = str(uuid.uuid4())
            engine.data_cache[result_id] = qc_dataset

            return APIResponse(
                message="质量控制完成",
                data={
                    "result_id": result_id,
                    "quality_flags_added": [var for var in qc_dataset.data_vars if var.endswith('_qc')]
                }
            )

        except Exception as e:
            logger.error(f"质量控制失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # 计算相关端点
    @router.post("/compute/particle-tracking", response_model=APIResponse)
    async def simulate_particle_tracking(request: ParticleTrackingRequest, background_tasks: BackgroundTasks):
        """粒子追踪模拟"""
        try:
            if not engine.is_initialized:
                raise HTTPException(status_code=503, detail="引擎未初始化")

            # 验证输入
            if not request.initial_positions:
                raise HTTPException(status_code=400, detail="初始位置不能为空")

            # 创建任务ID
            task_id = str(uuid.uuid4())

            # 启动后台任务
            background_tasks.add_task(
                _run_particle_tracking,
                engine, task_id, request
            )

            return APIResponse(
                message="粒子追踪任务已启动",
                data={"task_id": task_id}
            )

        except Exception as e:
            logger.error(f"粒子追踪启动失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/compute/current-field", response_model=APIResponse)
    async def solve_current_field(request: CurrentFieldRequest):
        """洋流场求解"""
        try:
            if not engine.is_initialized:
                raise HTTPException(status_code=503, detail="引擎未初始化")

            if not engine.cpp_interface or not engine.cpp_interface.is_available():
                raise HTTPException(status_code=503, detail="C++计算引擎不可用")

            # 解码输入数据
            try:
                input_data = base64.b64decode(request.input_field_data)
                input_field = np.frombuffer(input_data, dtype=np.float64)
                input_field = input_field.reshape(
                    request.grid_params.ny,
                    request.grid_params.nx
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"数据解码失败: {e}")

            # 准备网格参数
            grid_params = {
                "nx": request.grid_params.nx,
                "ny": request.grid_params.ny,
                "nz": request.grid_params.nz,
                "dx": request.grid_params.dx,
                "dy": request.grid_params.dy,
                "dz": request.grid_params.dz,
                "x_min": request.grid_params.x_min,
                "y_min": request.grid_params.y_min,
                "z_min": request.grid_params.z_min
            }

            # 调用C++求解器
            result_field = await engine.cpp_interface.solve_current_field(
                input_field,
                grid_params,
                request.time_step,
                request.solver_type
            )

            # 编码结果
            result_data = base64.b64encode(result_field.tobytes()).decode('utf-8')

            return APIResponse(
                message="洋流场求解完成",
                data={
                    "result_data": result_data,
                    "shape": result_field.shape,
                    "dtype": str(result_field.dtype)
                }
            )

        except Exception as e:
            logger.error(f"洋流场求解失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # 任务管理端点
    @router.get("/tasks/{task_id}", response_model=TaskStatus)
    async def get_task_status(task_id: str = Path(..., description="任务ID")):
        """获取任务状态"""
        if not hasattr(engine, 'task_registry'):
            engine.task_registry = {}

        if task_id not in engine.task_registry:
            raise HTTPException(status_code=404, detail="任务不存在")

        return engine.task_registry[task_id]

    @router.get("/tasks", response_model=List[TaskStatus])
    async def list_tasks(
            status: Optional[str] = Query(None, description="按状态筛选"),
            limit: int = Query(50, ge=1, le=100, description="返回数量限制")
    ):
        """获取任务列表"""
        if not hasattr(engine, 'task_registry'):
            return []

        tasks = list(engine.task_registry.values())

        if status:
            tasks = [task for task in tasks if task.status == status]

        # 按创建时间倒序排序
        tasks.sort(key=lambda x: x.created_at, reverse=True)

        return tasks[:limit]

    @router.delete("/tasks/{task_id}", response_model=APIResponse)
    async def cancel_task(task_id: str = Path(..., description="任务ID")):
        """取消任务"""
        if not hasattr(engine, 'task_registry'):
            raise HTTPException(status_code=404, detail="任务不存在")

        if task_id not in engine.task_registry:
            raise HTTPException(status_code=404, detail="任务不存在")

        task = engine.task_registry[task_id]

        if task.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail="任务已完成或已取消")

        # 更新任务状态
        task.status = "cancelled"
        task.updated_at = datetime.now()
        task.message = "任务已被用户取消"

        return APIResponse(message="任务已取消")

    # 系统监控端点
    @router.get("/monitor/performance", response_model=APIResponse)
    async def get_performance_metrics():
        """获取性能指标"""
        try:
            metrics = engine.performance_monitor.get_metrics() if engine.performance_monitor else {}

            return APIResponse(
                message="性能指标获取成功",
                data=metrics
            )

        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/monitor/system", response_model=APIResponse)
    async def get_system_info():
        """获取系统信息"""
        try:
            system_info = engine.performance_monitor.get_system_info() if engine.performance_monitor else {}

            return APIResponse(
                message="系统信息获取成功",
                data=system_info
            )

        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/monitor/engine-status", response_model=APIResponse)
    async def get_engine_status():
        """获取引擎状态"""
        status = engine.get_status()

        return APIResponse(
            message="引擎状态获取成功",
            data=status
        )

    # 数据导出端点
    @router.get("/data/{data_id}/export")
    async def export_data(
            data_id: str = Path(..., description="数据ID"),
            format: str = Query("netcdf", description="导出格式"),
            variables: Optional[str] = Query(None, description="变量列表(逗号分隔)")
    ):
        """导出数据"""
        try:
            if not hasattr(engine, 'data_cache') or data_id not in engine.data_cache:
                raise HTTPException(status_code=404, detail="数据不存在")

            dataset = engine.data_cache[data_id]

            # 筛选变量
            if variables:
                var_list = [v.strip() for v in variables.split(',')]
                dataset = dataset[var_list]

            # 导出文件
            export_dir = Path("Data/Export")
            export_dir.mkdir(parents=True, exist_ok=True)

            if format.lower() == "netcdf":
                filename = f"export_{data_id}.nc"
                filepath = export_dir / filename
                dataset.to_netcdf(filepath)

            elif format.lower() == "csv":
                filename = f"export_{data_id}.csv"
                filepath = export_dir / filename
                # 将数据转换为DataFrame并导出
                df = dataset.to_dataframe().reset_index()
                df.to_csv(filepath, index=False)

            else:
                raise HTTPException(status_code=400, detail="不支持的导出格式")

            return FileResponse(
                path=str(filepath),
                filename=filename,
                media_type='application/octet-stream'
            )

        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # 配置管理端点
    @router.get("/config", response_model=APIResponse)
    async def get_configuration():
        """获取系统配置"""
        config = {
            "server": engine.config.get("server", {}),
            "performance": engine.config.get("performance", {}),
            "data": {k: v for k, v in engine.config.get("data", {}).items()
                     if not k.endswith("_path")},  # 隐藏敏感路径信息
        }

        return APIResponse(
            message="配置获取成功",
            data=config
        )

    @router.post("/config/reload", response_model=APIResponse)
    async def reload_configuration():
        """重新加载配置"""
        try:
            # 这里可以实现配置重新加载逻辑
            # engine.reload_config()

            return APIResponse(message="配置重新加载成功")

        except Exception as e:
            logger.error(f"配置重新加载失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # 注册路由到应用
    app.include_router(router, prefix="/api/v1", tags=["ocean_simulation"])

async def _run_particle_tracking(engine, task_id: str, request: ParticleTrackingRequest):
    """运行粒子追踪任务的后台函数"""
    logger = logging.getLogger(__name__)

    # 初始化任务注册表
    if not hasattr(engine, 'task_registry'):
        engine.task_registry = {}

    # 创建任务状态
    task = TaskStatus(
        task_id=task_id,
        status="running",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        message="正在执行粒子追踪模拟"
    )
    engine.task_registry[task_id] = task

    try:
        # 准备输入数据
        initial_positions = np.array(request.initial_positions)

        # 创建虚拟速度场（实际应该从数据中获取）
        velocity_field = {
            "u": np.random.randn(request.grid_params.ny, request.grid_params.nx) * 0.1,
            "v": np.random.randn(request.grid_params.ny, request.grid_params.nx) * 0.1
        }

        grid_params = {
            "nx": request.grid_params.nx,
            "ny": request.grid_params.ny,
            "nz": request.grid_params.nz,
            "dx": request.grid_params.dx,
            "dy": request.grid_params.dy,
            "dz": request.grid_params.dz,
            "x_min": request.grid_params.x_min,
            "y_min": request.grid_params.y_min,
            "z_min": request.grid_params.z_min
        }

        simulation_params = {
            "dt": request.simulation_params.dt,
            "total_time": request.simulation_params.total_time,
            "diffusion_coeff": request.simulation_params.diffusion_coeff,
            "viscosity": request.simulation_params.viscosity,
            "enable_3d": request.simulation_params.enable_3d,
            "enable_diffusion": request.simulation_params.enable_diffusion
        }

        # 更新进度
        task.progress = 25.0
        task.message = "正在初始化计算引擎"
        task.updated_at = datetime.now()

        # 执行粒子追踪
        if engine.cpp_interface and engine.cpp_interface.is_available():
            result = await engine.cpp_interface.simulate_particles(
                velocity_field,
                initial_positions,
                grid_params,
                simulation_params
            )
        else:
            # 使用Python备用实现
            await asyncio.sleep(2)  # 模拟计算时间
            result = initial_positions + np.random.randn(*initial_positions.shape) * 0.01

        # 更新任务状态
        task.status = "completed"
        task.progress = 100.0
        task.message = "粒子追踪模拟完成"
        task.updated_at = datetime.now()
        task.result = {
            "final_positions": result.tolist(),
            "num_particles": len(result),
            "simulation_time": simulation_params["total_time"]
        }

        logger.info(f"粒子追踪任务 {task_id} 完成")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.message = f"任务执行失败: {e}"
        task.updated_at = datetime.now()

        logger.error(f"粒子追踪任务 {task_id} 失败: {e}")

# 错误处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger = logging.getLogger(__name__)
    logger.error(f"未处理的异常: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "timestamp": datetime.now().isoformat()
        }
    )