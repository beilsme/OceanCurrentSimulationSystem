#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: interpolation_engine.py
模块: Source.PythonEngine.core.interpolation_engine
功能: 提供高性能空间、时间、时空插值算法，支持二维/三维/矢量场/时序多种插值方法，并支持高阶插值器缓存与性能统计。
作者: beilsm
版本: v1.0.0
创建时间: 2025-06-28
最近更新: 2025-06-28
主要功能:
    - 支持LINEAR、CUBIC、NEAREST、BILINEAR、BICUBIC、SPLINE、RBF等多种插值方式
    - 支持2D/3D标量场、2D矢量场、时间序列、时空联合插值
    - 插值器LRU缓存管理、插值性能统计、可插拔配置参数
    - 与xarray、numpy等主流数据结构无缝集成
较上一版改进:
    - 首发版，全插值类型与缓存机制实现
接口说明:
    - interpolate_2d_scalar(), interpolate_2d_vector(), interpolate_3d_scalar(), interpolate_temporal(), interpolate_spatiotemporal()
    - get_statistics(), clear_cache(), cleanup()
测试方法:
    文件结尾可加 if __name__ == "__main__": 独立测试
"""


import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
from enum import Enum
import warnings

class InterpolationMethod(Enum):
    """插值方法枚举"""
    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SPLINE = "spline"
    RBF = "rbf"  # 径向基函数
    KRIGING = "kriging"  # 克里金插值

class InterpolationEngine:
    """插值计算引擎类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化插值引擎
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 性能统计
        self.stats = {
            "total_interpolations": 0,
            "total_time": 0.0,
            "method_usage": {},
            "cache_hits": 0,
            "cache_misses": 0
        }

        # 插值缓存
        self.interpolator_cache = {}
        self.cache_max_size = self.config.get("cache_max_size", 50)

        # 默认参数
        self.default_fill_value = self.config.get("default_fill_value", np.nan)
        self.enable_bounds_check = self.config.get("enable_bounds_check", True)
        self.enable_extrapolation = self.config.get("enable_extrapolation", False)

    def interpolate_2d_scalar(
            self,
            data: np.ndarray,
            source_coords: Tuple[np.ndarray, np.ndarray],
            target_coords: Tuple[np.ndarray, np.ndarray],
            method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            fill_value: Optional[float] = None,
            bounds_error: bool = False
    ) -> np.ndarray:
        """
        2D标量场插值
        
        Args:
            data: 源数据 (ny, nx)
            source_coords: 源坐标 (lon_array, lat_array)
            target_coords: 目标坐标 (target_lon, target_lat)
            method: 插值方法
            fill_value: 填充值
            bounds_error: 是否在超出边界时报错
            
        Returns:
            插值后的数据
        """
        start_time = time.time()

        try:
            # 参数预处理
            if isinstance(method, str):
                method = InterpolationMethod(method)

            if fill_value is None:
                fill_value = self.default_fill_value

            source_lon, source_lat = source_coords
            target_lon, target_lat = target_coords

            # 检查输入数据
            if data.shape != (len(source_lat), len(source_lon)):
                raise ValueError(f"数据形状 {data.shape} 与坐标不匹配 ({len(source_lat)}, {len(source_lon)})")

            # 创建插值器缓存键
            cache_key = self._create_cache_key(
                "2d_scalar",
                source_coords,
                method,
                data.shape
            )

            # 尝试从缓存获取插值器
            interpolator = self._get_cached_interpolator(cache_key)

            if interpolator is None:
                # 创建新的插值器
                interpolator = self._create_2d_interpolator(
                    source_lon, source_lat, method, bounds_error, fill_value
                )
                self._cache_interpolator(cache_key, interpolator)
                self.stats["cache_misses"] += 1
            else:
                self.stats["cache_hits"] += 1

            # 执行插值
            if method in [InterpolationMethod.LINEAR, InterpolationMethod.CUBIC, InterpolationMethod.NEAREST]:
                # 使用scipy.interpolate.RegularGridInterpolator
                result = interpolator((target_lat, target_lon))

            elif method in [InterpolationMethod.BILINEAR, InterpolationMethod.BICUBIC]:
                # 使用scipy.interpolate.RectBivariateSpline
                result = interpolator(target_lat, target_lon, grid=False)

            elif method == InterpolationMethod.RBF:
                # 径向基函数插值
                result = self._rbf_interpolate_2d(
                    data, source_lon, source_lat, target_lon, target_lat, fill_value
                )

            elif method == InterpolationMethod.SPLINE:
                # 样条插值
                result = self._spline_interpolate_2d(
                    data, source_lon, source_lat, target_lon, target_lat, fill_value
                )

            else:
                raise ValueError(f"不支持的插值方法: {method}")

            # 处理结果形状
            if isinstance(target_lon, np.ndarray) and isinstance(target_lat, np.ndarray):
                if target_lon.ndim == 1 and target_lat.ndim == 1:
                    # 如果是1D坐标数组，创建网格
                    result = result.reshape(len(target_lat), len(target_lon))

            # 更新统计信息
            elapsed_time = time.time() - start_time
            self._update_stats(method, elapsed_time)

            self.logger.debug(f"2D插值完成: {method.value}, 耗时: {elapsed_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"2D插值失败: {e}")
            raise

    def interpolate_2d_vector(
            self,
            u_data: np.ndarray,
            v_data: np.ndarray,
            source_coords: Tuple[np.ndarray, np.ndarray],
            target_coords: Tuple[np.ndarray, np.ndarray],
            method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            fill_value: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D矢量场插值
        
        Args:
            u_data: U分量数据
            v_data: V分量数据
            source_coords: 源坐标
            target_coords: 目标坐标
            method: 插值方法
            fill_value: 填充值
            
        Returns:
            插值后的(u, v)分量
        """
        # 分别插值u和v分量
        u_interp = self.interpolate_2d_scalar(
            u_data, source_coords, target_coords, method, fill_value
        )

        v_interp = self.interpolate_2d_scalar(
            v_data, source_coords, target_coords, method, fill_value
        )

        return u_interp, v_interp

    def interpolate_3d_scalar(
            self,
            data: np.ndarray,
            source_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
            target_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
            method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            fill_value: Optional[float] = None
    ) -> np.ndarray:
        """
        3D标量场插值
        
        Args:
            data: 源数据 (nz, ny, nx)
            source_coords: 源坐标 (lon_array, lat_array, depth_array)
            target_coords: 目标坐标 (target_lon, target_lat, target_depth)
            method: 插值方法
            fill_value: 填充值
            
        Returns:
            插值后的数据
        """
        start_time = time.time()

        try:
            if isinstance(method, str):
                method = InterpolationMethod(method)

            if fill_value is None:
                fill_value = self.default_fill_value

            source_lon, source_lat, source_depth = source_coords
            target_lon, target_lat, target_depth = target_coords

            # 检查数据形状
            if data.shape != (len(source_depth), len(source_lat), len(source_lon)):
                raise ValueError(f"数据形状与坐标不匹配")

            # 创建3D插值器
            if method == InterpolationMethod.LINEAR:
                interpolator = interpolate.RegularGridInterpolator(
                    (source_depth, source_lat, source_lon),
                    data,
                    method='linear',
                    bounds_error=False,
                    fill_value=fill_value
                )

                # 准备目标点
                if isinstance(target_lon, np.ndarray):
                    points = np.column_stack([
                        target_depth.ravel(),
                        target_lat.ravel(),
                        target_lon.ravel()
                    ])
                    result = interpolator(points)
                    result = result.reshape(target_depth.shape)
                else:
                    result = interpolator((target_depth, target_lat, target_lon))

            elif method == InterpolationMethod.NEAREST:
                interpolator = interpolate.RegularGridInterpolator(
                    (source_depth, source_lat, source_lon),
                    data,
                    method='nearest',
                    bounds_error=False,
                    fill_value=fill_value
                )

                if isinstance(target_lon, np.ndarray):
                    points = np.column_stack([
                        target_depth.ravel(),
                        target_lat.ravel(),
                        target_lon.ravel()
                    ])
                    result = interpolator(points)
                    result = result.reshape(target_depth.shape)
                else:
                    result = interpolator((target_depth, target_lat, target_lon))

            else:
                # 对于其他方法，逐层进行2D插值
                result = self._layered_3d_interpolation(
                    data, source_coords, target_coords, method, fill_value
                )

            elapsed_time = time.time() - start_time
            self._update_stats(method, elapsed_time)

            self.logger.debug(f"3D插值完成: {method.value}, 耗时: {elapsed_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"3D插值失败: {e}")
            raise

    def interpolate_temporal(
            self,
            data: np.ndarray,
            source_times: np.ndarray,
            target_times: np.ndarray,
            method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            fill_value: Optional[float] = None
    ) -> np.ndarray:
        """
        时间插值
        
        Args:
            data: 时间序列数据 (nt, ...)
            source_times: 源时间点
            target_times: 目标时间点
            method: 插值方法
            fill_value: 填充值
            
        Returns:
            插值后的数据
        """
        start_time = time.time()

        try:
            if isinstance(method, str):
                method = InterpolationMethod(method)

            if fill_value is None:
                fill_value = self.default_fill_value

            # 检查时间维度
            if data.shape[0] != len(source_times):
                raise ValueError("数据时间维度与时间坐标不匹配")

            # 准备结果数组
            output_shape = (len(target_times),) + data.shape[1:]
            result = np.full(output_shape, fill_value, dtype=data.dtype)

            # 处理不同的插值方法
            if method == InterpolationMethod.LINEAR:
                # 线性插值
                for i in range(len(target_times)):
                    t = target_times[i]

                    # 检查是否在范围内
                    if t < source_times[0] or t > source_times[-1]:
                        if not self.enable_extrapolation:
                            continue  # 保持填充值

                    # 找到相邻的时间点
                    idx = np.searchsorted(source_times, t)

                    if idx == 0:
                        result[i] = data[0]
                    elif idx >= len(source_times):
                        result[i] = data[-1]
                    else:
                        # 线性插值
                        t1, t2 = source_times[idx-1], source_times[idx]
                        w = (t - t1) / (t2 - t1)
                        result[i] = (1 - w) * data[idx-1] + w * data[idx]

            elif method == InterpolationMethod.CUBIC:
                # 三次样条插值
                if data.ndim == 1:
                    # 1D数据
                    interpolator = interpolate.interp1d(
                        source_times, data, kind='cubic',
                        bounds_error=False, fill_value=fill_value
                    )
                    result = interpolator(target_times)
                else:
                    # 多维数据，沿时间轴插值
                    for idx in np.ndindex(data.shape[1:]):
                        time_series = data[(slice(None),) + idx]
                        interpolator = interpolate.interp1d(
                            source_times, time_series, kind='cubic',
                            bounds_error=False, fill_value=fill_value
                        )
                        result[(slice(None),) + idx] = interpolator(target_times)

            elif method == InterpolationMethod.NEAREST:
                # 最近邻插值
                for i, t in enumerate(target_times):
                    nearest_idx = np.argmin(np.abs(source_times - t))
                    result[i] = data[nearest_idx]

            else:
                raise ValueError(f"时间插值不支持方法: {method}")

            elapsed_time = time.time() - start_time
            self._update_stats(method, elapsed_time)

            self.logger.debug(f"时间插值完成: {method.value}, 耗时: {elapsed_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"时间插值失败: {e}")
            raise

    def interpolate_spatiotemporal(
            self,
            data: xr.DataArray,
            target_coords: Dict[str, np.ndarray],
            spatial_method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            temporal_method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
            fill_value: Optional[float] = None
    ) -> xr.DataArray:
        """
        时空插值
        
        Args:
            data: 输入数据数组
            target_coords: 目标坐标 {"time": array, "lat": array, "lon": array}
            spatial_method: 空间插值方法
            temporal_method: 时间插值方法
            fill_value: 填充值
            
        Returns:
            插值后的数据数组
        """
        start_time = time.time()

        try:
            if fill_value is None:
                fill_value = self.default_fill_value

            # 首先进行时间插值
            if "time" in target_coords and "time" in data.dims:
                target_times = target_coords["time"]
                source_times = data.time.values

                # 检查是否需要时间插值
                if not np.array_equal(source_times, target_times):
                    # 重新排列维度，将时间维度放在第一位
                    dims_order = ["time"] + [d for d in data.dims if d != "time"]
                    data_reordered = data.transpose(*dims_order)

                    # 时间插值
                    data_time_interp = self.interpolate_temporal(
                        data_reordered.values,
                        source_times,
                        target_times,
                        temporal_method,
                        fill_value
                    )

                    # 创建新的坐标
                    new_coords = data.coords.copy()
                    new_coords["time"] = target_times

                    # 创建新的数据数组
                    data = xr.DataArray(
                        data_time_interp,
                        dims=dims_order,
                        coords=new_coords,
                        attrs=data.attrs
                    )

            # 然后进行空间插值
            if ("lat" in target_coords and "lon" in target_coords and
                    "lat" in data.dims and "lon" in data.dims):

                target_lat = target_coords["lat"]
                target_lon = target_coords["lon"]
                source_lat = data.lat.values
                source_lon = data.lon.values

                # 检查是否需要空间插值
                if (not np.array_equal(source_lat, target_lat) or
                        not np.array_equal(source_lon, target_lon)):

                    # 空间插值
                    result_data = self._spatiotemporal_interpolation(
                        data, target_lat, target_lon, spatial_method, fill_value
                    )

                    # 更新坐标
                    new_coords = data.coords.copy()
                    new_coords["lat"] = target_lat
                    new_coords["lon"] = target_lon

                    # 创建结果数据数组
                    result = xr.DataArray(
                        result_data,
                        dims=data.dims,
                        coords=new_coords,
                        attrs=data.attrs
                    )
                else:
                    result = data
            else:
                result = data

            elapsed_time = time.time() - start_time
            self.logger.debug(f"时空插值完成，耗时: {elapsed_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"时空插值失败: {e}")
            raise

    def _create_2d_interpolator(
            self,
            lon: np.ndarray,
            lat: np.ndarray,
            method: InterpolationMethod,
            bounds_error: bool,
            fill_value: float
    ):
        """创建2D插值器"""
        if method == InterpolationMethod.LINEAR:
            return interpolate.RegularGridInterpolator(
                (lat, lon), np.zeros((len(lat), len(lon))),
                method='linear', bounds_error=bounds_error, fill_value=fill_value
            )
        elif method == InterpolationMethod.CUBIC:
            return interpolate.RegularGridInterpolator(
                (lat, lon), np.zeros((len(lat), len(lon))),
                method='cubic', bounds_error=bounds_error, fill_value=fill_value
            )
        elif method == InterpolationMethod.NEAREST:
            return interpolate.RegularGridInterpolator(
                (lat, lon), np.zeros((len(lat), len(lon))),
                method='nearest', bounds_error=bounds_error, fill_value=fill_value
            )
        else:
            return None

    def _rbf_interpolate_2d(
            self,
            data: np.ndarray,
            source_lon: np.ndarray,
            source_lat: np.ndarray,
            target_lon: np.ndarray,
            target_lat: np.ndarray,
            fill_value: float
    ) -> np.ndarray:
        """径向基函数插值"""
        # 创建源点坐标
        lon_grid, lat_grid = np.meshgrid(source_lon, source_lat)
        points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
        values = data.ravel()

        # 移除NaN值
        valid_mask = ~np.isnan(values)
        points = points[valid_mask]
        values = values[valid_mask]

        # 创建RBF插值器
        rbf = interpolate.Rbf(points[:, 0], points[:, 1], values, function='linear')

        # 插值到目标点
        if isinstance(target_lon, np.ndarray) and isinstance(target_lat, np.ndarray):
            target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
            result = rbf(target_lon_grid, target_lat_grid)
        else:
            result = rbf(target_lon, target_lat)

        return result

    def _spline_interpolate_2d(
            self,
            data: np.ndarray,
            source_lon: np.ndarray,
            source_lat: np.ndarray,
            target_lon: np.ndarray,
            target_lat: np.ndarray,
            fill_value: float
    ) -> np.ndarray:
        """样条插值"""
        # 使用RectBivariateSpline
        spline = interpolate.RectBivariateSpline(
            source_lat, source_lon, data, kx=3, ky=3
        )

        if isinstance(target_lon, np.ndarray) and isinstance(target_lat, np.ndarray):
            result = spline(target_lat, target_lon)
        else:
            result = spline([target_lat], [target_lon])[0, 0]

        return result

    def _layered_3d_interpolation(
            self,
            data: np.ndarray,
            source_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
            target_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
            method: InterpolationMethod,
            fill_value: float
    ) -> np.ndarray:
        """分层3D插值"""
        source_lon, source_lat, source_depth = source_coords
        target_lon, target_lat, target_depth = target_coords

        # 首先在水平方向插值
        horizontal_result = np.full(
            (len(source_depth), len(target_lat), len(target_lon)),
            fill_value
        )

        for z_idx in range(len(source_depth)):
            try:
                horizontal_result[z_idx] = self.interpolate_2d_scalar(
                    data[z_idx],
                    (source_lon, source_lat),
                    (target_lon, target_lat),
                    method,
                    fill_value
                )
            except Exception as e:
                self.logger.warning(f"层 {z_idx} 水平插值失败: {e}")
                horizontal_result[z_idx] = fill_value

        # 然后在垂直方向插值
        if not np.array_equal(source_depth, target_depth):
            final_result = np.full(
                (len(target_depth), len(target_lat), len(target_lon)),
                fill_value
            )

            for lat_idx in range(len(target_lat)):
                for lon_idx in range(len(target_lon)):
                    profile = horizontal_result[:, lat_idx, lon_idx]
                    if not np.all(np.isnan(profile)):
                        interpolator = interpolate.interp1d(
                            source_depth, profile, kind='linear',
                            bounds_error=False, fill_value=fill_value
                        )
                        final_result[:, lat_idx, lon_idx] = interpolator(target_depth)

            return final_result
        else:
            return horizontal_result

    def _spatiotemporal_interpolation(
            self,
            data: xr.DataArray,
            target_lat: np.ndarray,
            target_lon: np.ndarray,
            method: InterpolationMethod,
            fill_value: float
    ) -> np.ndarray:
        """时空数据的空间插值"""
        source_lat = data.lat.values
        source_lon = data.lon.values

        # 确定数据维度顺序
        lat_dim = data.dims.index("lat")
        lon_dim = data.dims.index("lon")

        # 准备结果数组
        result_shape = list(data.shape)
        result_shape[lat_dim] = len(target_lat)
        result_shape[lon_dim] = len(target_lon)
        result = np.full(result_shape, fill_value)

        # 根据数据维度进行插值
        if data.ndim == 2:  # (lat, lon)
            result = self.interpolate_2d_scalar(
                data.values, (source_lon, source_lat),
                (target_lon, target_lat), method, fill_value
            )
        elif data.ndim == 3:  # (time, lat, lon) 或其他组合
            if "time" in data.dims:
                time_dim = data.dims.index("time")
                for t in range(data.shape[time_dim]):
                    if time_dim == 0:
                        slice_data = data.values[t, :, :]
                        result[t, :, :] = self.interpolate_2d_scalar(
                            slice_data, (source_lon, source_lat),
                            (target_lon, target_lat), method, fill_value
                        )
                    # 处理其他维度顺序...
            elif "depth" in data.dims:
                depth_dim = data.dims.index("depth")
                for z in range(data.shape[depth_dim]):
                    if depth_dim == 0:
                        slice_data = data.values[z, :, :]
                        result[z, :, :] = self.interpolate_2d_scalar(
                            slice_data, (source_lon, source_lat),
                            (target_lon, target_lat), method, fill_value
                        )

        return result

    def _create_cache_key(self, operation: str, coords, method, shape) -> str:
        """创建缓存键"""
        import hashlib
        key_str = f"{operation}_{method.value}_{shape}"
        if isinstance(coords, tuple):
            for coord in coords:
                if isinstance(coord, np.ndarray):
                    key_str += f"_{coord.shape}_{coord.dtype}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cached_interpolator(self, cache_key: str):
        """获取缓存的插值器"""
        return self.interpolator_cache.get(cache_key)

    def _cache_interpolator(self, cache_key: str, interpolator):
        """缓存插值器"""
        if len(self.interpolator_cache) >= self.cache_max_size:
            # 移除最老的缓存项
            oldest_key = next(iter(self.interpolator_cache))
            del self.interpolator_cache[oldest_key]

        self.interpolator_cache[cache_key] = interpolator

    def _update_stats(self, method: InterpolationMethod, elapsed_time: float):
        """更新统计信息"""
        self.stats["total_interpolations"] += 1
        self.stats["total_time"] += elapsed_time

        method_name = method.value
        if method_name not in self.stats["method_usage"]:
            self.stats["method_usage"][method_name] = 0
        self.stats["method_usage"][method_name] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取插值统计信息"""
        avg_time = (
            self.stats["total_time"] / self.stats["total_interpolations"]
            if self.stats["total_interpolations"] > 0 else 0
        )

        cache_hit_rate = (
            self.stats["cache_hits"] /
            (self.stats["cache_hits"] + self.stats["cache_misses"])
            if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
        )

        return {
            "total_interpolations": self.stats["total_interpolations"],
            "total_time": self.stats["total_time"],
            "average_time": avg_time,
            "method_usage": self.stats["method_usage"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.interpolator_cache)
        }

    def clear_cache(self):
        """清理插值器缓存"""
        self.interpolator_cache.clear()
        self.logger.info("插值器缓存已清理")

    def cleanup(self):
        """清理资源"""
        self.clear_cache()
        self.logger.info("插值引擎已关闭")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    eng = InterpolationEngine()
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    data = np.outer(np.sin(x), np.cos(y))
    target_x = np.linspace(0, 10, 21)
    target_y = np.linspace(0, 10, 21)
    target_xx, target_yy = np.meshgrid(target_x, target_y)

    # 传入 meshgrid 展开的一维点坐标
    result = eng.interpolate_2d_scalar(
        data,
        (x, y),
        (target_xx.ravel(), target_yy.ravel()),
        method="linear"
    )
    result = result.reshape(target_xx.shape)
    print("2D线性插值shape:", result.shape)
    print("插值统计信息:", eng.get_statistics())
