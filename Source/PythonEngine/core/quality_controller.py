#!/usr/bin/env python3
"""
数据质量控制器
负责海洋数据的质量检查、异常值检测和数据清洗
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from enum import Enum
import warnings
from scipy import stats
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

class QualityFlag(Enum):
    """质量标志枚举"""
    GOOD = 0           # 良好数据
    SUSPECT = 1        # 可疑数据
    BAD = 2           # 坏数据
    MISSING = 3       # 缺失数据
    INTERPOLATED = 4  # 插值数据
    EXTRAPOLATED = 5  # 外推数据

class QualityController:
    """数据质量控制器类"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化质量控制器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 质量控制参数
        self.outlier_threshold = self.config.get("outlier_threshold", 3.0)
        self.missing_threshold = self.config.get("missing_threshold", 0.5)
        self.gradient_threshold = self.config.get("gradient_threshold", 5.0)
        self.temporal_threshold = self.config.get("temporal_threshold", 2.0)

        # 物理约束参数
        self.physical_limits = self.config.get("physical_limits", {
            "temperature": (-5.0, 50.0),    # 海表温度范围 (°C)
            "salinity": (0.0, 50.0),        # 盐度范围 (PSU)
            "speed": (0.0, 5.0),            # 流速范围 (m/s)
            "u_velocity": (-3.0, 3.0),      # U分量范围 (m/s)
            "v_velocity": (-3.0, 3.0),      # V分量范围 (m/s)
            "w_velocity": (-0.5, 0.5),      # W分量范围 (m/s)
            "ssh": (-3.0, 3.0),             # 海表高度异常 (m)
            "pressure": (0.0, 1100.0)       # 压强范围 (dbar)
        })

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "good_data": 0,
            "suspect_data": 0,
            "bad_data": 0,
            "missing_data": 0,
            "interpolated_data": 0,
            "outliers_detected": 0,
            "gradient_failures": 0,
            "physical_violations": 0
        }

    def process_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        处理整个数据集的质量控制
        
        Args:
            dataset: 输入数据集
            
        Returns:
            质量控制后的数据集
        """
        self.logger.info("开始数据质量控制处理")

        # 创建结果数据集副本
        result_dataset = dataset.copy(deep=True)

        # 为每个变量添加质量标志
        for var_name in dataset.data_vars:
            qc_var_name = f"{var_name}_qc"
            result_dataset[qc_var_name] = xr.full_like(
                dataset[var_name], QualityFlag.GOOD.value, dtype=np.int8
            )
            result_dataset[qc_var_name].attrs = {
                "long_name": f"Quality flag for {var_name}",
                "flag_values": [f.value for f in QualityFlag],
                "flag_meanings": " ".join([f.name.lower() for f in QualityFlag])
            }

        # 对每个数据变量进行质量控制
        for var_name in dataset.data_vars:
            self.logger.debug(f"处理变量: {var_name}")

            data_var = dataset[var_name]
            qc_flags = result_dataset[f"{var_name}_qc"]

            # 1. 物理约束检查
            qc_flags = self._check_physical_limits(data_var, qc_flags, var_name)

            # 2. 统计异常值检测
            qc_flags = self._detect_statistical_outliers(data_var, qc_flags)

            # 3. 空间梯度检查
            if "lat" in data_var.dims and "lon" in data_var.dims:
                qc_flags = self._check_spatial_gradients(data_var, qc_flags)

            # 4. 时间连续性检查
            if "time" in data_var.dims:
                qc_flags = self._check_temporal_consistency(data_var, qc_flags)

            # 5. 缺失值标记
            qc_flags = self._mark_missing_values(data_var, qc_flags)

            # 6. 数据修复和插值
            result_dataset[var_name], qc_flags = self._repair_data(
                data_var, qc_flags
            )

            # 更新质量标志
            result_dataset[f"{var_name}_qc"] = qc_flags

        # 更新全局属性
        self._update_global_attributes(result_dataset)

        # 更新统计信息
        self._update_statistics(result_dataset)

        self.logger.info(f"质量控制完成，处理了 {len(dataset.data_vars)} 个变量")
        return result_dataset

    def _check_physical_limits(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray,
            var_name: str
    ) -> xr.DataArray:
        """检查物理约束"""
        # 确定变量类型
        var_type = self._identify_variable_type(var_name, data_var)

        if var_type not in self.physical_limits:
            return qc_flags

        min_val, max_val = self.physical_limits[var_type]

        # 检查超出物理范围的值
        out_of_range = (data_var < min_val) | (data_var > max_val)

        # 更新质量标志
        qc_flags = qc_flags.where(~out_of_range, QualityFlag.BAD.value)

        violations = int(out_of_range.sum())
        if violations > 0:
            self.stats["physical_violations"] += violations
            self.logger.warning(
                f"变量 {var_name} 发现 {violations} 个物理约束违反"
            )

        return qc_flags

    def _identify_variable_type(self, var_name: str, data_var: xr.DataArray) -> str:
        """识别变量类型"""
        name_lower = var_name.lower()

        # 温度相关
        if any(keyword in name_lower for keyword in ['temp', 'sst', 'temperature']):
            return "temperature"

        # 盐度相关
        elif any(keyword in name_lower for keyword in ['sal', 'salinity', 'psal']):
            return "salinity"

        # 速度相关
        elif any(keyword in name_lower for keyword in ['speed', 'velocity_magnitude']):
            return "speed"
        elif any(keyword in name_lower for keyword in ['u', 'u_velocity', 'water_u']):
            return "u_velocity"
        elif any(keyword in name_lower for keyword in ['v', 'v_velocity', 'water_v']):
            return "v_velocity"
        elif any(keyword in name_lower for keyword in ['w', 'w_velocity', 'water_w']):
            return "w_velocity"

        # 海表高度
        elif any(keyword in name_lower for keyword in ['ssh', 'sea_surface_height', 'sla']):
            return "ssh"

        # 压强
        elif any(keyword in name_lower for keyword in ['pressure', 'pres']):
            return "pressure"

        # 检查单位属性
        elif hasattr(data_var, 'attrs') and 'units' in data_var.attrs:
            units = data_var.attrs['units'].lower()
            if 'celsius' in units or '°c' in units:
                return "temperature"
            elif 'psu' in units or 'practical salinity' in units:
                return "salinity"
            elif 'm/s' in units:
                return "speed"
            elif 'dbar' in units or 'decibar' in units:
                return "pressure"

        return "unknown"

    def _detect_statistical_outliers(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray
    ) -> xr.DataArray:
        """统计异常值检测"""
        try:
            # 只对数值数据进行检测
            valid_data = data_var.where(~np.isnan(data_var))

            if valid_data.count() < 10:  # 数据点太少，跳过检测
                return qc_flags

            # 方法1: Z-score方法
            z_scores = np.abs(stats.zscore(valid_data, nan_policy='omit'))
            outliers_zscore = z_scores > self.outlier_threshold

            # 方法2: IQR方法
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = (valid_data < lower_bound) | (valid_data > upper_bound)

            # 方法3: 修正Z-score方法（基于中位数）
            median = valid_data.median()
            mad = np.median(np.abs(valid_data - median))
            modified_z_scores = 0.6745 * (valid_data - median) / mad
            outliers_modified = np.abs(modified_z_scores) > self.outlier_threshold

            # 组合多种方法的结果
            outliers_combined = outliers_zscore & outliers_iqr & outliers_modified

            # 更新质量标志
            qc_flags = qc_flags.where(~outliers_combined, QualityFlag.SUSPECT.value)

            outlier_count = int(outliers_combined.sum())
            if outlier_count > 0:
                self.stats["outliers_detected"] += outlier_count
                self.logger.debug(f"检测到 {outlier_count} 个统计异常值")

        except Exception as e:
            self.logger.warning(f"统计异常值检测失败: {e}")

        return qc_flags

    def _check_spatial_gradients(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray
    ) -> xr.DataArray:
        """空间梯度检查"""
        try:
            # 计算空间梯度
            if "lat" in data_var.dims and "lon" in data_var.dims:
                # 计算经度方向梯度
                lon_grad = data_var.differentiate("lon")
                lat_grad = data_var.differentiate("lat")

                # 计算总梯度幅度
                gradient_magnitude = np.sqrt(lon_grad**2 + lat_grad**2)

                # 计算梯度阈值（基于数据的统计特性）
                grad_threshold = gradient_magnitude.quantile(0.95) * 2

                # 如果用户设置了阈值，使用较小的值
                if hasattr(self, 'gradient_threshold'):
                    grad_threshold = min(grad_threshold, self.gradient_threshold)

                # 标记梯度过大的点
                steep_gradients = gradient_magnitude > grad_threshold

                # 更新质量标志
                qc_flags = qc_flags.where(~steep_gradients, QualityFlag.SUSPECT.value)

                gradient_failures = int(steep_gradients.sum())
                if gradient_failures > 0:
                    self.stats["gradient_failures"] += gradient_failures
                    self.logger.debug(f"检测到 {gradient_failures} 个空间梯度异常")

        except Exception as e:
            self.logger.warning(f"空间梯度检查失败: {e}")

        return qc_flags

    def _check_temporal_consistency(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray
    ) -> xr.DataArray:
        """时间连续性检查"""
        try:
            if "time" in data_var.dims and data_var.sizes["time"] > 2:
                # 计算时间差分
                time_diff = data_var.differentiate("time")

                # 计算时间变化率阈值
                time_threshold = time_diff.std() * self.temporal_threshold

                # 标记时间变化过大的点
                rapid_changes = np.abs(time_diff) > time_threshold

                # 更新质量标志（排除第一个时间点）
                if rapid_changes.sizes["time"] > 0:
                    # 为了匹配原始数据的时间维度，需要处理差分后的维度变化
                    rapid_changes_full = xr.concat([
                        xr.zeros_like(rapid_changes.isel(time=0)),
                        rapid_changes
                    ], dim="time")

                    qc_flags = qc_flags.where(~rapid_changes_full, QualityFlag.SUSPECT.value)

                    temporal_failures = int(rapid_changes.sum())
                    if temporal_failures > 0:
                        self.logger.debug(f"检测到 {temporal_failures} 个时间连续性异常")

        except Exception as e:
            self.logger.warning(f"时间连续性检查失败: {e}")

        return qc_flags

    def _mark_missing_values(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray
    ) -> xr.DataArray:
        """标记缺失值"""
        # 标记NaN值
        missing_mask = np.isnan(data_var)
        qc_flags = qc_flags.where(~missing_mask, QualityFlag.MISSING.value)

        missing_count = int(missing_mask.sum())
        if missing_count > 0:
            self.stats["missing_data"] += missing_count
            self.logger.debug(f"标记了 {missing_count} 个缺失值")

        return qc_flags

    def _repair_data(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """数据修复和插值"""
        repaired_data = data_var.copy()
        updated_flags = qc_flags.copy()

        try:
            # 获取需要修复的数据掩码
            need_repair = (qc_flags == QualityFlag.BAD.value) | (qc_flags == QualityFlag.MISSING.value)

            if need_repair.sum() == 0:
                return repaired_data, updated_flags

            # 空间插值修复
            if "lat" in data_var.dims and "lon" in data_var.dims:
                repaired_data, updated_flags = self._spatial_interpolation_repair(
                    repaired_data, updated_flags, need_repair
                )

            # 时间插值修复
            if "time" in data_var.dims:
                repaired_data, updated_flags = self._temporal_interpolation_repair(
                    repaired_data, updated_flags, need_repair
                )

            # 使用中值滤波进行平滑
            if self.config.get("enable_median_filter", True):
                repaired_data = self._apply_median_filter(repaired_data, need_repair)

        except Exception as e:
            self.logger.warning(f"数据修复失败: {e}")

        return repaired_data, updated_flags

    def _spatial_interpolation_repair(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray,
            need_repair: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """空间插值修复"""
        repaired_data = data_var.copy()
        updated_flags = qc_flags.copy()

        try:
            # 对每个时间步分别进行空间插值
            if "time" in data_var.dims:
                for t in range(data_var.sizes["time"]):
                    time_slice = data_var.isel(time=t)
                    repair_mask = need_repair.isel(time=t)

                    if repair_mask.sum() > 0:
                        interpolated = time_slice.interpolate_na(
                            dim=["lat", "lon"],
                            method="linear",
                            fill_value="extrapolate" if self.config.get("enable_extrapolation", False) else None
                        )

                        # 更新修复的数据点
                        repaired_slice = time_slice.where(~repair_mask, interpolated)
                        repaired_data[dict(time=t)] = repaired_slice

                        # 标记插值点
                        interpolated_mask = repair_mask & ~np.isnan(interpolated)
                        updated_flags = updated_flags.where(
                            ~interpolated_mask.isel(time=t) if "time" in updated_flags.dims
                            else ~interpolated_mask,
                            QualityFlag.INTERPOLATED.value
                        )
            else:
                # 对整个数据进行空间插值
                interpolated = data_var.interpolate_na(
                    dim=["lat", "lon"],
                    method="linear",
                    fill_value="extrapolate" if self.config.get("enable_extrapolation", False) else None
                )

                repaired_data = data_var.where(~need_repair, interpolated)

                # 标记插值点
                interpolated_mask = need_repair & ~np.isnan(interpolated)
                updated_flags = updated_flags.where(~interpolated_mask, QualityFlag.INTERPOLATED.value)

        except Exception as e:
            self.logger.warning(f"空间插值修复失败: {e}")

        return repaired_data, updated_flags

    def _temporal_interpolation_repair(
            self,
            data_var: xr.DataArray,
            qc_flags: xr.DataArray,
            need_repair: xr.DataArray
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """时间插值修复"""
        repaired_data = data_var.copy()
        updated_flags = qc_flags.copy()

        try:
            if "time" in data_var.dims and data_var.sizes["time"] > 2:
                # 时间方向插值
                interpolated = data_var.interpolate_na(
                    dim="time",
                    method="linear",
                    fill_value="extrapolate" if self.config.get("enable_extrapolation", False) else None
                )

                # 更新需要修复的点
                repaired_data = repaired_data.where(~need_repair, interpolated)

                # 标记插值点
                interpolated_mask = need_repair & ~np.isnan(interpolated)
                updated_flags = updated_flags.where(~interpolated_mask, QualityFlag.INTERPOLATED.value)

        except Exception as e:
            self.logger.warning(f"时间插值修复失败: {e}")

        return repaired_data, updated_flags

    def _apply_median_filter(
            self,
            data_var: xr.DataArray,
            need_repair: xr.DataArray
    ) -> xr.DataArray:
        """应用中值滤波"""
        try:
            if data_var.ndim >= 2:
                # 只对需要修复的区域应用滤波
                filtered_data = data_var.copy()

                # 转换为numpy数组进行滤波
                data_np = data_var.values
                repair_mask = need_repair.values

                # 应用中值滤波
                if data_np.ndim == 2:
                    filtered_np = median_filter(data_np, size=3)
                elif data_np.ndim == 3:
                    # 对每个时间层分别滤波
                    filtered_np = np.zeros_like(data_np)
                    for t in range(data_np.shape[0]):
                        filtered_np[t] = median_filter(data_np[t], size=3)

                # 只在需要修复的点应用滤波结果
                data_np = np.where(repair_mask, filtered_np, data_np)

                # 创建新的DataArray
                filtered_data = xr.DataArray(
                    data_np,
                    dims=data_var.dims,
                    coords=data_var.coords,
                    attrs=data_var.attrs
                )

                return filtered_data

        except Exception as e:
            self.logger.warning(f"中值滤波失败: {e}")

        return data_var

    def _update_global_attributes(self, dataset: xr.Dataset):
        """更新全局属性"""
        if 'history' not in dataset.attrs:
            dataset.attrs['history'] = ""

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dataset.attrs['history'] += f"\n{timestamp}: Quality control applied using OceanSimulation QualityController"
        dataset.attrs['quality_control'] = "Applied physical limits, outlier detection, gradient checks, and data interpolation"

    def _update_statistics(self, dataset: xr.Dataset):
        """更新统计信息"""
        total_points = 0

        for var_name in dataset.data_vars:
            if var_name.endswith('_qc'):
                qc_data = dataset[var_name]
                total_points += qc_data.size

                # 统计各种质量标志的数量
                for flag in QualityFlag:
                    count = int((qc_data == flag.value).sum())
                    if flag == QualityFlag.GOOD:
                        self.stats["good_data"] += count
                    elif flag == QualityFlag.SUSPECT:
                        self.stats["suspect_data"] += count
                    elif flag == QualityFlag.BAD:
                        self.stats["bad_data"] += count
                    elif flag == QualityFlag.MISSING:
                        self.stats["missing_data"] += count
                    elif flag == QualityFlag.INTERPOLATED:
                        self.stats["interpolated_data"] += count

        self.stats["total_processed"] += total_points

    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量控制报告"""
        total = self.stats["total_processed"]

        if total == 0:
            return {"message": "No data processed yet"}

        return {
            "总处理数据点": total,
            "良好数据": {
                "数量": self.stats["good_data"],
                "百分比": f"{100 * self.stats['good_data'] / total:.2f}%"
            },
            "可疑数据": {
                "数量": self.stats["suspect_data"],
                "百分比": f"{100 * self.stats['suspect_data'] / total:.2f}%"
            },
            "坏数据": {
                "数量": self.stats["bad_data"],
                "百分比": f"{100 * self.stats['bad_data'] / total:.2f}%"
            },
            "缺失数据": {
                "数量": self.stats["missing_data"],
                "百分比": f"{100 * self.stats['missing_data'] / total:.2f}%"
            },
            "插值数据": {
                "数量": self.stats["interpolated_data"],
                "百分比": f"{100 * self.stats['interpolated_data'] / total:.2f}%"
            },
            "检测统计": {
                "异常值检测": self.stats["outliers_detected"],
                "梯度异常": self.stats["gradient_failures"],
                "物理约束违反": self.stats["physical_violations"]
            }
        }

    def reset_statistics(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key] = 0
        self.logger.info("质量控制统计信息已重置")

    def set_physical_limits(self, var_type: str, min_val: float, max_val: float):
        """设置物理约束限制"""
        self.physical_limits[var_type] = (min_val, max_val)
        self.logger.info(f"更新物理约束: {var_type} = [{min_val}, {max_val}]")

    def cleanup(self):
        """清理资源"""
        self.logger.info("数据质量控制器已关闭")