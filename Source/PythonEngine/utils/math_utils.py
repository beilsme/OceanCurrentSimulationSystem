#!/usr/bin/env python3
"""
数学工具函数库
提供海洋数据处理中常用的数学计算功能
"""

import numpy as np
from scipy import interpolate, ndimage, signal
from scipy.spatial.distance import cdist
from typing import Tuple, Union, Optional, List, Any
import warnings

def interpolate_2d(
        data: np.ndarray,
        source_coords: Tuple[np.ndarray, np.ndarray],
        target_coords: Tuple[np.ndarray, np.ndarray],
        method: str = "linear",
        fill_value: float = np.nan
) -> np.ndarray:
    """
    2D数据插值
    
    Args:
        data: 源数据 (ny, nx)
        source_coords: 源坐标 (x_array, y_array)
        target_coords: 目标坐标 (target_x, target_y)
        method: 插值方法
        fill_value: 填充值
        
    Returns:
        插值后的数据
    """
    source_x, source_y = source_coords
    target_x, target_y = target_coords

    # 创建插值器
    if method == "linear":
        interpolator = interpolate.RegularGridInterpolator(
            (source_y, source_x), data,
            method='linear', bounds_error=False, fill_value=fill_value
        )
    elif method == "cubic":
        interpolator = interpolate.RegularGridInterpolator(
            (source_y, source_x), data,
            method='cubic', bounds_error=False, fill_value=fill_value
        )
    elif method == "nearest":
        interpolator = interpolate.RegularGridInterpolator(
            (source_y, source_x), data,
            method='nearest', bounds_error=False, fill_value=fill_value
        )
    else:
        raise ValueError(f"不支持的插值方法: {method}")

    # 创建目标网格
    if isinstance(target_x, np.ndarray) and isinstance(target_y, np.ndarray):
        if target_x.ndim == 1 and target_y.ndim == 1:
            target_xx, target_yy = np.meshgrid(target_x, target_y)
        else:
            target_xx, target_yy = target_x, target_y

        # 准备插值点
        points = np.column_stack([target_yy.ravel(), target_xx.ravel()])
        result = interpolator(points).reshape(target_yy.shape)
    else:
        result = interpolator((target_y, target_x))

    return result

def calculate_gradients(
        data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D数据的梯度
    
    Args:
        data: 数据数组 (ny, nx)
        x_coords: X坐标数组
        y_coords: Y坐标数组
        
    Returns:
        (gradient_x, gradient_y) 梯度分量
    """
    # 计算坐标间距
    dx = np.mean(np.diff(x_coords))
    dy = np.mean(np.diff(y_coords))

    # 计算梯度
    grad_y, grad_x = np.gradient(data, dy, dx)

    return grad_x, grad_y

def calculate_divergence(
        u: np.ndarray,
        v: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
) -> np.ndarray:
    """
    计算矢量场的散度
    
    Args:
        u: X方向分量
        v: Y方向分量
        x_coords: X坐标
        y_coords: Y坐标
        
    Returns:
        散度场
    """
    # 计算坐标间距
    dx = np.mean(np.diff(x_coords))
    dy = np.mean(np.diff(y_coords))

    # 计算偏导数
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # 散度 = du/dx + dv/dy
    divergence = du_dx + dv_dy

    return divergence

def calculate_vorticity(
        u: np.ndarray,
        v: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
) -> np.ndarray:
    """
    计算矢量场的涡度
    
    Args:
        u: X方向分量
        v: Y方向分量
        x_coords: X坐标
        y_coords: Y坐标
        
    Returns:
        涡度场
    """
    # 计算坐标间距
    dx = np.mean(np.diff(x_coords))
    dy = np.mean(np.diff(y_coords))

    # 计算偏导数
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)

    # 涡度 = dv/dx - du/dy
    vorticity = dv_dx - du_dy

    return vorticity

def calculate_strain_rate(
        u: np.ndarray,
        v: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算应变率张量
    
    Args:
        u: X方向分量
        v: Y方向分量
        x_coords: X坐标
        y_coords: Y坐标
        
    Returns:
        (strain_11, strain_12, strain_22) 应变率张量分量
    """
    # 计算坐标间距
    dx = np.mean(np.diff(x_coords))
    dy = np.mean(np.diff(y_coords))

    # 计算偏导数
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # 应变率张量分量
    strain_11 = du_dx
    strain_12 = 0.5 * (du_dy + dv_dx)
    strain_22 = dv_dy

    return strain_11, strain_12, strain_22

def calculate_okubo_weiss(
        u: np.ndarray,
        v: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
) -> np.ndarray:
    """
    计算Okubo-Weiss参数
    
    Args:
        u: X方向分量
        v: Y方向分量
        x_coords: X坐标
        y_coords: Y坐标
        
    Returns:
        Okubo-Weiss参数
    """
    # 计算应变率和涡度
    strain_11, strain_12, strain_22 = calculate_strain_rate(u, v, x_coords, y_coords)
    vorticity = calculate_vorticity(u, v, x_coords, y_coords)

    # Okubo-Weiss参数 = S^2 - ω^2
    # S^2 = (∂u/∂x - ∂v/∂y)^2 + (∂u/∂y + ∂v/∂x)^2
    # ω^2 = (∂v/∂x - ∂u/∂y)^2

    strain_normal = strain_11 - strain_22
    strain_shear = 2 * strain_12

    okubo_weiss = strain_normal**2 + strain_shear**2 - vorticity**2

    return okubo_weiss

def apply_smoothing_filter(
        data: np.ndarray,
        filter_type: str = "gaussian",
        sigma: float = 1.0,
        kernel_size: int = 3
) -> np.ndarray:
    """
    应用平滑滤波
    
    Args:
        data: 输入数据
        filter_type: 滤波类型 ("gaussian", "median", "uniform")
        sigma: 高斯滤波的标准差
        kernel_size: 滤波核大小
        
    Returns:
        滤波后的数据
    """
    if filter_type == "gaussian":
        return ndimage.gaussian_filter(data, sigma=sigma)
    elif filter_type == "median":
        return ndimage.median_filter(data, size=kernel_size)
    elif filter_type == "uniform":
        return ndimage.uniform_filter(data, size=kernel_size)
    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}")

def detect_features(
        data: np.ndarray,
        feature_type: str = "peaks",
        threshold: Optional[float] = None,
        min_distance: int = 1
) -> List[Tuple[int, int]]:
    """
    检测数据中的特征点
    
    Args:
        data: 输入数据
        feature_type: 特征类型 ("peaks", "valleys", "edges")
        threshold: 阈值
        min_distance: 最小距离
        
    Returns:
        特征点坐标列表
    """
    if feature_type == "peaks":
        # 检测峰值
        if threshold is None:
            threshold = np.mean(data) + np.std(data)

        peaks = signal.find_peaks_cwt(data.ravel(), np.arange(1, min_distance+1))
        # 转换为2D坐标
        coords = []
        for peak in peaks:
            y, x = np.unravel_index(peak, data.shape)
            if data[y, x] > threshold:
                coords.append((y, x))
        return coords

    elif feature_type == "valleys":
        # 检测谷值（负峰值）
        return detect_features(-data, "peaks", -threshold if threshold else None, min_distance)

    elif feature_type == "edges":
        # 检测边缘
        from scipy import ndimage
        edges = ndimage.sobel(data)
        if threshold is None:
            threshold = np.mean(edges) + np.std(edges)

        y_coords, x_coords = np.where(edges > threshold)
        return list(zip(y_coords, x_coords))

    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")

def calculate_empirical_orthogonal_functions(
        data: np.ndarray,
        n_modes: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算经验正交函数(EOF)
    
    Args:
        data: 时间序列数据 (time, space...)
        n_modes: 模态数量
        
    Returns:
        (eigenvalues, eigenvectors, time_coefficients) EOF分析结果
    """
    # 重塑数据为 (time, space)
    original_shape = data.shape
    if data.ndim > 2:
        data_reshaped = data.reshape(original_shape[0], -1)
    else:
        data_reshaped = data

    # 移除时间平均
    data_anomaly = data_reshaped - np.mean(data_reshaped, axis=0)

    # 计算协方差矩阵
    cov_matrix = np.cov(data_anomaly.T)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 按特征值大小排序
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 选择前n_modes个模态
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    # 计算时间系数
    time_coefficients = np.dot(data_anomaly, eigenvectors)

    # 重塑特征向量回原始空间形状
    if data.ndim > 2:
        spatial_shape = original_shape[1:]
        eigenvectors = eigenvectors.reshape(-1, n_modes, *spatial_shape)

    return eigenvalues, eigenvectors, time_coefficients

def calculate_spectral_analysis(
        time_series: np.ndarray,
        dt: float = 1.0,
        method: str = "periodogram"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算时间序列的频谱分析
    
    Args:
        time_series: 时间序列数据
        dt: 时间间隔
        method: 分析方法 ("periodogram", "welch", "multitaper")
        
    Returns:
        (frequencies, power_spectrum) 频率和功率谱
    """
    if method == "periodogram":
        frequencies, power_spectrum = signal.periodogram(
            time_series, fs=1.0/dt
        )
    elif method == "welch":
        frequencies, power_spectrum = signal.welch(
            time_series, fs=1.0/dt, nperseg=len(time_series)//4
        )
    elif method == "multitaper":
        # 需要安装spectrum包
        try:
            from spectrum import pmtm
            power_spectrum, frequencies = pmtm(time_series, NW=4, k=7)
            frequencies = frequencies / dt
        except ImportError:
            warnings.warn("spectrum包不可用，使用welch方法替代")
            frequencies, power_spectrum = signal.welch(
                time_series, fs=1.0/dt, nperseg=len(time_series)//4
            )
    else:
        raise ValueError(f"不支持的频谱分析方法: {method}")

    return frequencies, power_spectrum

def calculate_cross_correlation(
        x: np.ndarray,
        y: np.ndarray,
        mode: str = "full",
        normalize: bool = True
) -> np.ndarray:
    """
    计算两个序列的互相关
    
    Args:
        x: 第一个序列
        y: 第二个序列
        mode: 模式 ("full", "valid", "same")
        normalize: 是否归一化
        
    Returns:
        互相关结果
    """
    correlation = signal.correlate(x, y, mode=mode)

    if normalize:
        # 归一化
        norm = np.sqrt(np.sum(x**2) * np.sum(y**2))
        if norm > 0:
            correlation = correlation / norm

    return correlation

def fit_polynomial_surface(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        degree: int = 2
) -> Tuple[np.ndarray, callable]:
    """
    拟合多项式曲面
    
    Args:
        x: X坐标数组
        y: Y坐标数组
        z: Z值数组
        degree: 多项式次数
        
    Returns:
        (coefficients, surface_function) 系数和曲面函数
    """
    # 创建设计矩阵
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append(x**i * y**j)

    design_matrix = np.column_stack(terms)

    # 最小二乘拟合
    coefficients, residuals, rank, s = np.linalg.lstsq(
        design_matrix, z, rcond=None
    )

    def surface_function(xi, yi):
        """拟合的曲面函数"""
        result = np.zeros_like(xi)
        idx = 0
        for i in range(degree + 1):
            for j in range(degree + 1 - i):
                result += coefficients[idx] * xi**i * yi**j
                idx += 1
        return result

    return coefficients, surface_function

def calculate_distance_matrix(
        points1: np.ndarray,
        points2: Optional[np.ndarray] = None,
        metric: str = "euclidean"
) -> np.ndarray:
    """
    计算点集间的距离矩阵
    
    Args:
        points1: 第一个点集 (n, 2)
        points2: 第二个点集 (m, 2)，如果None则计算points1内部距离
        metric: 距离度量
        
    Returns:
        距离矩阵
    """
    if points2 is None:
        return cdist(points1, points1, metric=metric)
    else:
        return cdist(points1, points2, metric=metric)

def interpolate_gaps(
        data: np.ndarray,
        method: str = "linear",
        max_gap: Optional[int] = None
) -> np.ndarray:
    """
    插值填补数据缺口
    
    Args:
        data: 包含NaN的数据数组
        method: 插值方法
        max_gap: 最大可插值的缺口长度
        
    Returns:
        插值后的数据
    """
    result = data.copy()

    if data.ndim == 1:
        # 1D数据插值
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) < 2:
            return result

        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]

        if method == "linear":
            interpolator = interpolate.interp1d(
                valid_indices, valid_values, kind='linear',
                bounds_error=False, fill_value=np.nan
            )
        elif method == "cubic":
            if len(valid_values) >= 4:
                interpolator = interpolate.interp1d(
                    valid_indices, valid_values, kind='cubic',
                    bounds_error=False, fill_value=np.nan
                )
            else:
                # 数据点太少，使用线性插值
                interpolator = interpolate.interp1d(
                    valid_indices, valid_values, kind='linear',
                    bounds_error=False, fill_value=np.nan
                )
        else:
            raise ValueError(f"不支持的插值方法: {method}")

        # 识别缺口
        nan_mask = np.isnan(data)
        if max_gap is not None:
            # 检查缺口长度
            gap_starts = []
            gap_ends = []
            in_gap = False

            for i, is_nan in enumerate(nan_mask):
                if is_nan and not in_gap:
                    gap_starts.append(i)
                    in_gap = True
                elif not is_nan and in_gap:
                    gap_ends.append(i - 1)
                    in_gap = False

            if in_gap:
                gap_ends.append(len(data) - 1)

            # 只插值小于max_gap的缺口
            for start, end in zip(gap_starts, gap_ends):
                if end - start + 1 <= max_gap:
                    indices = np.arange(start, end + 1)
                    result[indices] = interpolator(indices)
        else:
            # 插值所有缺口
            all_indices = np.arange(len(data))
            result = interpolator(all_indices)
            # 保留原始有效数据
            result[valid_mask] = data[valid_mask]

    else:
        # 多维数据，沿第一个轴插值
        for idx in np.ndindex(data.shape[1:]):
            slice_data = data[(slice(None),) + idx]
            result[(slice(None),) + idx] = interpolate_gaps(
                slice_data, method, max_gap
            )

    return result

def calculate_running_statistics(
        data: np.ndarray,
        window_size: int,
        statistic: str = "mean"
) -> np.ndarray:
    """
    计算滑动窗口统计量
    
    Args:
        data: 输入数据
        window_size: 窗口大小
        statistic: 统计量类型 ("mean", "std", "var", "min", "max", "median")
        
    Returns:
        滑动统计量
    """
    if window_size <= 0 or window_size > len(data):
        raise ValueError("窗口大小必须在(0, len(data)]范围内")

    result = np.full(len(data), np.nan)

    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        window_data = data[start_idx:end_idx]

        # 移除NaN值
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) > 0:
            if statistic == "mean":
                result[i] = np.mean(valid_data)
            elif statistic == "std":
                result[i] = np.std(valid_data)
            elif statistic == "var":
                result[i] = np.var(valid_data)
            elif statistic == "min":
                result[i] = np.min(valid_data)
            elif statistic == "max":
                result[i] = np.max(valid_data)
            elif statistic == "median":
                result[i] = np.median(valid_data)
            else:
                raise ValueError(f"不支持的统计量: {statistic}")

    return result

def resample_data(
        data: np.ndarray,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        method: str = "linear"
) -> np.ndarray:
    """
    重采样1D数据到新的坐标
    
    Args:
        data: 源数据
        source_coords: 源坐标
        target_coords: 目标坐标
        method: 重采样方法
        
    Returns:
        重采样后的数据
    """
    if method == "linear":
        interpolator = interpolate.interp1d(
            source_coords, data, kind='linear',
            bounds_error=False, fill_value=np.nan
        )
    elif method == "cubic":
        interpolator = interpolate.interp1d(
            source_coords, data, kind='cubic',
            bounds_error=False, fill_value=np.nan
        )
    elif method == "nearest":
        interpolator = interpolate.interp1d(
            source_coords, data, kind='nearest',
            bounds_error=False, fill_value=np.nan
        )
    else:
        raise ValueError(f"不支持的重采样方法: {method}")

    return interpolator(target_coords)

def calculate_anomalies(
        data: np.ndarray,
        reference_data: Optional[np.ndarray] = None,
        method: str = "subtract"
) -> np.ndarray:
    """
    计算距平值
    
    Args:
        data: 输入数据
        reference_data: 参考数据（如果None则使用data的均值）
        method: 计算方法 ("subtract", "divide", "zscore")
        
    Returns:
        距平值
    """
    if reference_data is None:
        reference = np.nanmean(data, axis=0)
    else:
        reference = reference_data

    if method == "subtract":
        return data - reference
    elif method == "divide":
        return data / reference
    elif method == "zscore":
        std = np.nanstd(data, axis=0)
        return (data - reference) / std
    else:
        raise ValueError(f"不支持的距平计算方法: {method}")

def apply_quality_mask(
        data: np.ndarray,
        quality_flags: np.ndarray,
        good_flags: List[int] = [0]
) -> np.ndarray:
    """
    根据质量标志应用掩码
    
    Args:
        data: 输入数据
        quality_flags: 质量标志数组
        good_flags: 好数据的标志值列表
        
    Returns:
        应用掩码后的数据
    """
    mask = np.isin(quality_flags, good_flags)
    return np.where(mask, data, np.nan)

def calculate_climatology(
        data: np.ndarray,
        time_axis: int = 0,
        cycle_length: int = 365
) -> np.ndarray:
    """
    计算气候态
    
    Args:
        data: 时间序列数据
        time_axis: 时间轴
        cycle_length: 周期长度（天）
        
    Returns:
        气候态数据
    """
    if data.shape[time_axis] < cycle_length:
        warnings.warn("数据长度小于周期长度，无法计算完整气候态")
        return np.nanmean(data, axis=time_axis, keepdims=True)

    # 重塑数据为年循环
    n_years = data.shape[time_axis] // cycle_length
    remaining_days = data.shape[time_axis] % cycle_length

    if remaining_days > 0:
        # 截断到完整年份
        indices = [slice(None)] * data.ndim
        indices[time_axis] = slice(0, n_years * cycle_length)
        data_truncated = data[tuple(indices)]
    else:
        data_truncated = data

    # 重塑为 (年, 天数, 其他维度)
    new_shape = list(data_truncated.shape)
    new_shape[time_axis:time_axis+1] = [n_years, cycle_length]

    # 移动时间轴到第一位
    data_moved = np.moveaxis(data_truncated, time_axis, 0)
    data_reshaped = data_moved.reshape(n_years, cycle_length, -1)

    # 计算多年平均
    climatology = np.nanmean(data_reshaped, axis=0)

    # 恢复原始形状
    clim_shape = list(data.shape)
    clim_shape[time_axis] = cycle_length
    climatology = climatology.reshape(clim_shape)

    return climatology

def smooth_circular_data(
        data: np.ndarray,
        sigma: float = 1.0,
        axis: int = -1
) -> np.ndarray:
    """
    对循环数据应用平滑（如角度数据）
    
    Args:
        data: 循环数据（度）
        sigma: 平滑参数
        axis: 平滑轴
        
    Returns:
        平滑后的数据
    """
    # 转换为复数表示
    rad_data = np.radians(data)
    complex_data = np.exp(1j * rad_data)

    # 对实部和虚部分别平滑
    real_smooth = ndimage.gaussian_filter1d(
        complex_data.real, sigma=sigma, axis=axis
    )
    imag_smooth = ndimage.gaussian_filter1d(
        complex_data.imag, sigma=sigma, axis=axis
    )

    # 重新组合并转换回角度
    smooth_complex = real_smooth + 1j * imag_smooth
    smooth_angles = np.angle(smooth_complex)

    return np.degrees(smooth_angles)