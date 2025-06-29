"""
坐标系转换器模块
负责处理海洋数据分析中的各种坐标系转换
包括地理坐标、投影坐标、网格坐标等转换功能
"""

import numpy as np
import math
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

class CoordinateSystem(Enum):
    """支持的坐标系类型"""
    WGS84 = "WGS84"
    UTM = "UTM"
    MERCATOR = "MERCATOR"
    STEREOGRAPHIC = "STEREOGRAPHIC"
    LAMBERT_CONFORMAL = "LAMBERT_CONFORMAL"
    GRID_CARTESIAN = "GRID_CARTESIAN"
    SPHERICAL = "SPHERICAL"

@dataclass
class CoordinatePoint:
    """坐标点数据结构"""
    x: float
    y: float
    z: Optional[float] = None
    system: CoordinateSystem = CoordinateSystem.WGS84

    def __post_init__(self):
        """验证坐标值的有效性"""
        if self.system == CoordinateSystem.WGS84:
            if not (-180 <= self.x <= 180):
                warnings.warn(f"经度值 {self.x} 超出有效范围 [-180, 180]")
            if not (-90 <= self.y <= 90):
                warnings.warn(f"纬度值 {self.y} 超出有效范围 [-90, 90]")

@dataclass
class ProjectionParameters:
    """投影参数"""
    central_meridian: float = 0.0
    central_parallel: float = 0.0
    standard_parallel_1: Optional[float] = None
    standard_parallel_2: Optional[float] = None
    false_easting: float = 0.0
    false_northing: float = 0.0
    scale_factor: float = 1.0
    ellipsoid_a: float = 6378137.0  # WGS84 长半轴
    ellipsoid_b: float = 6356752.314245  # WGS84 短半轴

class CoordinateTransformer:
    """
    坐标系转换器
    支持多种坐标系统之间的相互转换
    """

    def __init__(self):
        """初始化坐标转换器"""
        self.earth_radius = 6371000.0  # 地球平均半径(米)
        self.wgs84_a = 6378137.0  # WGS84 长半轴
        self.wgs84_b = 6356752.314245  # WGS84 短半轴
        self.wgs84_f = (self.wgs84_a - self.wgs84_b) / self.wgs84_a  # 扁率
        self.wgs84_e2 = 2 * self.wgs84_f - self.wgs84_f ** 2  # 第一偏心率的平方

    def geographic_to_cartesian(self, lon: float, lat: float,
                                alt: float = 0.0) -> Tuple[float, float, float]:
        """
        地理坐标(经纬度)转换为地心直角坐标系
        
        Args:
            lon: 经度(度)
            lat: 纬度(度)
            alt: 高程(米)
            
        Returns:
            (x, y, z): 地心直角坐标(米)
        """
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)

        # 卯酉圈曲率半径
        N = self.wgs84_a / math.sqrt(1 - self.wgs84_e2 * math.sin(lat_rad) ** 2)

        x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
        y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
        z = (N * (1 - self.wgs84_e2) + alt) * math.sin(lat_rad)

        return x, y, z

    def cartesian_to_geographic(self, x: float, y: float,
                                z: float) -> Tuple[float, float, float]:
        """
        地心直角坐标系转换为地理坐标
        
        Args:
            x, y, z: 地心直角坐标(米)
            
        Returns:
            (lon, lat, alt): 经度(度), 纬度(度), 高程(米)
        """
        p = math.sqrt(x**2 + y**2)
        theta = math.atan2(z * self.wgs84_a, p * self.wgs84_b)

        lon = math.degrees(math.atan2(y, x))
        lat = math.degrees(math.atan2(
            z + self.wgs84_e2 * self.wgs84_b / (1 - self.wgs84_e2) * math.sin(theta)**3,
            p - self.wgs84_e2 * self.wgs84_a * math.cos(theta)**3
        ))

        lat_rad = math.radians(lat)
        N = self.wgs84_a / math.sqrt(1 - self.wgs84_e2 * math.sin(lat_rad) ** 2)
        alt = p / math.cos(lat_rad) - N

        return lon, lat, alt

    def mercator_projection(self, lon: float, lat: float,
                            params: Optional[ProjectionParameters] = None) -> Tuple[float, float]:
        """
        墨卡托投影
        
        Args:
            lon: 经度(度)
            lat: 纬度(度)
            params: 投影参数
            
        Returns:
            (x, y): 投影坐标(米)
        """
        if params is None:
            params = ProjectionParameters()

        lon_rad = math.radians(lon - params.central_meridian)
        lat_rad = math.radians(lat)

        x = self.wgs84_a * lon_rad + params.false_easting
        y = self.wgs84_a * math.log(math.tan(math.pi/4 + lat_rad/2)) + params.false_northing

        return x, y

    def inverse_mercator_projection(self, x: float, y: float,
                                    params: Optional[ProjectionParameters] = None) -> Tuple[float, float]:
        """
        墨卡托投影逆变换
        
        Args:
            x, y: 投影坐标(米)
            params: 投影参数
            
        Returns:
            (lon, lat): 经度(度), 纬度(度)
        """
        if params is None:
            params = ProjectionParameters()

        lon = math.degrees((x - params.false_easting) / self.wgs84_a) + params.central_meridian
        lat = math.degrees(2 * math.atan(math.exp((y - params.false_northing) / self.wgs84_a)) - math.pi/2)

        return lon, lat

    def stereographic_projection(self, lon: float, lat: float,
                                 params: Optional[ProjectionParameters] = None) -> Tuple[float, float]:
        """
        极地立体投影
        
        Args:
            lon: 经度(度)
            lat: 纬度(度)
            params: 投影参数
            
        Returns:
            (x, y): 投影坐标(米)
        """
        if params is None:
            params = ProjectionParameters(central_parallel=90.0)  # 北极投影

        lon_rad = math.radians(lon - params.central_meridian)
        lat_rad = math.radians(lat)
        lat0_rad = math.radians(params.central_parallel)

        if abs(params.central_parallel) == 90:  # 极地投影
            sign = 1 if params.central_parallel > 0 else -1
            k = 2 * self.wgs84_a / (1 + sign * math.sin(lat_rad))
            x = k * math.cos(lat_rad) * math.sin(lon_rad) + params.false_easting
            y = -sign * k * math.cos(lat_rad) * math.cos(lon_rad) + params.false_northing
        else:  # 一般立体投影
            c = 2 * self.wgs84_a / (1 + math.sin(lat0_rad) * math.sin(lat_rad) +
                                    math.cos(lat0_rad) * math.cos(lat_rad) * math.cos(lon_rad))
            x = c * math.cos(lat_rad) * math.sin(lon_rad) + params.false_easting
            y = c * (math.cos(lat0_rad) * math.sin(lat_rad) -
                     math.sin(lat0_rad) * math.cos(lat_rad) * math.cos(lon_rad)) + params.false_northing

        return x, y

    def utm_projection(self, lon: float, lat: float) -> Tuple[float, float, int, str]:
        """
        UTM投影
        
        Args:
            lon: 经度(度)
            lat: 纬度(度)
            
        Returns:
            (x, y, zone, hemisphere): UTM坐标(米), 带号, 半球('N'或'S')
        """
        zone = int((lon + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'

        lon0 = (zone - 1) * 6 - 180 + 3  # 中央经线

        params = ProjectionParameters(
            central_meridian=lon0,
            scale_factor=0.9996,
            false_easting=500000.0,
            false_northing=0.0 if hemisphere == 'N' else 10000000.0
        )

        # 使用横轴墨卡托投影
        x, y = self.transverse_mercator_projection(lon, lat, params)

        return x, y, zone, hemisphere

    def transverse_mercator_projection(self, lon: float, lat: float,
                                       params: ProjectionParameters) -> Tuple[float, float]:
        """
        横轴墨卡托投影
        
        Args:
            lon: 经度(度)
            lat: 纬度(度)
            params: 投影参数
            
        Returns:
            (x, y): 投影坐标(米)
        """
        lon_rad = math.radians(lon - params.central_meridian)
        lat_rad = math.radians(lat)

        N = self.wgs84_a / math.sqrt(1 - self.wgs84_e2 * math.sin(lat_rad) ** 2)
        T = math.tan(lat_rad) ** 2
        C = self.wgs84_e2 * math.cos(lat_rad) ** 2 / (1 - self.wgs84_e2)
        A = math.cos(lat_rad) * lon_rad

        M = self.meridional_arc_length(lat_rad)

        x = params.scale_factor * N * (A + (1 - T + C) * A**3 / 6 +
                                       (5 - 18*T + T**2 + 72*C - 58*self.wgs84_e2) * A**5 / 120) + params.false_easting

        y = params.scale_factor * (M + N * math.tan(lat_rad) * (A**2 / 2 +
                                                                (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                                                                (61 - 58*T + T**2 + 600*C - 330*self.wgs84_e2) * A**6 / 720)) + params.false_northing

        return x, y

    def meridional_arc_length(self, lat_rad: float) -> float:
        """
        计算子午线弧长
        
        Args:
            lat_rad: 纬度(弧度)
            
        Returns:
            子午线弧长(米)
        """
        e2 = self.wgs84_e2
        e4 = e2 ** 2
        e6 = e2 ** 3

        A0 = 1 - e2/4 - 3*e4/64 - 5*e6/256
        A2 = 3*(e2 + e4/4 + 15*e6/128) / 8
        A4 = 15*(e4 + 3*e6/4) / 256
        A6 = 35*e6 / 3072

        M = self.wgs84_a * (A0 * lat_rad - A2 * math.sin(2*lat_rad) +
                            A4 * math.sin(4*lat_rad) - A6 * math.sin(6*lat_rad))

        return M

    def grid_to_geographic(self, grid_x: int, grid_y: int,
                           origin_lon: float, origin_lat: float,
                           grid_spacing_x: float, grid_spacing_y: float,
                           rotation_angle: float = 0.0) -> Tuple[float, float]:
        """
        网格坐标转换为地理坐标
        
        Args:
            grid_x, grid_y: 网格坐标
            origin_lon, origin_lat: 网格原点地理坐标(度)
            grid_spacing_x, grid_spacing_y: 网格间距(米)
            rotation_angle: 网格旋转角度(度)
            
        Returns:
            (lon, lat): 地理坐标(度)
        """
        # 网格坐标转换为局部笛卡尔坐标
        local_x = grid_x * grid_spacing_x
        local_y = grid_y * grid_spacing_y

        # 应用旋转
        if rotation_angle != 0:
            angle_rad = math.radians(rotation_angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            rotated_x = local_x * cos_angle - local_y * sin_angle
            rotated_y = local_x * sin_angle + local_y * cos_angle

            local_x, local_y = rotated_x, rotated_y

        # 局部坐标转换为地理坐标(简化方法)
        lat_offset = local_y / (111320.0)  # 1度纬度约111320米
        lon_offset = local_x / (111320.0 * math.cos(math.radians(origin_lat)))

        lon = origin_lon + lon_offset
        lat = origin_lat + lat_offset

        return lon, lat

    def geographic_to_grid(self, lon: float, lat: float,
                           origin_lon: float, origin_lat: float,
                           grid_spacing_x: float, grid_spacing_y: float,
                           rotation_angle: float = 0.0) -> Tuple[int, int]:
        """
        地理坐标转换为网格坐标
        
        Args:
            lon, lat: 地理坐标(度)
            origin_lon, origin_lat: 网格原点地理坐标(度)
            grid_spacing_x, grid_spacing_y: 网格间距(米)
            rotation_angle: 网格旋转角度(度)
            
        Returns:
            (grid_x, grid_y): 网格坐标
        """
        # 地理坐标差值转换为局部坐标
        lon_offset = lon - origin_lon
        lat_offset = lat - origin_lat

        local_x = lon_offset * 111320.0 * math.cos(math.radians(origin_lat))
        local_y = lat_offset * 111320.0

        # 应用反向旋转
        if rotation_angle != 0:
            angle_rad = math.radians(-rotation_angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            rotated_x = local_x * cos_angle - local_y * sin_angle
            rotated_y = local_x * sin_angle + local_y * cos_angle

            local_x, local_y = rotated_x, rotated_y

        # 局部坐标转换为网格坐标
        grid_x = int(round(local_x / grid_spacing_x))
        grid_y = int(round(local_y / grid_spacing_y))

        return grid_x, grid_y

    def distance_on_sphere(self, lon1: float, lat1: float,
                           lon2: float, lat2: float) -> float:
        """
        计算球面上两点间的大圆距离(Haversine公式)
        
        Args:
            lon1, lat1: 第一个点的经纬度(度)
            lon2, lat2: 第二个点的经纬度(度)
            
        Returns:
            距离(米)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return self.earth_radius * c

    def bearing_on_sphere(self, lon1: float, lat1: float,
                          lon2: float, lat2: float) -> float:
        """
        计算从第一个点到第二个点的方位角
        
        Args:
            lon1, lat1: 第一个点的经纬度(度)
            lon2, lat2: 第二个点的经纬度(度)
            
        Returns:
            方位角(度, 从北向东为正)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)

        return (bearing_deg + 360) % 360

    def transform_coordinates(self, points: Union[CoordinatePoint, List[CoordinatePoint]],
                              target_system: CoordinateSystem,
                              projection_params: Optional[ProjectionParameters] = None) -> Union[CoordinatePoint, List[CoordinatePoint]]:
        """
        通用坐标转换接口
        
        Args:
            points: 单个坐标点或坐标点列表
            target_system: 目标坐标系
            projection_params: 投影参数(如需要)
            
        Returns:
            转换后的坐标点或坐标点列表
        """
        is_single_point = isinstance(points, CoordinatePoint)
        if is_single_point:
            points = [points]

        result_points = []

        for point in points:
            if point.system == target_system:
                result_points.append(point)
                continue

            # 实现具体的转换逻辑
            if point.system == CoordinateSystem.WGS84 and target_system == CoordinateSystem.MERCATOR:
                x, y = self.mercator_projection(point.x, point.y, projection_params)
                result_points.append(CoordinatePoint(x, y, point.z, target_system))

            elif point.system == CoordinateSystem.MERCATOR and target_system == CoordinateSystem.WGS84:
                lon, lat = self.inverse_mercator_projection(point.x, point.y, projection_params)
                result_points.append(CoordinatePoint(lon, lat, point.z, target_system))

            elif point.system == CoordinateSystem.WGS84 and target_system == CoordinateSystem.UTM:
                x, y, zone, hemisphere = self.utm_projection(point.x, point.y)
                result_points.append(CoordinatePoint(x, y, point.z, target_system))

            else:
                raise NotImplementedError(f"从 {point.system} 到 {target_system} 的转换尚未实现")

        return result_points[0] if is_single_point else result_points

    def validate_coordinates(self, point: CoordinatePoint) -> bool:
        """
        验证坐标点的有效性
        
        Args:
            point: 坐标点
            
        Returns:
            是否有效
        """
        if point.system == CoordinateSystem.WGS84:
            return (-180 <= point.x <= 180) and (-90 <= point.y <= 90)

        # 其他坐标系的验证规则可以在这里添加
        return True

    def get_grid_bounds(self, points: List[CoordinatePoint]) -> Tuple[float, float, float, float]:
        """
        获取坐标点集合的边界框
        
        Args:
            points: 坐标点列表
            
        Returns:
            (min_x, min_y, max_x, max_y): 边界框
        """
        if not points:
            raise ValueError("坐标点列表不能为空")

        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]

        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


class CoordinateTransformManager:
    """
    坐标转换管理器
    提供高级的坐标转换和管理功能
    """

    def __init__(self):
        """初始化转换管理器"""
        self.transformer = CoordinateTransformer()
        self.cached_transforms = {}

    def create_transformation_chain(self, source_system: CoordinateSystem,
                                    target_system: CoordinateSystem,
                                    intermediate_systems: Optional[List[CoordinateSystem]] = None) -> List[CoordinateSystem]:
        """
        创建坐标转换链
        
        Args:
            source_system: 源坐标系
            target_system: 目标坐标系
            intermediate_systems: 中间坐标系列表
            
        Returns:
            转换链
        """
        if intermediate_systems is None:
            return [source_system, target_system]

        return [source_system] + intermediate_systems + [target_system]

    def batch_transform(self, points: List[CoordinatePoint],
                        target_system: CoordinateSystem,
                        projection_params: Optional[ProjectionParameters] = None,
                        chunk_size: int = 1000) -> List[CoordinatePoint]:
        """
        批量坐标转换
        
        Args:
            points: 坐标点列表
            target_system: 目标坐标系
            projection_params: 投影参数
            chunk_size: 批处理大小
            
        Returns:
            转换后的坐标点列表
        """
        result = []

        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]
            transformed_chunk = self.transformer.transform_coordinates(
                chunk, target_system, projection_params
            )
            result.extend(transformed_chunk)

        return result

    def calculate_transformation_accuracy(self, source_points: List[CoordinatePoint],
                                          target_points: List[CoordinatePoint]) -> float:
        """
        计算坐标转换的精度
        
        Args:
            source_points: 源坐标点
            target_points: 目标坐标点
            
        Returns:
            平均误差(米)
        """
        if len(source_points) != len(target_points):
            raise ValueError("源坐标点和目标坐标点数量不匹配")

        total_error = 0.0
        valid_points = 0

        for src, tgt in zip(source_points, target_points):
            if src.system == CoordinateSystem.WGS84 and tgt.system == CoordinateSystem.WGS84:
                error = self.transformer.distance_on_sphere(src.x, src.y, tgt.x, tgt.y)
                total_error += error
                valid_points += 1

        return total_error / valid_points if valid_points > 0 else 0.0


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建坐标转换器
    transformer = CoordinateTransformer()

    # 示例1: 地理坐标转换为墨卡托投影
    lon, lat = 5.0, 60.0  # 挪威海域
    x, y = transformer.mercator_projection(lon, lat)
    print(f"地理坐标 ({lon}, {lat}) -> 墨卡托投影 ({x:.2f}, {y:.2f})")

    # 示例2: UTM投影
    x_utm, y_utm, zone, hemisphere = transformer.utm_projection(lon, lat)
    print(f"UTM投影: ({x_utm:.2f}, {y_utm:.2f}), 带号: {zone}, 半球: {hemisphere}")

    # 示例3: 计算两点间距离
    lon1, lat1 = 5.0, 60.0
    lon2, lat2 = 6.0, 61.0
    distance = transformer.distance_on_sphere(lon1, lat1, lon2, lat2)
    print(f"两点间距离: {distance:.2f} 米")

    # 示例4: 网格坐标转换
    grid_x, grid_y = transformer.geographic_to_grid(
        lon, lat, origin_lon=0.0, origin_lat=55.0,
        grid_spacing_x=1000.0, grid_spacing_y=1000.0
    )
    print(f"网格坐标: ({grid_x}, {grid_y})")

    # 示例5: 使用坐标点对象
    point = CoordinatePoint(lon, lat, system=CoordinateSystem.WGS84)
    print(f"坐标点: {point}")

    # 示例6: 坐标转换管理器
    manager = CoordinateTransformManager()
    points = [CoordinatePoint(5.0, 60.0), CoordinatePoint(6.0, 61.0)]

    # 批量转换
    transformed_points = manager.batch_transform(
        points, CoordinateSystem.MERCATOR
    )
    print(f"批量转换完成: {len(transformed_points)} 个点")

    print("坐标转换器模块测试完成!")