# ==============================================================================
# 文件名称：core/data_processor.py
# 接口名称：DataProcessor
# 作者：beilsm
# 版本号：v1.2.1
# 最新更改时间：2025-06-29
# ==============================================================================
# ✅ 功能简介：
#   - 核心数据处理器，支持洋流场（uv）专业可视化与矢量导出
#   - 可作为所有后续模拟、分析、可视化的基础处理层
# ==============================================================================
# ✅ 更新说明（v1.2.1）：
#   - 优化海陆掩膜逻辑，严格保证底图填色不会覆盖陆地
#   - quiver 箭头与底图掩膜解耦，陆地灰色表现与 show.py 完全一致
#   - 保留专业可扩展接口，方便后续多深度/多时刻扩展
#   - 其余接口保持兼容，便于后续功能拓展
# ==============================================================================


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString, Point
import geopandas as gpd
from Source.PythonEngine.utils.chinese_config import setup_chinese_all
from typing import Optional
from datetime import datetime

class DataProcessor:
    """
    核心数据处理器：支持uv场可视化、导出等功能
    """
    def __init__(self, u, v, lat, lon, depth=None, time_info=None):
        self.u = u
        self.v = v
        self.lat = lat
        self.lon = lon
        self.depth = 0  # 默认深度为0，后续维护可以增加深度选择功能
        self.time_info = time_info

    def plot_vector_field(self,
                          skip: int = 3,
                          show: bool = True,
                          save_path: Optional[str] = None,
                          lon_min: Optional[float] = None,
                          lon_max: Optional[float] = None,
                          lat_min: Optional[float] = None,
                          lat_max: Optional[float] = None,
                          contourf_levels: int = 100,
                          contourf_cmap: str = 'coolwarm',
                          contourf_zorder: int = 1,
                          quiver_zorder: int = 2,
                          quiver_scale: float = 30,
                          quiver_width: float = 0.001,
                          font_size: int = 14,
                          dpi: int = 120):
            """
            严格复刻 show.py 的可视化，含底图、掩膜、箭头层、地图范围、标题等
            """
            # ====== 中文配置 ======
            setup_chinese_all(font_size=font_size, dpi=dpi, test=False)

          
            
            # 1. 经纬度范围（默认用全部，或传参控制）
            lon = self.lon
            lat = self.lat
            u = self.u
            v = self.v
            if lon_min is not None and lon_max is not None:
                lon_mask = (lon >= lon_min) & (lon <= lon_max)
                lon = lon[lon_mask]
                u = u[:, lon_mask]
                v = v[:, lon_mask]
            if lat_min is not None and lat_max is not None:
                lat_mask = (lat >= lat_min) & (lat <= lat_max)
                lat = lat[lat_mask]
                u = u[lat_mask, :]
                v = v[lat_mask, :]
    
            # 2. 计算流速、流向
            magnitude = u ** 2 + v ** 2
            magnitude = np.where(np.isnan(magnitude), 0, magnitude)
            speed = np.sqrt(magnitude)
            water_mask = np.logical_and(np.isfinite(u), np.isfinite(v))
            speed_plot = np.where(water_mask, speed, np.nan)
            direction = np.arctan2(v, u)
            direction_deg = np.degrees(direction)
    
            # 3. meshgrid
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            # 4. 掩膜有效数据
            valid_mask = np.logical_and(~np.isnan(u), ~np.isnan(v))
    
            # 5. 地图绘制（参数与show.py保持一致）
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            if lon_min and lon_max and lat_min and lat_max:
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
            # 流速填色底图
            contour = ax.contourf(
                lon_grid, lat_grid, speed_plot,
                cmap=contourf_cmap,
                levels=contourf_levels,
                transform=ccrs.PlateCarree(),
                zorder=contourf_zorder
            )
            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='流速 (m/s)')
    
            # 有效掩膜区域画箭头
            q = ax.quiver(
                lon_grid[valid_mask][::skip], lat_grid[valid_mask][::skip],
                u[valid_mask][::skip], v[valid_mask][::skip],
                transform=ccrs.PlateCarree(),
                color='black',
                scale=quiver_scale, width=quiver_width,
                zorder=quiver_zorder
            )
    
            # 地理特征
            ax.add_feature(cfeature.COASTLINE, edgecolor='black')
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
    
            # 经纬网
            gridlines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.8, color='gray', alpha=0.7, linestyle='--')
            gridlines.xlabels_top = False
            gridlines.ylabels_right = False
            gridlines.xlabel_style = {'size': 10, 'color': 'gray'}
            gridlines.ylabel_style = {'size': 10, 'color': 'gray'}
    
            # 生成“当前时间”字符串
            now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 标题拼接
            title = "洋流场\n"




            if self.depth is not None:
                title += f"深度: {self.depth:.2f} m, "
            if self.time_info:
                title += f"数据时刻: {self.time_info}, "
            title += f"生成时间: {now_time}"
            ax.set_title(title, fontsize=font_size+2)
    
            # 输出
            if save_path:
                plt.savefig(save_path, dpi=dpi)
            if show:
                plt.show()
            plt.close()


    def export_vector_shapefile(self, out_path, skip=5, file_type="shp"):
        """
        导出矢量场为Shapefile/GeoJSON
        """
        features = []
        for i in range(0, len(self.lat), skip):
            for j in range(0, len(self.lon), skip):
                u_ij = self.u[i, j]
                v_ij = self.v[i, j]
                if np.isfinite(u_ij) and np.isfinite(v_ij):
                    start = Point(self.lon[j], self.lat[i])
                    direction = np.arctan2(v_ij, u_ij)
                    speed = np.sqrt(u_ij**2 + v_ij**2)
                    end = Point(self.lon[j] + speed * np.cos(direction),
                                self.lat[i] + speed * np.sin(direction))
                    line = LineString([start, end])
                    features.append({
                        'geometry': line,
                        'speed': float(speed),
                        'direction': float(np.degrees(direction))
                    })
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        if file_type == "shp":
            gdf.to_file(out_path + ".shp")
        elif file_type == "geojson":
            gdf.to_file(out_path + ".geojson", driver="GeoJSON")
        print(f"[导出完成] 矢量场已导出到 {out_path}.{file_type}")

# ================== 可独立运行测试块 ==================
if __name__ == '__main__':
    from netcdf_handler import NetCDFHandler
    nc_path = '../data/raw_data/merged_data.nc'
    handler = NetCDFHandler(nc_path)
    u, v, lat, lon = handler.get_uv(time_idx=1, depth_idx=2)
    # 假定 depth/time 字段分别为2.0/“2024-09-01 06:00:00”，实际可用 handler 读出来
    processor = DataProcessor(u, v, lat, lon, depth=2.0, time_info="2024-09-01 06:00:00")
    processor.plot_vector_field(
        skip=3,
        show=True,
        save_path="../output/专业矢量场.png",
        lon_min=118, lon_max=124, lat_min=21, lat_max=26.5
    )
    handler.close()
