# ==============================================================================
# 文件名称：core/data_processor.py
# 接口名称：DataProcessor
# 作者：beilsm
# 版本号：v1.0.0
# 创建时间：2025-07-01 18:30
# 最新更改时间：2025-07-01 18:30
# ==============================================================================
# ✅ 功能简介：
#   - 核心数据处理器，支持洋流场（uv）可视化与矢量导出
#   - 可作为所有后续模拟、分析、可视化的基础处理层
# ==============================================================================
# ✅ 更新说明：
#   - 首次实现，集成可视化与shapefile导出
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString, Point
import geopandas as gpd

class DataProcessor:
    """
    核心数据处理器：支持uv场可视化、导出等功能
    """
    def __init__(self, u, v, lat, lon):
        self.u = u
        self.v = v
        self.lat = lat
        self.lon = lon

    def plot_vector_field(self, skip=5, show=True, save_path=None):
        """
        可视化uv矢量场
        """
        speed = np.sqrt(self.u**2 + self.v**2)
        speed_norm = (speed - speed.min()) / (speed.max() - speed.min())
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        colors = plt.cm.viridis(speed_norm[::skip, ::skip])
        colors = colors.reshape(-1, 4)
        ax.quiver(self.lon[::skip], self.lat[::skip], self.u[::skip, ::skip], self.v[::skip, ::skip],
                  angles='xy', scale_units='xy', scale=1, color=colors, width=0.002)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=speed.min(), vmax=speed.max())),
            ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Speed (m/s)', fontsize=12)
        ax.set_title('Ocean Current Vector Field', fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE, edgecolor='black')
        if save_path:
            plt.savefig(save_path, dpi=150)
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
    nc_path = '../data/raw_data/merged_data.nc'  # 示例路径
    handler = NetCDFHandler(nc_path)
    u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
    processor = DataProcessor(u, v, lat, lon)
    processor.plot_vector_field(skip=10, show=True, save_path="../output/vector_field.png")
    processor.export_vector_shapefile("../output/current_vectors", skip=10, file_type="shp")
    handler.close()
