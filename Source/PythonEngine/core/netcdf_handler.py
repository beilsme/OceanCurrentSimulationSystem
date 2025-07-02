# ===============================================================================
# 文件名称：core/netcdf_handler.py
# 模块接口：class NetCDFHandler
# 作者：beilsm
# 版本号：v0.1.0
# 功能：NetCDF文件的统一读取、基础数据提取接口
# 主要功能：
#   - 读取NetCDF（如HYCOM）格式洋流数据文件，提取U/V分量、经纬度、深度等字段
#   - 提供灵活的接口，允许按时间/深度维度检索
#   - 便于后续用于矢量场、轨迹模拟、数据导出等各类模块调用
# 较上一版改进：
#   - 抽象为面向对象类接口，支持多次复用
#   - 不再写死路径与变量名，参数灵活可调
#   - 兼容HYCOM与常规结构NetCDF，方便未来拓展
# 最新更改时间：20250-06-28
# ===============================================================================
import xarray as xr

class NetCDFHandler:
    """
    NetCDF文件处理类。负责读取、解析HYCOM等格式的NetCDF数据，
    支持按需抽取指定时间、深度的U/V分量和空间坐标。
    """

    def __init__(self, file_path):
        """
        初始化，加载NetCDF文件
        :param file_path: NetCDF文件路径
        """
        self.ds = xr.open_dataset(file_path)

    def list_variables(self):
        """
        输出当前NetCDF文件包含的全部变量名
        """
        print("变量列表：")
        for var in self.ds.variables:
            print(f" - {var}")


    def get_uv(self, time_idx=0, depth_idx=0):
        """
        提取指定时间、深度层的u/v/lat/lon（2D场）
        :param time_idx: 时间维度索引（默认0）
        :param depth_idx: 深度维度索引（默认0）
        :return: u, v, lat, lon（均为ndarray）
        """
        u = self.ds['water_u'].isel(time=time_idx, depth=depth_idx)
        v = self.ds['water_v'].isel(time=time_idx, depth=depth_idx)
        lat = self.ds['lat'].values
        lon = self.ds['lon'].values
        return u.values, v.values, lat, lon
    
    def get_3d_uv(self, time_idx=0):
        """
        提取指定时间的三维u/v（Depth, Lat, Lon）
        :param time_idx: 时间索引
        :return: u, v, depth, lat, lon
        """
        u = self.ds['water_u'].isel(time=time_idx)
        v = self.ds['water_v'].isel(time=time_idx)
        depth = self.ds['depth'].values
        lat = self.ds['lat'].values
        lon = self.ds['lon'].values
        return u.values, v.values, depth, lat, lon


    def close(self):
        """关闭数据集（释放内存）"""
        self.ds.close()

if __name__ == '__main__':
    test_path = r'../data/raw_data/merged_data.nc'
    handler = NetCDFHandler(test_path)
    handler.list_variables()
    u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
    print("u shape:", u.shape)
    print("v shape:", v.shape)
    print("lat shape:", lat.shape)
    print("lon shape:", lon.shape)
    handler.close()


