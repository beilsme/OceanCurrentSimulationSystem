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


    def get_time(self, index=0):
        """
        获取指定索引的时间，默认为第一个
        - 如果是多时刻数据，会自动提示
        - index 参数保留可扩展，后续可用来交互
        """
        if "time" in self.ds.variables:
            time_var = self.ds["time"]
            try:
                times = time_var.values
                if times is not None and len(times) > 0:
                    if len(times) > 1:
                        print(f"[INFO] 检测到 NetCDF 中包含多个时间点 (共 {len(times)} 个)，当前默认使用索引 {index}")
                        # 后续可以支持外部传入index
                    import pandas as pd
                    t_value = pd.to_datetime(times[index])
                    return t_value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    print("[WARN] time 变量存在但为空")
                    return None
            except Exception as e:
                print(f"[WARN] 时间读取失败: {e}")
                return None
        else:
            print("[WARN] 文件中未找到 time 变量")
        return None

    def get_depth(self, index=0):
        """
        获取指定索引的深度，默认为第一个
        - 如果是多层深度数据，会自动提示
        - index 参数可扩展供用户后续交互
        """
        if "depth" in self.ds.variables:
            depth_var = self.ds["depth"]
            try:
                depths = depth_var.values
                if depths is not None and len(depths) > 0:
                    if len(depths) > 1:
                        print(f"[INFO] 检测到 NetCDF 中包含多个深度层 (共 {len(depths)} 层)，当前默认使用索引 {index}")
                        # 后续可以支持外部传入 index
                    depth_value = float(depths[index])
                    return depth_value
                else:
                    print("[WARN] depth 变量存在但为空")
                    return None
            except Exception as e:
                print(f"[WARN] 深度读取失败: {e}")
                return None
        else:
            print("[WARN] 文件中未找到 depth 变量")
        return None

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

    print("\n===== 列出所有变量 =====")
    handler.list_variables()

    # 先打印 time 变量本身
    if "time" in handler.ds.variables:
        time_var = handler.ds["time"]
        print("\n========== DEBUG: time_var ==========")
        print(time_var)
        print("\n========== DEBUG: time_var[:10] ==========")
        try:
            print(time_var[:10])
        except Exception as e:
            print(f"取前10失败: {e}")
        print("\n========== DEBUG: time_var attributes ==========")
        attrs = getattr(time_var, "attrs", None)
        if attrs:
            print(attrs)
        else:
            try:
                print(time_var.attrs)
            except:
                print("没有 attrs")
    else:
        print("⚠️ 文件中没有 time 变量")

    # 测试 get_time()
    times = handler.get_time()
    print("\n===== 测试 get_time() 返回值 =====")
    print(times)

    if times is not None:
        if isinstance(times, (list, tuple)) or hasattr(times, "__getitem__"):
            print("第一个时间索引:", times[0])
        else:
            print("时间标量:", times)
    else:
        print("没有读取到时间")

    # 再测 uv
    u, v, lat, lon = handler.get_uv(time_idx=0, depth_idx=0)
    print("\n===== uv 维度检查 =====")
    print("u shape:", u.shape)
    print("v shape:", v.shape)
    print("lat shape:", lat.shape)
    print("lon shape:", lon.shape)

    handler.close()



