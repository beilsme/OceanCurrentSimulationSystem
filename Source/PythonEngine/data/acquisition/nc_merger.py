# ==============================================================================
# 文件名称：nc_merger.py
# 接口名称：merge_uv_ncfiles
# 作者：beilsm
# 版本号：v1.0.0
# 创建时间：2025-06-28
# 最新更改时间：2025-06-28
# ==============================================================================
# ✅ 功能简介：
#   - 批量合并存放在不同文件夹下的 water_u 与 water_v NetCDF 文件
#   - 输出为单一包含全部时间步的多变量 NetCDF 文件（含完整维度）
#   - 适配专业工程结构与下游数据处理模块调用
# ==============================================================================
# ✅ 更新说明：
#   - 首次工程化封装
#   - 标准路径参数化，文件排序安全
#   - 支持直接命令行测试
# ==============================================================================

import glob
import xarray as xr
import os

def merge_uv_ncfiles(path_u, path_v, output_file):
    """
    合并指定文件夹下的 water_u 和 water_v NetCDF 文件为一个多变量时序文件
    :param path_u: 存放 water_u nc 文件的文件夹路径
    :param path_v: 存放 water_v nc 文件的文件夹路径
    :param output_file: 合并后输出 nc 文件的路径
    :return: 输出文件路径
    """
    all_files_u = glob.glob(os.path.join(path_u, 'HYCOM_water_u_*.nc'))
    all_files_v = glob.glob(os.path.join(path_v, 'HYCOM_water_v_*.nc'))
    all_files_u.sort()
    all_files_v.sort()

    if not all_files_u or not all_files_v:
        print("[错误] 未检测到 water_u 或 water_v 文件，请检查路径和文件名！")
        return None

    print(f"[INFO] 共检测到 {len(all_files_u)} 组数据，将依次合并...")

    time_list = []
    for idx, (u_file, v_file) in enumerate(zip(all_files_u, all_files_v)):
        print(f"  [合并] 第 {idx+1} 组：{os.path.basename(u_file)} & {os.path.basename(v_file)}")
        ds_u = xr.open_dataset(u_file)
        ds_v = xr.open_dataset(v_file)
        water_u = ds_u['water_u']
        water_v = ds_v['water_v']
        merged_data = xr.Dataset({'water_u': water_u, 'water_v': water_v})
        time_list.append(merged_data)
        ds_u.close()
        ds_v.close()

    merged_ds = xr.concat(time_list, dim='time')
    merged_ds.to_netcdf(output_file)
    print(f"[完成] 合并后的多变量多时刻 NetCDF 已保存：{output_file}")
    return output_file

# ================== 可独立运行测试块 ==================
if __name__ == '__main__':
    # 推荐用工程绝对路径或项目根目录下相对路径
    path_u = '../raw_data/water_u'
    path_v = '../raw_data/water_v'
    output_file = '../raw_data/merged_uv.nc'

    out = merge_uv_ncfiles(path_u, path_v, output_file)
    if out:
        with xr.open_dataset(out) as merged:
            print("合并后的数据集信息如下：")
            print(merged)

# ================== requirements.txt 需包含 ==================
# xarray
