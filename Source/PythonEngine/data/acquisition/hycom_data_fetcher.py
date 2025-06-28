# ============================================================
# 模块名称：hycom_data_fetcher.py
# 接口说明：HYCOM 洋流数据下载器（无 dataset 参数版本）
# 作者：beilsm
# 版本：v2.0.2
# 创建时间：2025-06-14
# 最后修改时间：2025-06-14
# ============================================================
# ✅ 核心功能：
#   - 从默认 HYCOM 接口下载 water_u / water_v 等变量数据
#   - 自动构造时间戳序列并生成标准文件路径
#   - 支持失败重试机制
#   - 可与读取模块联动使用
# ============================================================
# ✅ 更新说明（v2.0.2）：
#   - 修复 v2.0.1 中 download() 参数错误（store_path → output_dir）
#   - 明确统一输出路径参数为 output_dir
#   - 保持参数接口一致性，提升兼容性与稳定性
# ============================================================


import os
import time
from datetime import datetime, timedelta
from oafuncs.oa_down.hycom_3hourly import download


def generate_time_list(start_str: str, end_str: str, interval_hours: int = 3) -> list:
    time_format = "%Y%m%d%H"
    start = datetime.strptime(start_str, time_format)
    end = datetime.strptime(end_str, time_format)
    result = []
    while start <= end:
        result.append(start.strftime(time_format))
        start += timedelta(hours=interval_hours)
    return result


def download_with_retry(variable: str,
                        start_time: str,
                        end_time: str,
                        lon_min: float,
                        lon_max: float,
                        lat_min: float,
                        lat_max: float,
                        output_dir: str,
                        retries: int = 3,
                        interval_hours: int = 3) -> list:
    """
    使用默认的 HYCOM 下载接口（无 dataset/version 参数）
    """
    for attempt in range(retries):
        try:
            print(f"[尝试下载] 第 {attempt + 1} 次：{variable}")
            download(
                variable,
                start_time=start_time,
                end_time=end_time,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                output_dir=output_dir,
            )
            print(f"[成功] 下载完成：{variable}（{start_time} ~ {end_time}）")
            break
        except Exception as e:
            print(f"[失败] 第 {attempt + 1} 次失败：{e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("[终止] 下载失败，超过最大重试次数。")
                return []

    time_list = generate_time_list(start_time, end_time, interval_hours)
    file_list = [os.path.join(store_path, f"HYCOM_{variable}_{t}.nc") for t in time_list]
    return file_list


# ✅ 测试入口
if __name__ == "__main__":
    store_path = "../raw_data"
    os.makedirs(store_path, exist_ok=True)

    file_list = download_with_retry(
        variable='water_u',
        start_time='2024090100',
        end_time='2024093021',
        lon_min=118,
        lon_max=124,
        lat_min=21,
        lat_max=26.5,
        output_dir=store_path
    )

    print(f"[文件路径列表]")
    for f in file_list:
        print(f)
