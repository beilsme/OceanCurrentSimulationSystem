#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: quick_start.py
功能: TOPAZ系统测试数据快速下载脚本
作者: beilsm
版本: v1.0.0

简化版本，专注于获取最基本的测试数据：
- 公开的NOAA海表温度数据
- GEBCO海底地形数据
- 模拟的观测数据用于算法测试
"""

import os
import sys
import requests
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import json
from typing import Tuple, Dict, Any


class QuickStartDownloader:
    """快速启动数据下载器 - 无需认证的公开数据"""

    def __init__(self, output_dir: str = "./test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 测试区域配置 (挪威海小区域)
        self.test_region = {
            'lat_min': 65.0,
            'lat_max': 75.0,
            'lon_min': 0.0,
            'lon_max': 20.0
        }

        # 测试时间 (一周)
        self.test_dates = self.generate_test_dates("2008-04-01", 7)

        self.logger.info(f"快速启动下载器初始化完成，输出目录: {self.output_dir}")

    def generate_test_dates(self, start_date: str, num_days: int) -> list:
        """生成测试日期列表"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(num_days)]

    def download_file_simple(self, url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """简单的文件下载方法"""
        try:
            self.logger.info(f"下载: {url}")

            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"下载完成: {output_path.name} ({file_size:.2f} MB)")
            return True

        except Exception as e:
            self.logger.error(f"下载失败: {str(e)}")
            return False

    def download_noaa_sst_sample(self) -> bool:
        """下载NOAA海表温度样本数据"""
        self.logger.info("下载NOAA海表温度数据...")

        # NOAA OISST高分辨率数据 (公开访问)
        base_url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"

        # 下载2008年4月的数据
        year_month = "200804"
        filename = f"oisst-avhrr-v02r01.{year_month}01.nc"
        url = f"{base_url}/{year_month[:4]}/{year_month[4:]}/{filename}"

        output_path = self.output_dir / "sst" / filename

        return self.download_file_simple(url, output_path)

    def download_gebco_sample(self) -> bool:
        """下载GEBCO海底地形样本数据"""
        self.logger.info("下载GEBCO海底地形数据...")

        # 使用GEBCO Web API获取区域子集
        bbox = f"{self.test_region['lon_min']},{self.test_region['lat_min']}," + \
               f"{self.test_region['lon_max']},{self.test_region['lat_max']}"

        # GEBCO 2023 Web Map Service
        url = f"https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv?" + \
              f"request=GetMap&service=WMS&version=1.3.0&layers=GEBCO_LATEST&styles=default&" + \
              f"crs=EPSG:4326&bbox={bbox}&width=200&height=200&format=image/png"

        # 改用直接的NetCDF下载链接 (如果可用)
        gebco_url = "https://download.gebco.net/api/v1/coverage/gebco_2023.nc?" + \
                    f"bbox={bbox}&format=netcdf"

        output_path = self.output_dir / "bathymetry" / "gebco_test_region.nc"

        return self.download_file_simple(gebco_url, output_path)

    def create_synthetic_sla_data(self) -> bool:
        """创建合成的海平面高度异常数据用于测试"""
        self.logger.info("创建合成海平面高度异常数据...")

        try:
            # 创建网格
            lats = np.linspace(self.test_region['lat_min'], self.test_region['lat_max'], 20)
            lons = np.linspace(self.test_region['lon_min'], self.test_region['lon_max'], 40)

            output_dir = self.output_dir / "sla_synthetic"
            output_dir.mkdir(parents=True, exist_ok=True)

            for date_str in self.test_dates:
                filename = f"sla_synthetic_{date_str.replace('-', '')}.nc"
                output_path = output_dir / filename

                # 创建合成数据 (模拟真实的海洋信号)
                LON, LAT = np.meshgrid(lons, lats)

                # 添加时间变化
                day_of_year = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
                time_factor = np.sin(2 * np.pi * day_of_year / 365.0)

                # 模拟海平面高度异常 (cm)
                sla = (10 * np.sin(2 * np.pi * LON / 20) * np.cos(2 * np.pi * LAT / 10) +
                       5 * time_factor +
                       np.random.normal(0, 2, LON.shape))

                # 创建NetCDF文件
                with nc.Dataset(output_path, 'w') as ncfile:
                    # 创建维度
                    ncfile.createDimension('latitude', len(lats))
                    ncfile.createDimension('longitude', len(lons))
                    ncfile.createDimension('time', 1)

                    # 创建变量
                    lat_var = ncfile.createVariable('latitude', 'f4', ('latitude',))
                    lon_var = ncfile.createVariable('longitude', 'f4', ('longitude',))
                    time_var = ncfile.createVariable('time', 'f4', ('time',))
                    sla_var = ncfile.createVariable('sla', 'f4', ('time', 'latitude', 'longitude'))

                    # 填充数据
                    lat_var[:] = lats
                    lon_var[:] = lons
                    time_var[0] = (datetime.strptime(date_str, "%Y-%m-%d") -
                                   datetime(1950, 1, 1)).days
                    sla_var[0, :, :] = sla

                    # 添加属性
                    lat_var.units = 'degrees_north'
                    lon_var.units = 'degrees_east'
                    time_var.units = 'days since 1950-01-01'
                    sla_var.units = 'cm'
                    sla_var.long_name = 'Sea Level Anomaly'

                    # 全局属性
                    ncfile.title = 'Synthetic Sea Level Anomaly for TOPAZ Testing'
                    ncfile.source = 'Generated for EnKF algorithm testing'
                    ncfile.date_created = datetime.now().isoformat()

                self.logger.info(f"创建合成SLA文件: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"创建合成SLA数据失败: {str(e)}")
            return False

    def create_synthetic_argo_data(self) -> bool:
        """创建合成的Argo剖面数据"""
        self.logger.info("创建合成Argo剖面数据...")

        try:
            output_dir = self.output_dir / "argo_synthetic"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 创建几个虚拟的Argo剖面
            depths = np.array([0, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000])

            for i, date_str in enumerate(self.test_dates[:3]):  # 只创建3个剖面
                # 随机位置在测试区域内
                lat = np.random.uniform(self.test_region['lat_min'], self.test_region['lat_max'])
                lon = np.random.uniform(self.test_region['lon_min'], self.test_region['lon_max'])

                filename = f"argo_profile_{i+1}_{date_str.replace('-', '')}.nc"
                output_path = output_dir / filename

                # 创建典型的北大西洋温盐剖面
                temperature = 15 - 0.01 * depths - 0.001 * depths**1.2 + np.random.normal(0, 0.1, len(depths))
                salinity = 35.0 - 0.002 * depths + np.random.normal(0, 0.02, len(depths))

                # 应用物理约束
                temperature = np.maximum(-2, np.minimum(25, temperature))
                salinity = np.maximum(30, np.minimum(37, salinity))

                with nc.Dataset(output_path, 'w') as ncfile:
                    # 创建维度
                    ncfile.createDimension('N_LEVELS', len(depths))
                    ncfile.createDimension('N_PROF', 1)

                    # 创建变量
                    pres_var = ncfile.createVariable('PRES', 'f4', ('N_PROF', 'N_LEVELS'))
                    temp_var = ncfile.createVariable('TEMP', 'f4', ('N_PROF', 'N_LEVELS'))
                    sal_var = ncfile.createVariable('PSAL', 'f4', ('N_PROF', 'N_LEVELS'))
                    lat_var = ncfile.createVariable('LATITUDE', 'f4', ('N_PROF',))
                    lon_var = ncfile.createVariable('LONGITUDE', 'f4', ('N_PROF',))

                    # 填充数据
                    pres_var[0, :] = depths
                    temp_var[0, :] = temperature
                    sal_var[0, :] = salinity
                    lat_var[0] = lat
                    lon_var[0] = lon

                    # 添加属性
                    pres_var.units = 'decibar'
                    temp_var.units = 'degree_Celsius'
                    sal_var.units = 'psu'
                    lat_var.units = 'degree_north'
                    lon_var.units = 'degree_east'

                    # 全局属性
                    ncfile.title = 'Synthetic Argo Profile for TOPAZ Testing'
                    ncfile.institution = 'Test Data Generator'
                    ncfile.date_created = datetime.now().isoformat()

                self.logger.info(f"创建合成Argo剖面: {filename}")

            return True

        except Exception as e:
            self.logger.error(f"创建合成Argo数据失败: {str(e)}")
            return False

    def create_test_configuration(self) -> bool:
        """创建测试配置文件"""
        self.logger.info("创建测试配置文件...")

        config = {
            "test_metadata": {
                "description": "TOPAZ EnKF系统测试数据集",
                "region": "挪威海测试区域",
                "time_period": f"{self.test_dates[0]} 到 {self.test_dates[-1]}",
                "created": datetime.now().isoformat()
            },
            "spatial_domain": self.test_region,
            "temporal_domain": {
                "start_date": self.test_dates[0],
                "end_date": self.test_dates[-1],
                "num_days": len(self.test_dates)
            },
            "data_inventory": {
                "sea_surface_temperature": "NOAA OISST (真实数据)",
                "sea_level_anomaly": "合成数据 (基于物理模型)",
                "argo_profiles": "合成数据 (典型北大西洋剖面)",
                "bathymetry": "GEBCO (真实数据)"
            },
            "enkf_parameters": {
                "ensemble_size": 20,  # 测试用减少的集合大小
                "localization_radius_km": 100,
                "inflation_factor": 1.02,
                "grid_resolution_km": 50
            },
            "usage_notes": [
                "此数据集仅用于EnKF算法功能测试",
                "合成数据包含物理合理的海洋信号",
                "网格分辨率已优化用于快速计算",
                "实际应用需要使用真实观测数据"
            ]
        }

        config_file = self.output_dir / "test_config.json"

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"测试配置文件已保存: {config_file}")
        return True

    def validate_downloaded_data(self) -> Dict[str, Any]:
        """验证下载的数据"""
        self.logger.info("验证下载的数据...")

        validation_results = {
            "sst_data": False,
            "sla_data": False,
            "argo_data": False,
            "bathymetry_data": False,
            "total_size_mb": 0.0,
            "file_count": 0
        }

        # 检查各类数据文件
        data_dirs = {
            "sst": "sst_data",
            "sla_synthetic": "sla_data",
            "argo_synthetic": "argo_data",
            "bathymetry": "bathymetry_data"
        }

        for dir_name, result_key in data_dirs.items():
            data_dir = self.output_dir / dir_name
            if data_dir.exists():
                files = list(data_dir.glob("*.nc"))
                if files:
                    validation_results[result_key] = True
                    for file_path in files:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        validation_results["total_size_mb"] += size_mb
                        validation_results["file_count"] += 1

        # 生成验证报告
        report_file = self.output_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        return validation_results

    def run_quick_download(self) -> bool:
        """执行快速下载流程"""
        self.logger.info("开始快速测试数据下载...")

        start_time = time.time()

        # 执行下载任务
        tasks = [
            ("NOAA海表温度数据", self.download_noaa_sst_sample),
            ("GEBCO海底地形数据", self.download_gebco_sample),
            ("合成海平面高度异常数据", self.create_synthetic_sla_data),
            ("合成Argo剖面数据", self.create_synthetic_argo_data),
            ("测试配置文件", self.create_test_configuration)
        ]

        results = {}
        for task_name, task_func in tasks:
            self.logger.info(f"执行任务: {task_name}")
            try:
                results[task_name] = task_func()
                if results[task_name]:
                    self.logger.info(f"✅ {task_name} - 完成")
                else:
                    self.logger.warning(f"⚠️ {task_name} - 失败")
            except Exception as e:
                self.logger.error(f"❌ {task_name} - 错误: {str(e)}")
                results[task_name] = False

        # 验证数据
        validation = self.validate_downloaded_data()

        end_time = time.time()

        # 输出摘要
        successful_tasks = sum(1 for success in results.values() if success)
        total_tasks = len(results)

        print("\n" + "="*60)
        print("TOPAZ测试数据下载完成!")
        print("="*60)
        print(f"成功任务: {successful_tasks}/{total_tasks}")
        print(f"总耗时: {end_time - start_time:.1f} 秒")
        print(f"数据文件数量: {validation['file_count']}")
        print(f"总数据大小: {validation['total_size_mb']:.1f} MB")
        print(f"输出目录: {self.output_dir.absolute()}")

        if successful_tasks == total_tasks:
            print("\n🎉 所有测试数据准备就绪!")
            print("💡 现在可以开始测试您的EnKF系统了")
            return True
        else:
            print(f"\n⚠️ 部分任务失败，但您仍可以使用可用的数据进行测试")
            return False


def main():
    """主函数"""
    print("TOPAZ EnKF系统 - 快速测试数据下载器")
    print("="*50)
    print("此脚本将下载小量测试数据，无需认证配置")
    print("适合快速验证系统功能")
    print()

    # 获取输出目录
    output_dir = "./test_data"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    print(f"输出目录: {output_dir}")
    print("开始下载...")
    print()

    # 创建下载器并执行
    downloader = QuickStartDownloader(output_dir)

    try:
        success = downloader.run_quick_download()

        if success:
            print("\n下一步:")
            print("1. 检查生成的test_config.json配置文件")
            print("2. 运行您的EnKF系统进行测试")
            print("3. 如果测试成功，可以下载完整的数据集")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n❌ 用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 下载过程发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()