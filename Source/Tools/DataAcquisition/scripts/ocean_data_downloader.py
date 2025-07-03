#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: ocean_data_downloader.py
功能: TOPAZ系统海洋观测数据自动下载脚本
作者: beilsm
版本: v1.0.0
创建时间: 2025-07-03

主要功能:
- 海平面高度异常数据下载 (AVISO+)
- 海表温度数据下载 (NOAA Reynolds)
- 海冰浓度数据下载 (NSIDC)
- Argo浮标数据下载 (GDAC)
- 大气再分析数据下载 (ECMWF)
- 海底地形数据下载 (GEBCO)
"""

import os
import sys
import requests
import json
import yaml
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, date
from pathlib import Path
import time
import hashlib
import gzip
import shutil
from typing import Dict, List, Tuple, Optional, Any
import ftplib
from urllib.error import URLError, HTTPError
import xml.etree.ElementTree as ET
import netrc
from dataclasses import dataclass


@dataclass
class DownloadConfig:
    """下载配置数据类"""
    # 时间范围
    start_date: str = "2008-04-01"
    end_date: str = "2008-04-07"

    # 空间范围 (测试区域: 挪威海)
    lat_min: float = 65.0
    lat_max: float = 75.0
    lon_min: float = 0.0
    lon_max: float = 20.0

    # 输出目录
    base_output_dir: str = "./ocean_data"

    # 数据源配置
    download_sla: bool = True      # 海平面高度异常
    download_sst: bool = True      # 海表温度
    download_ice: bool = True      # 海冰数据
    download_argo: bool = True     # Argo数据
    download_atmos: bool = True    # 大气数据
    download_bathy: bool = True    # 海底地形

    # 下载参数
    max_retries: int = 3
    retry_delay: int = 5
    chunk_size: int = 8192
    timeout: int = 300


class OceanDataDownloader:
    """海洋数据下载管理器"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TOPAZ-EnKF-System/1.0 (Research Purpose)'
        })

        # 设置日志
        self.setup_logging()

        # 创建输出目录结构
        self.setup_directories()

        # 统计信息
        self.download_stats = {
            'total_files': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_size_mb': 0.0
        }

    def setup_logging(self):
        """设置日志记录"""
        log_dir = Path(self.config.base_output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("海洋数据下载器初始化完成")

    def setup_directories(self):
        """创建目录结构"""
        base_dir = Path(self.config.base_output_dir)

        self.dirs = {
            'sla': base_dir / "sea_level_anomaly",
            'sst': base_dir / "sea_surface_temperature",
            'ice': base_dir / "sea_ice",
            'argo': base_dir / "argo_profiles",
            'atmos': base_dir / "atmospheric_forcing",
            'bathy': base_dir / "bathymetry",
            'logs': base_dir / "logs",
            'config': base_dir / "config"
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"目录结构创建完成: {base_dir}")

    def download_file(self, url: str, output_path: Path,
                      auth: Optional[Tuple[str, str]] = None,
                      headers: Optional[Dict[str, str]] = None) -> bool:
        """通用文件下载方法"""

        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"下载文件 (尝试 {attempt + 1}/{self.config.max_retries}): {url}")

                # 准备请求
                req_headers = self.session.headers.copy()
                if headers:
                    req_headers.update(headers)

                # 发送请求
                if auth:
                    response = self.session.get(url, auth=auth, headers=req_headers,
                                                timeout=self.config.timeout, stream=True)
                else:
                    response = self.session.get(url, headers=req_headers,
                                                timeout=self.config.timeout, stream=True)

                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 下载文件
                output_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_size = 0

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                # 验证下载
                if total_size > 0 and downloaded_size != total_size:
                    raise ValueError(f"下载不完整: {downloaded_size}/{total_size} bytes")

                file_size_mb = downloaded_size / (1024 * 1024)
                self.download_stats['total_size_mb'] += file_size_mb
                self.download_stats['successful_downloads'] += 1

                self.logger.info(f"下载成功: {output_path.name} ({file_size_mb:.2f} MB)")
                return True

            except Exception as e:
                self.logger.warning(f"下载失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"下载最终失败: {url}")
                    self.download_stats['failed_downloads'] += 1
                    return False

        return False

    def generate_date_range(self) -> List[date]:
        """生成日期范围"""
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d").date()

        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)

        return dates

    def download_aviso_sla_data(self) -> bool:
        """下载AVISO海平面高度异常数据"""
        if not self.config.download_sla:
            return True

        self.logger.info("开始下载AVISO海平面高度异常数据...")

        dates = self.generate_date_range()
        success_count = 0

        for date_obj in dates:
            year = date_obj.year
            month = date_obj.month
            day = date_obj.day

            # AVISO数据URL格式 (需要根据实际API调整)
            filename = f"dt_global_allsat_phy_l4_{date_obj.strftime('%Y%m%d')}_20190101.nc"

            # 构建下载URL (示例URL，需要替换为实际的AVISO服务地址)
            base_url = "https://my.aviso.altimetry.fr/fileserver/oa/dataset-duacs-rep-global-merged-allsat-phy-l4"
            url = f"{base_url}/{year}/{month:02d}/{filename}"

            output_path = self.dirs['sla'] / filename

            if output_path.exists():
                self.logger.info(f"文件已存在，跳过: {filename}")
                continue

            # 注意：AVISO需要认证，您需要在.netrc文件中配置用户名和密码
            # 或者使用环境变量 AVISO_USERNAME 和 AVISO_PASSWORD
            auth = self.get_aviso_credentials()

            if self.download_file(url, output_path, auth=auth):
                success_count += 1

            # 避免过于频繁的请求
            time.sleep(1)

        self.logger.info(f"AVISO数据下载完成: {success_count}/{len(dates)} 文件成功")
        return success_count > 0

    def download_reynolds_sst_data(self) -> bool:
        """下载Reynolds海表温度数据"""
        if not self.config.download_sst:
            return True

        self.logger.info("开始下载Reynolds海表温度数据...")

        dates = self.generate_date_range()
        success_count = 0

        for date_obj in dates:
            year = date_obj.year

            # Reynolds SST文件名格式
            filename = f"sst.day.mean.{date_obj.strftime('%Y')}.nc"

            # NOAA PSL数据URL
            base_url = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
            url = f"{base_url}/{filename}"

            output_path = self.dirs['sst'] / filename

            if output_path.exists():
                self.logger.info(f"文件已存在，跳过: {filename}")
                success_count += 1
                continue

            if self.download_file(url, output_path):
                success_count += 1

            time.sleep(1)

        self.logger.info(f"Reynolds SST数据下载完成: {success_count}/{len(dates)} 文件成功")
        return success_count > 0

    def download_nsidc_ice_data(self) -> bool:
        """下载NSIDC海冰数据"""
        if not self.config.download_ice:
            return True

        self.logger.info("开始下载NSIDC海冰浓度数据...")

        dates = self.generate_date_range()
        success_count = 0

        for date_obj in dates:
            # NSIDC海冰浓度文件格式
            filename = f"nt_{date_obj.strftime('%Y%m%d')}_f17_nrt_n.bin"

            # NSIDC数据URL (需要认证)
            base_url = "https://n5eil01u.ecs.nsidc.org/SIPS/NSIDC-0051.002"
            year_month = date_obj.strftime('%Y.%m.%d')
            url = f"{base_url}/{year_month}/{filename}"

            output_path = self.dirs['ice'] / filename

            if output_path.exists():
                self.logger.info(f"文件已存在，跳过: {filename}")
                continue

            # NSIDC需要EarthData登录认证
            auth = self.get_earthdata_credentials()

            if self.download_file(url, output_path, auth=auth):
                success_count += 1

            time.sleep(1)

        self.logger.info(f"NSIDC海冰数据下载完成: {success_count}/{len(dates)} 文件成功")
        return success_count > 0

    def download_argo_profiles(self) -> bool:
        """下载Argo剖面数据"""
        if not self.config.download_argo:
            return True

        self.logger.info("开始下载Argo剖面数据...")

        # Argo数据通过FTP下载
        ftp_host = "ftp.ifremer.fr"
        base_path = "/ifremer/argo/dac"

        success_count = 0

        try:
            with ftplib.FTP(ftp_host) as ftp:
                ftp.login()  # 匿名登录

                # 获取测试区域内的剖面列表
                profiles = self.get_argo_profiles_in_region(ftp, base_path)

                for profile_path in profiles[:10]:  # 限制下载数量用于测试
                    filename = os.path.basename(profile_path)
                    output_path = self.dirs['argo'] / filename

                    if output_path.exists():
                        continue

                    try:
                        with open(output_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {profile_path}', f.write)

                        success_count += 1
                        self.logger.info(f"Argo剖面下载成功: {filename}")

                    except Exception as e:
                        self.logger.warning(f"Argo剖面下载失败: {filename} - {str(e)}")

                    time.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Argo数据下载失败: {str(e)}")
            return False

        self.logger.info(f"Argo数据下载完成: {success_count} 个剖面")
        return success_count > 0

    def download_era_interim_data(self) -> bool:
        """下载ERA-Interim大气再分析数据"""
        if not self.config.download_atmos:
            return True

        self.logger.info("开始下载ERA-Interim大气数据...")

        # 注意：这里需要ECMWF API密钥
        # 实际实现需要安装ecmwf-api-client包并配置API密钥

        try:
            from ecmwfapi import ECMWFDataServer
        except ImportError:
            self.logger.error("需要安装ecmwf-api-client: pip install ecmwf-api-client")
            return False

        server = ECMWFDataServer()

        # ERA-Interim请求参数
        request_params = {
            "class": "ei",
            "dataset": "interim",
            "date": f"{self.config.start_date}/to/{self.config.end_date}",
            "expver": "1",
            "grid": "0.75/0.75",
            "levtype": "sfc",
            "param": "165.128/166.128/167.128/168.128/169.128/170.128/171.128/172.128",  # 表面变量
            "step": "0",
            "stream": "oper",
            "time": "00:00:00/06:00:00/12:00:00/18:00:00",
            "type": "an",
            "format": "netcdf",
            "area": f"{self.config.lat_max}/{self.config.lon_min}/{self.config.lat_min}/{self.config.lon_max}",
        }

        output_file = self.dirs['atmos'] / "era_interim_surface.nc"

        try:
            server.retrieve(request_params, str(output_file))
            self.logger.info("ERA-Interim数据下载成功")
            return True
        except Exception as e:
            self.logger.error(f"ERA-Interim数据下载失败: {str(e)}")
            return False

    def download_gebco_bathymetry(self) -> bool:
        """下载GEBCO海底地形数据"""
        if not self.config.download_bathy:
            return True

        self.logger.info("开始下载GEBCO海底地形数据...")

        # GEBCO 2023网格数据
        url = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/"
        filename = "GEBCO_2023.nc"

        # 对于测试，下载区域子集
        gebco_url = f"https://download.gebco.net/api/v1/coverage/gebco_2023.nc?" + \
                    f"bbox={self.config.lon_min},{self.config.lat_min}," + \
                    f"{self.config.lon_max},{self.config.lat_max}"

        output_path = self.dirs['bathy'] / filename

        if output_path.exists():
            self.logger.info("GEBCO数据已存在，跳过下载")
            return True

        success = self.download_file(gebco_url, output_path)

        if success:
            self.logger.info("GEBCO海底地形数据下载成功")

        return success

    def get_aviso_credentials(self) -> Optional[Tuple[str, str]]:
        """获取AVISO认证信息"""
        try:
            # 首先尝试环境变量
            username = os.environ.get('AVISO_USERNAME')
            password = os.environ.get('AVISO_PASSWORD')

            if username and password:
                return (username, password)

            # 尝试.netrc文件
            netrc_auth = netrc.netrc()
            auth_data = netrc_auth.authenticators('my.aviso.altimetry.fr')
            if auth_data:
                return (auth_data[0], auth_data[2])

        except Exception as e:
            self.logger.warning(f"无法获取AVISO认证信息: {str(e)}")

        return None

    def get_earthdata_credentials(self) -> Optional[Tuple[str, str]]:
        """获取NASA EarthData认证信息"""
        try:
            username = os.environ.get('EARTHDATA_USERNAME')
            password = os.environ.get('EARTHDATA_PASSWORD')

            if username and password:
                return (username, password)

            # 尝试.netrc文件
            netrc_auth = netrc.netrc()
            auth_data = netrc_auth.authenticators('urs.earthdata.nasa.gov')
            if auth_data:
                return (auth_data[0], auth_data[2])

        except Exception as e:
            self.logger.warning(f"无法获取EarthData认证信息: {str(e)}")

        return None

    def get_argo_profiles_in_region(self, ftp: ftplib.FTP, base_path: str) -> List[str]:
        """获取区域内的Argo剖面列表"""
        # 简化实现：返回示例剖面路径
        # 实际实现需要解析Argo索引文件

        sample_profiles = [
            f"{base_path}/csio/2902746/profiles/D2902746_001.nc",
            f"{base_path}/csio/2902746/profiles/D2902746_002.nc",
            f"{base_path}/bodc/6900772/profiles/D6900772_001.nc",
            f"{base_path}/bodc/6900772/profiles/D6900772_002.nc",
            f"{base_path}/nmdis/2902115/profiles/D2902115_001.nc"
        ]

        return sample_profiles

    def save_download_summary(self):
        """保存下载摘要"""
        summary = {
            'download_config': {
                'time_range': f"{self.config.start_date} to {self.config.end_date}",
                'spatial_range': {
                    'lat': [self.config.lat_min, self.config.lat_max],
                    'lon': [self.config.lon_min, self.config.lon_max]
                }
            },
            'download_statistics': self.download_stats,
            'data_inventory': {
                'sea_level_anomaly': list(self.dirs['sla'].glob('*.nc')),
                'sea_surface_temperature': list(self.dirs['sst'].glob('*.nc')),
                'sea_ice': list(self.dirs['ice'].glob('*')),
                'argo_profiles': list(self.dirs['argo'].glob('*.nc')),
                'atmospheric_forcing': list(self.dirs['atmos'].glob('*.nc')),
                'bathymetry': list(self.dirs['bathy'].glob('*.nc'))
            },
            'timestamp': datetime.now().isoformat()
        }

        # 转换Path对象为字符串
        for category in summary['data_inventory']:
            summary['data_inventory'][category] = [str(p) for p in summary['data_inventory'][category]]

        summary_file = Path(self.config.base_output_dir) / "download_summary.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"下载摘要已保存: {summary_file}")

    def run_complete_download(self) -> bool:
        """执行完整的数据下载流程"""
        self.logger.info("开始执行完整的海洋数据下载流程...")

        start_time = time.time()

        # 执行各类数据下载
        download_results = {
            'sla': self.download_aviso_sla_data(),
            'sst': self.download_reynolds_sst_data(),
            'ice': self.download_nsidc_ice_data(),
            'argo': self.download_argo_profiles(),
            'atmos': self.download_era_interim_data(),
            'bathy': self.download_gebco_bathymetry()
        }

        end_time = time.time()
        total_time = end_time - start_time

        # 统计结果
        successful_categories = sum(1 for success in download_results.values() if success)
        total_categories = len(download_results)

        self.logger.info(f"数据下载完成!")
        self.logger.info(f"总耗时: {total_time:.2f} 秒")
        self.logger.info(f"成功类别: {successful_categories}/{total_categories}")
        self.logger.info(f"下载统计: {self.download_stats}")

        # 保存下载摘要
        self.save_download_summary()

        return successful_categories == total_categories


def main():
    """主函数"""
    print("TOPAZ系统海洋数据自动下载器")
    print("=" * 50)

    # 创建配置
    config = DownloadConfig()

    # 允许命令行参数覆盖配置
    if len(sys.argv) > 1:
        config.base_output_dir = sys.argv[1]

    # 创建下载器并执行下载
    downloader = OceanDataDownloader(config)

    try:
        success = downloader.run_complete_download()

        if success:
            print("\n✅ 所有数据下载成功完成!")
            print(f"数据保存目录: {config.base_output_dir}")
        else:
            print("\n⚠️  部分数据下载失败，请检查日志文件")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n❌ 用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 下载过程发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()