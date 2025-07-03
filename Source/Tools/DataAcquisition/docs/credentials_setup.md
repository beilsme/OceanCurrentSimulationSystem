# 海洋数据下载认证配置指南

本指南将帮助您配置访问各个海洋数据源所需的认证信息。

## 1. AVISO+海平面高度数据认证

### 注册AVISO+账户
1. 访问 https://www.aviso.altimetry.fr/en/my-aviso-plus.html
2. 点击"Register"创建新账户
3. 填写个人信息和研究用途说明
4. 等待账户激活邮件(通常1-2个工作日)

### 配置认证信息
方法1：使用.netrc文件(推荐)
```bash
# 创建或编辑.netrc文件
echo "machine my.aviso.altimetry.fr login YOUR_USERNAME password YOUR_PASSWORD" >> ~/.netrc
chmod 600 ~/.netrc
```

方法2：使用环境变量
```bash
export AVISO_USERNAME="your_username"
export AVISO_PASSWORD="your_password"
```

## 2. NASA EarthData认证(用于NSIDC海冰数据)

### 注册EarthData账户
1. 访问 https://urs.earthdata.nasa.gov/users/new
2. 创建NASA EarthData账户
3. 激活账户(检查邮箱)

### 配置认证信息
方法1：使用.netrc文件
```bash
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" >> ~/.netrc
```

方法2：使用环境变量
```bash
export EARTHDATA_USERNAME="your_username"
export EARTHDATA_PASSWORD="your_password"
```

## 3. ECMWF Climate Data Store认证

### 注册CDS账户
1. 访问 https://cds.climate.copernicus.eu/user/register
2. 创建Copernicus Climate Data Store账户
3. 激活账户

### 获取API密钥
1. 登录后访问 https://cds.climate.copernicus.eu/api-how-to
2. 复制您的API URL和密钥
3. 创建配置文件：

```bash
# 创建.cdsapirc文件
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF
chmod 600 ~/.cdsapirc
```

或使用环境变量：
```bash
export CDSAPI_URL="https://cds.climate.copernicus.eu/api/v2"
export CDSAPI_KEY="YOUR_UID:YOUR_API_KEY"
```

## 4. 安装依赖包

### Python包依赖
```bash
pip install requests pyyaml netcdf4 xarray ecmwf-api-client cdsapi
```

### 系统依赖(Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libnetcdf-dev libhdf5-dev
```

### 系统依赖(CentOS/RHEL)
```bash
sudo yum install netcdf-devel hdf5-devel
```

## 5. 验证配置

### 测试脚本
```python
#!/usr/bin/env python3
import os
import netrc
from ecmwfapi import ECMWFDataServer

def test_credentials():
    print("测试认证配置...")
    
    # 测试AVISO认证
    try:
        netrc_auth = netrc.netrc()
        aviso_auth = netrc_auth.authenticators('my.aviso.altimetry.fr')
        if aviso_auth:
            print("✅ AVISO认证配置正确")
        else:
            print("❌ AVISO认证配置缺失")
    except:
        print("❌ .netrc文件配置有误")
    
    # 测试EarthData认证
    try:
        earthdata_auth = netrc_auth.authenticators('urs.earthdata.nasa.gov')
        if earthdata_auth:
            print("✅ EarthData认证配置正确")
        else:
            print("❌ EarthData认证配置缺失")
    except:
        print("❌ EarthData认证配置有误")
    
    # 测试ECMWF认证
    try:
        server = ECMWFDataServer()
        print("✅ ECMWF API配置正确")
    except:
        print("❌ ECMWF API配置有误")

if __name__ == "__main__":
    test_credentials()
```

## 6. 使用说明

### 基本使用
```bash
# 使用默认配置
python ocean_data_downloader.py

# 指定输出目录
python ocean_data_downloader.py /path/to/output/directory

# 使用自定义配置文件
python ocean_data_downloader.py --config custom_config.yaml
```

### 配置文件示例
```yaml
# 自定义配置示例
time_range:
  start_date: "2008-04-01"
  end_date: "2008-04-07"

spatial_range:
  lat_min: 60.0
  lat_max: 80.0
  lon_min: -10.0
  lon_max: 30.0

data_sources:
  sea_level_anomaly: true
  sea_surface_temperature: true
  sea_ice_concentration: false  # 跳过海冰数据
  argo_profiles: true
  atmospheric_forcing: false    # 跳过大气数据
  bathymetry: true
```

## 7. 常见问题解决

### 问题1：SSL证书错误
```bash
# 临时解决方案(不推荐用于生产环境)
export PYTHONHTTPSVERIFY=0

# 或更新证书
pip install --upgrade certifi
```

### 问题2：下载速度慢
```bash
# 设置代理(如果需要)
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="https://proxy.example.com:8080"
```

### 问题3：磁盘空间不足
```bash
# 检查可用空间
df -h

# 清理临时文件
python ocean_data_downloader.py --cleanup-only
```

## 8. 数据使用须知

### AVISO数据
- 仅限科研和教育用途
- 需要在论文中引用相关文献
- 不得用于商业目的

### NASA数据
- 数据免费使用
- 建议引用数据源
- 遵循NASA数据使用政策

### ECMWF数据
- ERA-Interim数据免费使用
- 实时数据可能需要许可证
- 商业用途需要特殊授权

## 9. 技术支持

### 联系方式
- AVISO支持: aviso@altimetry.fr
- NASA EarthData: support@earthdata.nasa.gov
- ECMWF支持: copernicus-support@ecmwf.int

### 社区资源
- AVISO论坛: https://www.aviso.altimetry.fr/en/user-corner/forums.html
- NASA Earthdata论坛: https://forum.earthdata.nasa.gov/
- ECMWF用户论坛: https://confluence.ecmwf.int/

## 10. 最佳实践

### 数据管理
1. 定期备份重要数据
2. 使用描述性的文件命名
3. 维护数据清单和元数据
4. 定期清理不需要的文件

### 下载策略
1. 避免在高峰时段大量下载
2. 使用断点续传功能
3. 监控下载状态和错误日志
4. 合理设置重试间隔

### 存储优化
1. 使用数据压缩
2. 按时间或区域组织数据
3. 定期检查数据完整性
4. 考虑使用云存储服务