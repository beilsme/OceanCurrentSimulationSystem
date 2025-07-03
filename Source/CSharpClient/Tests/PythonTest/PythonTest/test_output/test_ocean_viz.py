
import json
import sys
import os
import numpy as np

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    # 生成模拟的海流数据
    def generate_ocean_current_data():
        # 创建网格 - 模拟南海区域
        lat = np.linspace(21, 26.5, 20)
        lon = np.linspace(118, 124, 25)
        LAT, LON = np.meshgrid(lat, lon, indexing='ij')

        # 生成涡旋流场
        center_lat, center_lon = 23.75, 121.0
        dx = LON - center_lon
        dy = LAT - center_lat
        r = np.sqrt(dx**2 + dy**2)

        # 避免除零
        r = np.where(r < 0.01, 0.01, r)

        # 涡旋强度
        strength = 0.5 * np.exp(-r * 2)

        # 速度分量
        u = -dy / r * strength
        v = dx / r * strength

        return lat, lon, u, v

    # 生成数据
    lat, lon, u, v = generate_ocean_current_data()

    # 创建可视化
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算速度大小
    speed = np.sqrt(u**2 + v**2)

    # 绘制速度场
    skip = 2
    X, Y = np.meshgrid(lon[::skip], lat[::skip])
    U = u[::skip, ::skip]
    V = v[::skip, ::skip]

    # 背景颜色图
    im = ax.contourf(X, Y, speed[::skip, ::skip], levels=20, cmap='coolwarm', alpha=0.7)
    cbar = plt.colorbar(im, ax=ax, label='流速 (m/s)')

    # 矢量箭头
    ax.quiver(X, Y, U, V, scale=10, width=0.003, color='black', alpha=0.8)

    ax.set_xlabel('经度 (°E)', fontsize=12)
    ax.set_ylabel('纬度 (°N)', fontsize=12)
    ax.set_title('模拟海流场可视化测试', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加地理信息
    ax.text(0.02, 0.98, '南海区域', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 保存图像
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'ocean_current_test.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    # 返回结果
    result = {
        'success': True,
        'message': '海流数据可视化测试成功',
        'image_path': output_path,
        'data_shape': f'{u.shape}',
        'lon_range': f'{lon.min():.2f} - {lon.max():.2f}',
        'lat_range': f'{lat.min():.2f} - {lat.max():.2f}',
        'max_speed': f'{speed.max():.3f}',
        'min_speed': f'{speed.min():.3f}',
        'file_exists': os.path.exists(output_path)
    }

    print(json.dumps(result, indent=2))

except Exception as e:
    result = {
        'success': False,
        'message': f'海流可视化失败: {str(e)}',
        'error_type': type(e).__name__
    }
    print(json.dumps(result, indent=2))
