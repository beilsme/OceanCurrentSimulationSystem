import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator


# 加载 nc 文件
file_path = "../data/raw_data/merged_data.nc"
nc_data = Dataset(file_path, mode='r')

# 提取变量
lat = nc_data.variables['lat'][:]
lon = nc_data.variables['lon'][:]
time = nc_data.variables['time'][:]
water_u = nc_data.variables['water_u'][:]
water_v = nc_data.variables['water_v'][:]

time_units = nc_data.variables['time'].units
calendar = nc_data.variables['time'].calendar if 'calendar' in nc_data.variables['time'].ncattrs() else 'standard'
time_readable = num2date(time, units=time_units, calendar=calendar)

# 经纬度范围设置
lon_min, lon_max = 118, 124
lat_min, lat_max = 21, 26.5

lon_mask = (lon >= lon_min) & (lon <= lon_max)
lat_mask = (lat >= lat_min) & (lat <= lat_max)
lon_filtered = lon[lon_mask]
lat_filtered = lat[lat_mask]
water_u_filtered = water_u[:, :, lat_mask, :][:, :, :, lon_mask]
water_v_filtered = water_v[:, :, lat_mask, :][:, :, :, lon_mask]

# 选择台湾海峡的起始点
lat_start = 22.5  # 台湾海峡位置
lon_start = 119.5

# 创建插值器，允许选择不同的插值方法
def create_interpolator(data, lat_filtered, lon_filtered, method='linear'):
    return RegularGridInterpolator((lat_filtered, lon_filtered), data,
                                   bounds_error=False,
                                   fill_value=0,
                                   method=method)

# 时间步长设置（3小时）
time_step = 3
total_hours = 30 * 24  # 30天总时长
num_steps = total_hours // time_step  # 每3小时一个步长

# 初始化轨迹数组
track_lat = [lat_start]
track_lon = [lon_start]
current_lat = lat_start
current_lon = lon_start

# 对每个时间步长进行模拟
for time_idx in range(len(time)):
    # 获取当前时间步的速度场
    u = water_u_filtered[time_idx, 0, :, :].copy()  # 使用表层数据
    v = water_v_filtered[time_idx, 0, :, :].copy()

    # 处理掩码数组
    if isinstance(u, np.ma.MaskedArray):
        u = u.filled(0)
    if isinstance(v, np.ma.MaskedArray):
        v = v.filled(0)

    # 创建当前时间步的插值器
    u_interp = create_interpolator(u, lat_filtered, lon_filtered, method='linear')
    v_interp = create_interpolator(v, lat_filtered, lon_filtered, method='linear')

    try:
        # 获取当前位置的流速
        current_u = float(u_interp([[current_lat, current_lon]])[0])
        current_v = float(v_interp([[current_lat, current_lon]])[0])

        # 计算位置变化（3小时的位移）
        delta_lat = current_v * time_step * 3600 / 111000  # time_step(3h) * 3600s/h，转换为度
        delta_lon = current_u * time_step * 3600 / (111000 * np.cos(np.radians(current_lat)))

        # 更新位置
        new_lat = current_lat + delta_lat
        new_lon = current_lon + delta_lon

        # 处理边界：如果小船越界，就终止模拟
        if new_lat < lat_min or new_lat > lat_max or new_lon < lon_min or new_lon > lon_max:
            print(f"Boat hit the boundary at time step {time_idx}. Ending simulation.")
            break  # 终止该船的模拟

        # 更新位置
        current_lat = new_lat
        current_lon = new_lon
        track_lat.append(current_lat)
        track_lon.append(current_lon)
        print(f"At time step {time_idx}, current_u: {current_u}, current_v: {current_v}")

    except Exception as e:
        print(f"Error at time step {time_idx}: {str(e)}")
        break

# 创建图形
plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# 添加地图要素
ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1.5)
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

# 绘制轨迹
ax.plot(track_lon, track_lat, 'r-', linewidth=2, label='Drift Trajectory', transform=ccrs.PlateCarree())
ax.plot(track_lon[0], track_lat[0], 'go', label='Start Point', markersize=10, transform=ccrs.PlateCarree())
ax.plot(track_lon[-1], track_lat[-1], 'ro', label='End Point', markersize=10, transform=ccrs.PlateCarree())

# 添加标题和图例
plt.title(f"Boat Drift Simulation (2024090100-2024093021)\nStart: ({lat_start:.2f}°N, {lon_start:.2f}°E)", pad=20)
plt.legend(loc='upper right')

plt.show()

# 关闭文件
nc_data.close()

# 打印信息
print(f"Starting Position: {track_lat[0]:.3f}°N, {track_lon[0]:.3f}°E")
print(f"Ending Position: {track_lat[-1]:.3f}°N, {track_lon[-1]:.3f}°E")
print(f"Total time steps: {len(track_lat)}")


# 计算总漂移距离
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（公里）
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * \
        np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


total_distance = haversine_distance(track_lat[0], track_lon[0], track_lat[-1], track_lon[-1])
print(f"Total drift distance: {total_distance:.2f} km")
