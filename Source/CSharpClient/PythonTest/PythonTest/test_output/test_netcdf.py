
import json
import sys
import os

def test_netcdf_processing(nc_file_path):
    try:
        # 检查文件是否存在
        if not os.path.exists(nc_file_path):
            return {
                'success': False,
                'message': f'NetCDF文件不存在: {nc_file_path}'
            }

        # 尝试导入netCDF4
        try:
            import netCDF4 as nc
        except ImportError:
            return {
                'success': False,
                'message': '缺少netCDF4模块，请安装: pip install netCDF4'
            }

        # 读取NetCDF文件信息
        with nc.Dataset(nc_file_path, 'r') as dataset:
            # 获取基本信息
            dimensions = dict(dataset.dimensions)
            variables = list(dataset.variables.keys())

            # 获取文件大小
            file_size = os.path.getsize(nc_file_path)

            result = {
                'success': True,
                'message': 'NetCDF文件读取成功',
                'file_path': nc_file_path,
                'file_size_mb': round(file_size / (1024*1024), 2),
                'dimensions': {name: size.size for name, size in dimensions.items()},
                'variables': variables[:10],  # 只显示前10个变量
                'total_variables': len(variables)
            }

            # 如果有常见的海洋变量，尝试读取一些数据
            if 'u' in variables and 'v' in variables:
                u_var = dataset.variables['u']
                v_var = dataset.variables['v']
                result['has_velocity_data'] = True
                result['u_shape'] = list(u_var.shape)
                result['v_shape'] = list(v_var.shape)

                # 尝试读取一小部分数据
                if len(u_var.shape) >= 2:
                    u_sample = u_var[0, 0] if len(u_var.shape) == 2 else u_var[0, 0, 0]
                    v_sample = v_var[0, 0] if len(v_var.shape) == 2 else v_var[0, 0, 0]
                    result['sample_u'] = float(u_sample) if not hasattr(u_sample, 'mask') or not u_sample.mask else 'masked'
                    result['sample_v'] = float(v_sample) if not hasattr(v_sample, 'mask') or not v_sample.mask else 'masked'

            return result

    except Exception as e:
        return {
            'success': False,
            'message': f'NetCDF处理失败: {str(e)}',
            'error_type': type(e).__name__
        }

# 主程序
if __name__ == '__main__':
    nc_file = sys.argv[1] if len(sys.argv) > 1 else 'test.nc'
    result = test_netcdf_processing(nc_file)
    print(json.dumps(result, indent=2))
