# ==============================================================================
# simulation/current_simulation_wrapper.py  
# ==============================================================================
"""
洋流模拟包装器 - 调用C++洋流场求解器
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import netCDF4
import os


try:
    import oceansim
except ImportError:
    logging.warning("Oceansim package not found, using Python implementation instead.")
    oceansim = None


class CurrentSimulationWrapper:
    """洋流模拟包装器"""

    def __init__(self, domain_config: Dict[str, Any], solver_config: Optional[Dict] = None):
        """
        初始化洋流模拟器
        
        Args:
            domain_config: 计算域配置
            solver_config: 求解器配置
        """
        self.domain_config = domain_config
        self.solver_config = solver_config or {}
        self._cpp_solver = None

        # 自动计算 coriolis
        # ==========================================================
        # 自动计算 coriolis + dx/dy/dz
        # ==========================================================
        nc_file = self.domain_config.get('nc_file')
        if nc_file and os.path.exists(nc_file):
            try:
                with netCDF4.Dataset(nc_file) as ds:
                    # 经纬度
                    lat = ds.variables['lat'][:]
                    lon = ds.variables['lon'][:]
                    # 中心纬度
                    lat_center = (np.min(lat) + np.max(lat)) / 2
                    omega = 7.2921e-5
                    coriolis = 2 * omega * np.sin(np.radians(lat_center))
        
                    if 'coriolis' not in self.solver_config:
                        self.solver_config['coriolis'] = coriolis
                        logging.info(f"自动计算 coriolis={coriolis:.6e} s^-1 based on nc")
        
                    # 计算 dx/dy
                    if len(lat) > 1:
                        dy = abs(lat[1] - lat[0]) * np.pi/180 * 6371000
                    else:
                        dy = 0
        
                    if len(lon) > 1:
                        dx = abs(lon[1] - lon[0]) * np.pi/180 * 6371000 * np.cos(np.radians(lat_center))
                    else:
                        dx = 0
        
                    depth = ds.variables.get('depth')
                    if depth is not None and len(depth) > 1:
                        dz = abs(depth[1] - depth[0])
                    else:
                        dz = 0
        
                    self.domain_config['dx'] = dx
                    self.domain_config['dy'] = dy
                    self.domain_config['dz'] = dz
        
                    logging.info(f"自动计算 dx={dx:.2f} m dy={dy:.2f} m dz={dz:.2f} m from nc")
        
            except Exception as e:
                logging.warning(f"从 nc_file 自动计算失败，将使用默认值: {e}")
        
        else:
            if 'coriolis' not in self.solver_config:
                logging.warning("未提供 nc_file 且未传入 coriolis，默认 coriolis=0")
                self.solver_config['coriolis'] = 0.0



        if oceansim is not None:
            self._init_cpp_solver()

    def _init_cpp_solver(self):
        """初始化C++洋流求解器"""
        try:
            # 创建网格
            grid = oceansim.GridDataStructure(
                self.domain_config.get('nx', 100),
                self.domain_config.get('ny', 100),
                self.domain_config.get('nz', 50)
            )

            # 创建洋流场求解器
            self._cpp_solver = oceansim.CurrentFieldSolver(grid)

            # 设置求解器参数
            if 'viscosity' in self.solver_config:
                self._cpp_solver.setViscosity(self.solver_config['viscosity'])

            if 'coriolis' in self.solver_config:
                self._cpp_solver.setCoriolisParameter(self.solver_config['coriolis'])

        except Exception as e:
            logging.error(f"Failed to initialize C++ current solver: {e}")
            self._cpp_solver = None

    def solve_momentum_equations(self,
                                 boundary_conditions: Dict[str, Any],
                                 forcing_terms: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解动量方程
        
        Args:
            boundary_conditions: 边界条件
            forcing_terms: 强迫项
            
        Returns:
            (u, v, w) 速度分量
        """
        if self._cpp_solver:
            # 使用C++求解器
            return self._solve_cpp(boundary_conditions, forcing_terms)
        else:
            # Python备用实现
            return self._solve_python(boundary_conditions, forcing_terms)

    def _solve_cpp(self, boundary_conditions: Dict, forcing_terms: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """C++求解器实现"""
        try:
            # 设置边界条件
            if 'surface_wind' in boundary_conditions:
                wind = boundary_conditions['surface_wind']
                self._cpp_solver.setSurfaceWind(wind['u'], wind['v'])

            # 设置强迫项
            if forcing_terms is not None:
                self._cpp_solver.setForcingTerms(forcing_terms.tolist())

            # 求解
            result = self._cpp_solver.solve()

            # 提取速度分量
            u = np.array(result.getVelocityU())
            v = np.array(result.getVelocityV())
            w = np.array(result.getVelocityW())

            return u, v, w

        except Exception as e:
            logging.error(f"C++ solver failed: {e}")
            return self._solve_python(boundary_conditions, forcing_terms)

    def _solve_python(self, boundary_conditions: Dict, forcing_terms: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Python备用求解器"""
        # 简化的有限差分求解
        nx, ny, nz = self.domain_config.get('nx', 100), self.domain_config.get('ny', 100), self.domain_config.get('nz', 50)

        # 初始化速度场
        u = np.zeros((nx, ny, nz))
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))

        # 应用边界条件
        if 'surface_wind' in boundary_conditions:
            wind = boundary_conditions['surface_wind']
            u[:, :, -1] = wind.get('u', 0.0)
            v[:, :, -1] = wind.get('v', 0.0)

        # 简化的求解（实际中需要迭代求解Navier-Stokes方程）
        logging.info("Using simplified Python current solver")

        return u, v, w

    def get_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """计算涡度"""
        if self._cpp_solver:
            try:
                return np.array(self._cpp_solver.computeVorticity(u.tolist(), v.tolist()))
            except:
                pass

        # Python计算涡度
        dy, dx = 1.0, 1.0  # 简化网格间距
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        return dvdx - dudy

# ==============================================================================
# current_simulation_wrapper.py 测试代码
# ==============================================================================

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    domain_config = {
        'nx': 40, 'ny': 40, 'nz': 10,
        'dx': 1.0, 'dy': 1.0, 'dz': 1.0,
        "nc_file" : "/Users/beilsmindex/洋流模拟/OceanCurrentSimulationSystem/Source/PythonEngine/data/raw_data/merged_data.nc"
    }

    solver_config = {
        'viscosity': 0.01,
        # 'coriolis': 1e-4
    }

    # 创建洋流模拟器
    current_sim = CurrentSimulationWrapper(domain_config, solver_config)

    # 设置边界条件
    boundary_conditions = {
        'surface_wind': {
            'u': 10.0,  # 10 m/s 东向风
            'v': 5.0    # 5 m/s 北向风
        }
    }

    # 创建强迫项（可选）
    forcing = np.random.normal(0, 0.1, (domain_config['nx'], domain_config['ny'], domain_config['nz']))

    print("开始洋流模拟...")

    # 求解动量方程
    u, v, w = current_sim.solve_momentum_equations(boundary_conditions, forcing)

    print(f"速度场计算完成:")
    print(f"U分量范围: [{np.min(u):.3f}, {np.max(u):.3f}] m/s")
    print(f"V分量范围: [{np.min(v):.3f}, {np.max(v):.3f}] m/s")
    print(f"W分量范围: [{np.min(w):.3f}, {np.max(w):.3f}] m/s")

    # 计算涡度
    vorticity = current_sim.get_vorticity(u[:, :, -1], v[:, :, -1])  # 表面涡度

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 表面流速
    im1 = axes[0, 0].quiver(u[:, :, -1], v[:, :, -1], scale=50)
    axes[0, 0].set_title('表面流速场')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')

    # U分量
    im2 = axes[0, 1].contourf(u[:, :, -1], levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('U分量 (东向)')
    plt.colorbar(im2, ax=axes[0, 1])

    # V分量
    im3 = axes[1, 0].contourf(v[:, :, -1], levels=20, cmap='RdBu_r')
    axes[1, 0].set_title('V分量 (北向)')
    plt.colorbar(im3, ax=axes[1, 0])

    # 涡度
    im4 = axes[1, 1].contourf(vorticity, levels=20, cmap='RdBu_r')
    axes[1, 1].set_title('相对涡度')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # 统计信息
    speed = np.sqrt(u**2 + v**2)
    print(f"\n统计信息:")
    print(f"最大流速: {np.max(speed):.3f} m/s")
    print(f"平均流速: {np.mean(speed):.3f} m/s")
    print(f"最大涡度: {np.max(np.abs(vorticity)):.6f} s⁻¹")
