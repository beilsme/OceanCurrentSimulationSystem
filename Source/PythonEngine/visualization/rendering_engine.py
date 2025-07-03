# ==============================================================================
# visualization/rendering_engine.py
# ==============================================================================
"""
渲染引擎 - 高级3D渲染和交互式可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patches as patches
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import sys

# 导入中文配置
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from PythonEngine.utils.chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    chinese_config = None


class RenderingEngine:
    """高级海洋数据渲染引擎"""

    def __init__(self, grid_config: Optional[Dict[str, Any]] = None, chinese_support: bool = True):
        """
        初始化渲染引擎
        
        Args:
             grid_config: 网格配置，如果为 ``None`` 则使用默认配置
            chinese_support: 中文支持
        """

        if grid_config is None:
            grid_config = {
                'nx': 100, 'ny': 100, 'nz': 20,
                'dx': 1.0, 'dy': 1.0, 'dz': 1.0
            }
        self.grid_config = grid_config
        self.nx = grid_config.get('nx', 100)
        self.ny = grid_config.get('ny', 100)
        self.nz = grid_config.get('nz', 20)
        self.dx = grid_config.get('dx', 1.0)
        self.dy = grid_config.get('dy', 1.0)
        self.dz = grid_config.get('dz', 1.0)

        # 坐标网格
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.z = np.linspace(0, self.nz * self.dz, self.nz)

        # 中文支持
        if chinese_support and chinese_config:
            self.font_config = chinese_config.setup_chinese_support()
            allowed_keys = {"family", "size", "weight", "color"}
            self.font_config = {k: v for k, v in self.font_config.items() if k in allowed_keys}

        else:
            self.font_config = {}

        # 自定义颜色映射
        self._setup_custom_colormaps()

    def _setup_custom_colormaps(self):
        """设置自定义颜色映射"""
        # 海洋主题颜色映射
        ocean_colors = ['#000080', '#0066CC', '#00AAFF', '#66CCFF', '#CCEEAA', '#FFFF99']
        self.ocean_cmap = LinearSegmentedColormap.from_list('ocean', ocean_colors)

        # 污染物颜色映射
        pollution_colors = ['#FFFFFF', '#FFEEAA', '#FFCC66', '#FF9933', '#FF6600', '#CC0000']
        self.pollution_cmap = LinearSegmentedColormap.from_list('pollution', pollution_colors)

        # 深度颜色映射
        depth_colors = ['#E6F3FF', '#99CCFF', '#3399FF', '#0066CC', '#003399', '#000066']
        self.depth_cmap = LinearSegmentedColormap.from_list('depth', depth_colors)

    def render_3d_velocity_field(self,
                                 velocity_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 title: str = "3D海流速度场",
                                 slice_positions: Optional[Dict[str, int]] = None,
                                 streamlines: bool = True,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        渲染3D速度场
        
        Args:
            velocity_data: (u, v, w) 速度分量
            title: 图表标题
            slice_positions: 切片位置 {'x': idx, 'y': idx, 'z': idx}
            streamlines: 是否显示流线
            save_path: 保存路径
        """
        u, v, w = velocity_data

        fig = plt.figure(figsize=(16, 12))

        # 主3D视图
        ax_main = fig.add_subplot(2, 2, (1, 2), projection='3d')

        # 设置切片位置
        if slice_positions is None:
            slice_positions = {
                'x': self.nx // 2,
                'y': self.ny // 2,
                'z': self.nz - 1  # 表层
            }

        # 创建3D网格
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # 绘制速度幅值等值面
        speed = np.sqrt(u**2 + v**2 + w**2)

        # X切片
        if 'x' in slice_positions:
            x_idx = slice_positions['x']
            Y_slice, Z_slice = np.meshgrid(self.y, self.z, indexing='ij')
            speed_x = speed[x_idx, :, :].T
            ax_main.contourf(np.full_like(Y_slice, self.x[x_idx]), Y_slice, Z_slice,
                             speed_x, levels=20, cmap=self.ocean_cmap, alpha=0.8)

        # Y切片
        if 'y' in slice_positions:
            y_idx = slice_positions['y']
            X_slice, Z_slice = np.meshgrid(self.x, self.z, indexing='ij')
            speed_y = speed[:, y_idx, :].T
            ax_main.contourf(X_slice, np.full_like(X_slice, self.y[y_idx]), Z_slice,
                             speed_y, levels=20, cmap=self.ocean_cmap, alpha=0.8)

        # Z切片 (表层)
        if 'z' in slice_positions:
            z_idx = slice_positions['z']
            X_slice, Y_slice = np.meshgrid(self.x, self.y, indexing='ij')
            speed_z = speed[:, :, z_idx]
            ax_main.contourf(X_slice, Y_slice, np.full_like(X_slice, self.z[z_idx]),
                             speed_z, levels=20, cmap=self.ocean_cmap, alpha=0.8)

        # 添加流线
        if streamlines:
            # 简化的流线显示
            skip = max(1, min(self.nx, self.ny, self.nz) // 10)
            ax_main.quiver(X[::skip, ::skip, ::skip], Y[::skip, ::skip, ::skip], Z[::skip, ::skip, ::skip],
                           u[::skip, ::skip, ::skip], v[::skip, ::skip, ::skip], w[::skip, ::skip, ::skip],
                           length=0.1, normalize=True, alpha=0.6, color='red')

        ax_main.set_xlabel('X (m)', **self.font_config)
        ax_main.set_ylabel('Y (m)', **self.font_config)
        ax_main.set_zlabel('Z (m)', **self.font_config)
        ax_main.set_title('3D速度场', **self.font_config)

        # 表层流场
        ax_surface = fig.add_subplot(2, 2, 3)
        surface_speed = speed[:, :, -1]
        X_surf, Y_surf = np.meshgrid(self.x, self.y, indexing='ij')

        cs = ax_surface.contourf(X_surf, Y_surf, surface_speed, levels=20, cmap=self.ocean_cmap)
        skip_surf = max(1, min(self.nx, self.ny) // 15)
        ax_surface.quiver(X_surf[::skip_surf, ::skip_surf], Y_surf[::skip_surf, ::skip_surf],
                          u[::skip_surf, ::skip_surf, -1], v[::skip_surf, ::skip_surf, -1],
                          scale=50, alpha=0.8)

        plt.colorbar(cs, ax=ax_surface, label='速度 (m/s)')
        ax_surface.set_xlabel('X (m)', **self.font_config)
        ax_surface.set_ylabel('Y (m)', **self.font_config)
        ax_surface.set_title('表层流场', **self.font_config)

        # 垂直剖面
        ax_profile = fig.add_subplot(2, 2, 4)
        y_mid = self.ny // 2
        Z_prof, X_prof = np.meshgrid(self.z, self.x, indexing='ij')
        profile_speed = speed[:, y_mid, :].T

        cs_prof = ax_profile.contourf(X_prof, Z_prof, profile_speed, levels=20, cmap=self.ocean_cmap)
        plt.colorbar(cs_prof, ax=ax_profile, label='速度 (m/s)')
        ax_profile.set_xlabel('X (m)', **self.font_config)
        ax_profile.set_ylabel('深度 (m)', **self.font_config)
        ax_profile.set_title('垂直剖面', **self.font_config)
        ax_profile.invert_yaxis()  # 深度向下

        fig.suptitle(title, **self.font_config, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"3D速度场图保存至: {save_path}")

        return fig

    def render_3d_concentration_field(self,
                                      concentration_data: np.ndarray,
                                      title: str = "3D污染物浓度场",
                                      isosurface_levels: Optional[List[float]] = None,
                                      transparency: float = 0.7,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        渲染3D浓度场
        
        Args:
            concentration_data: 浓度数据
            title: 图表标题
            isosurface_levels: 等值面级别
            transparency: 透明度
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 12))

        # 主3D视图
        ax_main = fig.add_subplot(2, 2, (1, 2), projection='3d')

        # 计算等值面级别
        if isosurface_levels is None:
            max_conc = np.max(concentration_data)
            isosurface_levels = [max_conc * 0.1, max_conc * 0.5, max_conc * 0.9]

        # 创建3D网格
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

        # 绘制等值面
        colors = ['blue', 'green', 'red']
        for i, level in enumerate(isosurface_levels):
            if level > 0:
                # 简化的等值面显示 - 使用切片代替真正的等值面
                mask = concentration_data >= level

                # 在每个方向选择切片显示
                for z_idx in range(0, self.nz, max(1, self.nz // 5)):
                    if np.any(mask[:, :, z_idx]):
                        X_slice, Y_slice = np.meshgrid(self.x, self.y, indexing='ij')
                        masked_conc = np.ma.masked_where(~mask[:, :, z_idx],
                                                         concentration_data[:, :, z_idx])
                        if not masked_conc.mask.all():
                            ax_main.contourf(X_slice, Y_slice,
                                             np.full_like(X_slice, self.z[z_idx]),
                                             masked_conc, levels=[level, np.max(concentration_data)],
                                             colors=[colors[i]], alpha=transparency)

        ax_main.set_xlabel('X (m)', **self.font_config)
        ax_main.set_ylabel('Y (m)', **self.font_config)
        ax_main.set_zlabel('Z (m)', **self.font_config)
        ax_main.set_title('3D浓度等值面', **self.font_config)

        # 表层浓度分布
        ax_surface = fig.add_subplot(2, 2, 3)
        surface_conc = concentration_data[:, :, -1]
        X_surf, Y_surf = np.meshgrid(self.x, self.y, indexing='ij')

        cs = ax_surface.contourf(X_surf, Y_surf, surface_conc, levels=20, cmap=self.pollution_cmap)
        plt.colorbar(cs, ax=ax_surface, label='浓度 (kg/m³)')
        ax_surface.set_xlabel('X (m)', **self.font_config)
        ax_surface.set_ylabel('Y (m)', **self.font_config)
        ax_surface.set_title('表层浓度', **self.font_config)

        # 垂直积分浓度
        ax_integrated = fig.add_subplot(2, 2, 4)
        integrated_conc = np.sum(concentration_data, axis=2) * self.dz

        cs_int = ax_integrated.contourf(X_surf, Y_surf, integrated_conc, levels=20, cmap=self.pollution_cmap)
        plt.colorbar(cs_int, ax=ax_integrated, label='积分浓度 (kg/m²)')
        ax_integrated.set_xlabel('X (m)', **self.font_config)
        ax_integrated.set_ylabel('Y (m)', **self.font_config)
        ax_integrated.set_title('垂直积分浓度', **self.font_config)

        fig.suptitle(title, **self.font_config, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"3D浓度场图保存至: {save_path}")

        return fig

    def render_comprehensive_dashboard(self,
                                       velocity_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                       concentration_data: np.ndarray,
                                       particle_positions: np.ndarray,
                                       time_info: Dict[str, Any],
                                       statistics: Dict[str, Any],
                                       title: str = "海洋模拟综合仪表板",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        渲染综合仪表板
        
        Args:
            velocity_data: 速度场数据
            concentration_data: 浓度场数据
            particle_positions: 粒子位置
            time_info: 时间信息
            statistics: 统计信息
            title: 仪表板标题
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 14))

        u, v, w = velocity_data

        # 1. 表层流场 (左上)
        ax1 = plt.subplot(3, 4, (1, 2))
        speed_surface = np.sqrt(u[:, :, -1]**2 + v[:, :, -1]**2)
        X_surf, Y_surf = np.meshgrid(self.x, self.y, indexing='ij')

        cs1 = ax1.contourf(X_surf, Y_surf, speed_surface, levels=20, cmap=self.ocean_cmap)
        skip = max(1, min(self.nx, self.ny) // 20)
        ax1.quiver(X_surf[::skip, ::skip], Y_surf[::skip, ::skip],
                   u[::skip, ::skip, -1], v[::skip, ::skip, -1], scale=100, alpha=0.7)
        plt.colorbar(cs1, ax=ax1, label='速度 (m/s)')
        ax1.set_title('表层海流', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)

        # 2. 浓度分布 (右上)
        ax2 = plt.subplot(3, 4, (3, 4))
        conc_surface = concentration_data[:, :, -1]
        cs2 = ax2.contourf(X_surf, Y_surf, conc_surface, levels=20, cmap=self.pollution_cmap)
        plt.colorbar(cs2, ax=ax2, label='浓度 (kg/m³)')
        ax2.set_title('表层污染物浓度', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)

        # 3. 粒子分布 (左中)
        ax3 = plt.subplot(3, 4, (5, 6))
        if len(particle_positions) > 0:
            x_particles = particle_positions[:, 0] * self.dx
            y_particles = particle_positions[:, 1] * self.dy
            z_particles = particle_positions[:, 2] * self.dz if particle_positions.shape[1] > 2 else None

            scatter = ax3.scatter(x_particles, y_particles,
                                  c=z_particles if z_particles is not None else 'red',
                                  cmap=self.depth_cmap, s=15, alpha=0.8)
            if z_particles is not None:
                plt.colorbar(scatter, ax=ax3, label='深度 (m)')

        ax3.set_title('粒子分布', **self.font_config)
        ax3.set_xlabel('X (m)', **self.font_config)
        ax3.set_ylabel('Y (m)', **self.font_config)

        # 4. 垂直剖面 (右中)
        ax4 = plt.subplot(3, 4, (7, 8))
        y_mid = self.ny // 2
        Z_prof, X_prof = np.meshgrid(self.z, self.x, indexing='ij')
        conc_profile = concentration_data[:, y_mid, :].T

        cs4 = ax4.contourf(X_prof, Z_prof, conc_profile, levels=20, cmap=self.pollution_cmap)
        plt.colorbar(cs4, ax=ax4, label='浓度 (kg/m³)')
        ax4.set_title('垂直剖面 (中线)', **self.font_config)
        ax4.set_xlabel('X (m)', **self.font_config)
        ax4.set_ylabel('深度 (m)', **self.font_config)
        ax4.invert_yaxis()

        # 5. 统计信息 (左下)
        ax5 = plt.subplot(3, 4, 9)
        ax5.axis('off')

        # 格式化统计信息
        stats_text = f"""模拟统计信息
        
时间: {time_info.get('current_time', 0):.2f} h
时间步长: {time_info.get('dt', 0):.3f} h

速度统计:
最大流速: {np.max(np.sqrt(u**2 + v**2 + w**2)):.3f} m/s
平均流速: {np.mean(np.sqrt(u**2 + v**2)):.3f} m/s

浓度统计:
最大浓度: {np.max(concentration_data):.2e} kg/m³
总质量: {np.sum(concentration_data):.2e} kg

粒子统计:
活跃粒子数: {len(particle_positions)}
"""

        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                 verticalalignment='top', **self.font_config,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 6. 时间序列图 (中下)
        ax6 = plt.subplot(3, 4, 10)
        if 'time_series' in statistics:
            time_data = statistics['time_series']
            ax6.plot(time_data.get('time', []), time_data.get('max_speed', []),
                     'b-', label='最大流速')
            ax6.set_xlabel('时间 (h)', **self.font_config)
            ax6.set_ylabel('流速 (m/s)', **self.font_config)
            ax6.set_title('流速时间演化', **self.font_config)
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, '无时间序列数据', ha='center', va='center',
                     transform=ax6.transAxes, **self.font_config)
            ax6.set_title('流速时间演化', **self.font_config)

        # 7. 浓度分布直方图 (右下)
        ax7 = plt.subplot(3, 4, 11)
        valid_conc = concentration_data[concentration_data > 0]
        if len(valid_conc) > 0:
            ax7.hist(valid_conc, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax7.set_title('浓度分布', **self.font_config)

        # 8. 性能监控 (最右下)
        ax8 = plt.subplot(3, 4, 12)
        if 'performance' in statistics:
            perf_data = statistics['performance']
            labels = list(perf_data.keys())
            values = list(perf_data.values())

            bars = ax8.bar(range(len(labels)), values, alpha=0.7, color='green')
            ax8.set_xticks(range(len(labels)))
            ax8.set_xticklabels(labels, rotation=45, **self.font_config)
            ax8.set_ylabel('时间 (s)', **self.font_config)
            ax8.set_title('性能指标', **self.font_config)

            # 添加数值标签
            for bar, value in zip(bars, values):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', **self.font_config)
        else:
            ax8.text(0.5, 0.5, '无性能数据', ha='center', va='center',
                     transform=ax8.transAxes, **self.font_config)
            ax8.set_title('性能指标', **self.font_config)

        fig.suptitle(title, **self.font_config, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"综合仪表板保存至: {save_path}")

        return fig

    def create_interactive_plot(self,
                                data_dict: Dict[str, np.ndarray],
                                plot_type: str = "velocity",
                                title: str = "交互式海洋数据可视化") -> plt.Figure:
        """
        创建交互式图表
        
        Args:
            data_dict: 数据字典
            plot_type: 图表类型 ("velocity", "concentration", "particles")
            title: 图表标题
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if plot_type == "velocity" and 'u' in data_dict and 'v' in data_dict:
            u = data_dict['u'][:, :, -1]  # 表层
            v = data_dict['v'][:, :, -1]
            speed = np.sqrt(u**2 + v**2)

            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
            cs = ax.contourf(X, Y, speed, levels=20, cmap=self.ocean_cmap)

            # 添加交互式颜色条
            cbar = plt.colorbar(cs, ax=ax)
            cbar.set_label('速度 (m/s)', **self.font_config)

            # 添加矢量场
            skip = max(1, min(self.nx, self.ny) // 15)
            Q = ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                          u[::skip, ::skip], v[::skip, ::skip],
                          scale=50, alpha=0.8)

            ax.set_title(f'{title} - 速度场', **self.font_config)

        elif plot_type == "concentration" and 'concentration' in data_dict:
            conc = data_dict['concentration'][:, :, -1]
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')

            cs = ax.contourf(X, Y, conc, levels=20, cmap=self.pollution_cmap)
            cbar = plt.colorbar(cs, ax=ax)
            cbar.set_label('浓度 (kg/m³)', **self.font_config)

            ax.set_title(f'{title} - 浓度场', **self.font_config)

        elif plot_type == "particles" and 'particles' in data_dict:
            particles = data_dict['particles']
            if len(particles) > 0:
                x_particles = particles[:, 0] * self.dx
                y_particles = particles[:, 1] * self.dy
                z_particles = particles[:, 2] * self.dz if particles.shape[1] > 2 else None

                scatter = ax.scatter(x_particles, y_particles,
                                     c=z_particles if z_particles is not None else 'red',
                                     cmap=self.depth_cmap, s=20, alpha=0.8)

                if z_particles is not None:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('深度 (m)', **self.font_config)

            ax.set_title(f'{title} - 粒子分布', **self.font_config)

        ax.set_xlabel('X (m)', **self.font_config)
        ax.set_ylabel('Y (m)', **self.font_config)
        ax.grid(True, alpha=0.3)

        # 添加点击事件处理
        def on_click(event):
            if event.inaxes == ax and event.button == 1:  # 左键点击
                x_click, y_click = event.xdata, event.ydata
                if x_click is not None and y_click is not None:
                    # 转换为网格索引
                    i = int(np.clip(x_click / self.dx, 0, self.nx - 1))
                    j = int(np.clip(y_click / self.dy, 0, self.ny - 1))

                    # 显示点击位置的信息
                    info_text = f'位置: ({x_click:.1f}, {y_click:.1f}) m\n网格: ({i}, {j})'

                    if plot_type == "velocity" and 'u' in data_dict:
                        u_val = data_dict['u'][i, j, -1]
                        v_val = data_dict['v'][i, j, -1]
                        speed_val = np.sqrt(u_val**2 + v_val**2)
                        info_text += f'\n速度: {speed_val:.3f} m/s\nU: {u_val:.3f} m/s\nV: {v_val:.3f} m/s'

                    elif plot_type == "concentration" and 'concentration' in data_dict:
                        conc_val = data_dict['concentration'][i, j, -1]
                        info_text += f'\n浓度: {conc_val:.2e} kg/m³'

                    # 更新标题显示点击信息
                    ax.set_title(f'{title} - {info_text}', **self.font_config)
                    plt.draw()

        fig.canvas.mpl_connect('button_press_event', on_click)

        return fig

if __name__ == "__main__":
    # rendering_engine.py 测试
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    grid_config = {
        'nx': 50, 'ny': 50, 'nz': 20,
        'dx': 100.0, 'dy': 100.0, 'dz': 2.0
    }

    # 创建渲染引擎
    renderer = RenderingEngine(grid_config)

    # 生成复杂的3D测试数据
    print("生成3D测试数据...")

    # 创建涡旋速度场
    X = np.linspace(0, 50*100, 50)
    Y = np.linspace(0, 50*100, 50)
    Z = np.linspace(0, 20*2, 20)

    u = np.zeros((50, 50, 20))
    v = np.zeros((50, 50, 20))
    w = np.zeros((50, 50, 20))

    center_x, center_y = 25, 25
    for i in range(50):
        for j in range(50):
            for k in range(20):
                dx = i - center_x
                dy = j - center_y
                r = np.sqrt(dx**2 + dy**2)

                if r > 0:
                    # 涡旋流场
                    strength = 2.0 * np.exp(-r/15) * np.exp(-k/10)
                    u[i, j, k] = -dy/r * strength
                    v[i, j, k] = dx/r * strength
                    w[i, j, k] = 0.1 * np.sin(2*np.pi*r/20) * np.exp(-k/8)

    velocity_data = (u, v, w)

    # 生成3D速度场可视化
    print("渲染3D速度场...")
    fig1 = renderer.render_3d_velocity_field(velocity_data,
                                             title="3D涡旋海流可视化",
                                             slice_positions={'x': 25, 'y': 25, 'z': 15})
    plt.show()

    # 生成3D浓度场
    print("生成3D浓度场数据...")
    concentration = np.zeros((50, 50, 20))

    # 多个污染源
    sources = [(15, 15, 15), (35, 35, 15), (25, 40, 10)]
    strengths = [100, 80, 60]

    for (sx, sy, sz), strength in zip(sources, strengths):
        for i in range(50):
            for j in range(50):
                for k in range(20):
                    dist = np.sqrt((i-sx)**2 + (j-sy)**2 + (k-sz)**2)
                    concentration[i, j, k] += strength * np.exp(-dist**2/50)

    # 渲染3D浓度场
    print("渲染3D浓度场...")
    fig2 = renderer.render_3d_concentration_field(concentration,
                                                  title="3D污染物浓度可视化",
                                                  isosurface_levels=[20, 50, 80])
    plt.show()

    # 生成综合仪表板数据
    print("生成综合仪表板...")

    # 模拟粒子位置
    n_particles = 150
    particle_positions = np.random.uniform(0, 1, (n_particles, 3))

    # 时间和统计信息
    time_info = {
        'current_time': 1.5,
        'dt': 0.05
    }

    statistics = {
        'time_series': {
            'time': np.linspace(0, 1.5, 30),
            'max_speed': 2.0 + 0.5*np.sin(np.linspace(0, 4*np.pi, 30))
        },
        'performance': {
            '粒子追踪': 0.125,
            '洋流求解': 0.356,
            '扩散计算': 0.089,
            '可视化': 0.234
        }
    }

    # 渲染综合仪表板
    fig3 = renderer.render_comprehensive_dashboard(velocity_data, concentration,
                                                   particle_positions, time_info, statistics,
                                                   title="海洋模拟实时监控仪表板")
    plt.show()

    # 测试交互式图表
    print("创建交互式图表...")
    data_dict = {
        'u': u,
        'v': v,
        'w': w,
        'concentration': concentration,
        'particles': particle_positions
    }

    fig4 = renderer.create_interactive_plot(data_dict, plot_type="velocity",
                                            title="交互式海流可视化")
    plt.show()