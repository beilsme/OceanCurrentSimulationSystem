# ==============================================================================
# visualization/animation_generator.py
# ==============================================================================
"""
动画生成器 - 创建海洋数据的时间演化动画
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from pathlib import Path
import sys
from typing import Tuple


# 导入中文配置
sys.path.append(str(Path(__file__).parent.parent / "utils"))
try:
    from chinese_config import ChineseConfig
    chinese_config = ChineseConfig()
except ImportError:
    chinese_config = None


class AnimationGenerator:
    """海洋数据动画生成器"""

    def __init__(self, grid_config: Optional[Dict[str, Any]] = None, chinese_support: bool = True):
        """
        初始化动画生成器
        
        Args:
            grid_config: 网格配置，如果为 ``None`` 则使用默认配置
            chinese_support: 中文支持
        """

        if grid_config is None:
            grid_config = {
                'nx': 100, 'ny': 100,
                'dx': 1.0, 'dy': 1.0
            }

        self.grid_config = grid_config
        self.nx = grid_config.get('nx', 100)
        self.ny = grid_config.get('ny', 100)
        self.dx = grid_config.get('dx', 1.0)
        self.dy = grid_config.get('dy', 1.0)

        # 坐标网格
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        # 中文支持
        if chinese_support and chinese_config:
            self.font_config = chinese_config.setup_chinese_support()
            allowed_keys = {"family", "size", "weight", "color"}
            self.font_config = {k: v for k, v in self.font_config.items() if k in allowed_keys}
        else:
            self.font_config = {}

    def create_velocity_animation(self,
                                  velocity_time_series: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                  time_steps: List[float],
                                  layer_index: int = -1,
                                  title: str = "海流演化动画",
                                  save_path: Optional[str] = None,
                                  fps: int = 10) -> animation.FuncAnimation:
        """
        创建速度场演化动画
        
        Args:
            velocity_time_series: 时间序列速度数据列表
            time_steps: 对应的时间步
            layer_index: 显示层级
            title: 动画标题
            save_path: 保存路径
            fps: 帧率
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 计算全局数据范围用于颜色映射
        all_speeds = []
        for u, v, w in velocity_time_series:
            speed = np.sqrt(u[:, :, layer_index]**2 + v[:, :, layer_index]**2)
            all_speeds.append(speed)

        vmin = np.min([np.min(s) for s in all_speeds])
        vmax = np.max([np.max(s) for s in all_speeds])
        norm = Normalize(vmin=vmin, vmax=vmax)

        # 初始化图形元素
        u0, v0, w0 = velocity_time_series[0]
        speed0 = np.sqrt(u0[:, :, layer_index]**2 + v0[:, :, layer_index]**2)

        # 流线图
        strm = ax1.streamplot(self.X, self.Y, u0[:, :, layer_index], v0[:, :, layer_index],
                              color=speed0, cmap='viridis', density=2, norm=norm)
        ax1.set_title('海流流线', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)

        # 速度幅值等值线
        cs = ax2.contourf(self.X, self.Y, speed0, levels=20, cmap='plasma', norm=norm)
        ax2.set_title('流速大小', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)

        # 颜色条
        cbar = plt.colorbar(cs, ax=ax2)
        cbar.set_label('速度 (m/s)', **self.font_config)

        # 时间显示
        time_text = fig.suptitle(f'{title} - 时间: {time_steps[0]:.2f}h',
                                 **self.font_config, fontsize=14)

        def animate(frame):
            """动画更新函数"""
            # 清除之前的图形
            ax1.clear()
            ax2.clear()

            # 获取当前帧数据
            u, v, w = velocity_time_series[frame]
            u_layer = u[:, :, layer_index]
            v_layer = v[:, :, layer_index]
            speed = np.sqrt(u_layer**2 + v_layer**2)

            # 更新流线图
            strm = ax1.streamplot(self.X, self.Y, u_layer, v_layer,
                                  color=speed, cmap='viridis', density=2, norm=norm)
            ax1.set_title('海流流线', **self.font_config)
            ax1.set_xlabel('X (m)', **self.font_config)
            ax1.set_ylabel('Y (m)', **self.font_config)

            # 更新等值线图
            cs = ax2.contourf(self.X, self.Y, speed, levels=20, cmap='plasma', norm=norm)
            ax2.set_title('流速大小', **self.font_config)
            ax2.set_xlabel('X (m)', **self.font_config)
            ax2.set_ylabel('Y (m)', **self.font_config)

            # 更新时间
            time_text.set_text(f'{title} - 时间: {time_steps[frame]:.2f}h')

            return []

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(velocity_time_series),
                                       interval=1000//fps, blit=False, repeat=True)

        plt.tight_layout()

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            else:
                anim.save(save_path + '.gif', writer='pillow', fps=fps)
            logging.info(f"速度场动画保存至: {save_path}")

        return anim

    def create_concentration_animation(self,
                                       concentration_time_series: List[np.ndarray],
                                       time_steps: List[float],
                                       layer_index: int = -1,
                                       title: str = "污染物扩散动画",
                                       colormap: str = 'YlOrRd',
                                       log_scale: bool = False,
                                       save_path: Optional[str] = None,
                                       fps: int = 10) -> animation.FuncAnimation:
        """
        创建浓度场演化动画
        
        Args:
            concentration_time_series: 浓度时间序列
            time_steps: 时间步
            layer_index: 显示层级
            title: 动画标题
            colormap: 颜色映射
            log_scale: 对数刻度
            save_path: 保存路径
            fps: 帧率
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 计算全局数据范围
        all_concentrations = [conc[:, :, layer_index] for conc in concentration_time_series]

        if log_scale:
            # 对数刻度处理
            valid_concs = [conc[conc > 0] for conc in all_concentrations]
            if any(len(c) > 0 for c in valid_concs):
                vmin = np.log10(np.min([np.min(c) for c in valid_concs if len(c) > 0]))
                vmax = np.log10(np.max([np.max(c) for c in all_concentrations]))
            else:
                vmin, vmax = 0, 1
        else:
            vmin = np.min([np.min(c) for c in all_concentrations])
            vmax = np.max([np.max(c) for c in all_concentrations])

        norm = Normalize(vmin=vmin, vmax=vmax)

        # 初始化图形
        conc0 = all_concentrations[0]
        if log_scale and np.any(conc0 > 0):
            conc0_plot = np.log10(conc0 + 1e-10)
        else:
            conc0_plot = conc0

        # 等值线图
        cs = ax1.contourf(self.X, self.Y, conc0_plot, levels=20, cmap=colormap, norm=norm)
        ax1.set_title('浓度分布', **self.font_config)
        ax1.set_xlabel('X (m)', **self.font_config)
        ax1.set_ylabel('Y (m)', **self.font_config)

        # 图像显示
        im = ax2.imshow(conc0_plot.T, origin='lower', cmap=colormap, norm=norm,
                        extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]])
        ax2.set_title('浓度图像', **self.font_config)
        ax2.set_xlabel('X (m)', **self.font_config)
        ax2.set_ylabel('Y (m)', **self.font_config)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax2)
        if log_scale:
            cbar.set_label('log₁₀(浓度) (kg/m³)', **self.font_config)
        else:
            cbar.set_label('浓度 (kg/m³)', **self.font_config)

        # 时间和统计信息显示
        time_text = fig.suptitle(f'{title} - 时间: {time_steps[0]:.2f}h',
                                 **self.font_config, fontsize=14)

        # 添加统计信息文本
        stats_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                              verticalalignment='top', **self.font_config,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def animate(frame):
            """动画更新函数"""
            # 清除图形
            ax1.clear()

            # 获取当前帧数据
            conc = all_concentrations[frame]
            if log_scale and np.any(conc > 0):
                conc_plot = np.log10(conc + 1e-10)
            else:
                conc_plot = conc

            # 更新等值线图
            cs = ax1.contourf(self.X, self.Y, conc_plot, levels=20, cmap=colormap, norm=norm)
            ax1.set_title('浓度分布', **self.font_config)
            ax1.set_xlabel('X (m)', **self.font_config)
            ax1.set_ylabel('Y (m)', **self.font_config)

            # 更新图像
            im.set_array(conc_plot.T)

            # 更新统计信息
            max_conc = np.max(conc)
            mean_conc = np.mean(conc[conc > 0]) if np.any(conc > 0) else 0
            total_mass = np.sum(conc)

            stats_info = f'最大浓度: {max_conc:.2e} kg/m³\n'
            stats_info += f'平均浓度: {mean_conc:.2e} kg/m³\n'
            stats_info += f'总质量: {total_mass:.2e} kg'

            stats_text = ax1.text(0.02, 0.98, stats_info, transform=ax1.transAxes,
                                  verticalalignment='top', **self.font_config,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # 更新时间
            time_text.set_text(f'{title} - 时间: {time_steps[frame]:.2f}h')

            return []

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(concentration_time_series),
                                       interval=1000//fps, blit=False, repeat=True)

        plt.tight_layout()

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            else:
                anim.save(save_path + '.gif', writer='pillow', fps=fps)
            logging.info(f"浓度动画保存至: {save_path}")

        return anim

    def create_particle_trajectory_animation(self,
                                             particle_trajectories: List[np.ndarray],
                                             time_steps: List[float],
                                             background_field: Optional[List[np.ndarray]] = None,
                                             title: str = "粒子轨迹动画",
                                             trail_length: int = 10,
                                             save_path: Optional[str] = None,
                                             fps: int = 15) -> animation.FuncAnimation:
        """
        创建粒子轨迹演化动画
        
        Args:
            particle_trajectories: 粒子轨迹时间序列
            time_steps: 时间步
            background_field: 背景场数据
            title: 动画标题
            trail_length: 轨迹尾迹长度
            save_path: 保存路径
            fps: 帧率
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # 计算显示范围
        all_positions = np.concatenate(particle_trajectories, axis=0)
        x_range = [np.min(all_positions[:, 0]) * self.dx, np.max(all_positions[:, 0]) * self.dx]
        y_range = [np.min(all_positions[:, 1]) * self.dy, np.max(all_positions[:, 1]) * self.dy]

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('X (m)', **self.font_config)
        ax.set_ylabel('Y (m)', **self.font_config)
        ax.set_title(title, **self.font_config)

        # 初始化粒子显示
        particles = ax.scatter([], [], s=20, c='red', alpha=0.8, zorder=5)

        # 轨迹线存储
        trail_lines = []
        for i in range(len(particle_trajectories[0])):
            line, = ax.plot([], [], 'b-', alpha=0.6, linewidth=1)
            trail_lines.append(line)

        # 背景场显示
        if background_field:
            bg_im = ax.contourf(self.X, self.Y, background_field[0][:, :, -1],
                                levels=20, cmap='Blues', alpha=0.3)

        # 时间显示
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            verticalalignment='top', **self.font_config,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        def animate(frame):
            """动画更新函数"""
            nonlocal particles, trail_lines
            current_positions = particle_trajectories[frame]

            # 更新粒子位置
            x_particles = current_positions[:, 0] * self.dx
            y_particles = current_positions[:, 1] * self.dy

            particles.set_offsets(np.column_stack([x_particles, y_particles]))

            # 更新轨迹尾迹
            start_frame = max(0, frame - trail_length)
            for i, line in enumerate(trail_lines):
                if i < len(current_positions):
                    # 获取该粒子的历史轨迹
                    traj_x = []
                    traj_y = []
                    for t in range(start_frame, frame + 1):
                        if t < len(particle_trajectories) and i < len(particle_trajectories[t]):
                            pos = particle_trajectories[t][i]
                            traj_x.append(pos[0] * self.dx)
                            traj_y.append(pos[1] * self.dy)

                    line.set_data(traj_x, traj_y)
                    # 设置透明度渐变
                    alpha = 0.8 * (len(traj_x) / trail_length) if len(traj_x) > 0 else 0
                    line.set_alpha(min(alpha, 0.8))

            # 更新背景场
            if background_field and frame < len(background_field):
                ax.clear()
                ax.contourf(self.X, self.Y, background_field[frame][:, :, -1],
                            levels=20, cmap='Blues', alpha=0.3)
                ax.set_xlim(x_range)
                ax.set_ylim(y_range)
                ax.set_xlabel('X (m)', **self.font_config)
                ax.set_ylabel('Y (m)', **self.font_config)
                ax.set_title(title, **self.font_config)

                # 重新绘制粒子和轨迹
                particles = ax.scatter(x_particles, y_particles, s=20, c='red', alpha=0.8, zorder=5)
                for i, line in enumerate(trail_lines):
                    trail_lines[i], = ax.plot(line.get_xdata(), line.get_ydata(), 'b-',
                                              alpha=line.get_alpha(), linewidth=1)


            # 更新时间和统计信息
            active_particles = len(current_positions)
            time_info = f'时间: {time_steps[frame]:.2f}h\n活跃粒子: {active_particles}'
            time_text.set_text(time_info)

            return [particles] + trail_lines + [time_text]

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(particle_trajectories),
                                       interval=1000//fps, blit=False, repeat=True)

        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=fps)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=fps)
            else:
                anim.save(save_path + '.gif', writer='pillow', fps=fps)
            logging.info(f"粒子轨迹动画保存至: {save_path}")

        return anim
if __name__ == "__main__":
    # animation_generator.py 测试
    import numpy as np
    import matplotlib.pyplot as plt

    # 测试配置
    grid_config = {
        'nx': 40, 'ny': 40, 'nz': 10,
        'dx': 200.0, 'dy': 200.0, 'dz': 5.0
    }

    # 创建动画生成器
    anim_gen = AnimationGenerator(grid_config)

    # 生成时间序列数据
    print("生成时间序列测试数据...")
    n_frames = 20
    time_steps = np.linspace(0, 2.0, n_frames)  # 2小时

    # 速度场时间序列
    velocity_series = []
    for t in time_steps:
        u = np.sin(2*np.pi*t/2) * np.ones((40, 40, 10)) + 0.1*np.random.randn(40, 40, 10)
        v = np.cos(2*np.pi*t/2) * np.ones((40, 40, 10)) + 0.1*np.random.randn(40, 40, 10)
        w = 0.1 * np.sin(4*np.pi*t/2) * np.random.randn(40, 40, 10)
        velocity_series.append((u, v, w))

    # 创建速度场动画
    print("创建速度场动画...")
    anim1 = anim_gen.create_velocity_animation(velocity_series, time_steps,
                                               title="测试海流演化动画",
                                               save_path="test_velocity_animation.gif")
    plt.show()

    # 浓度场时间序列
    concentration_series = []
    for i, t in enumerate(time_steps):
        conc = np.zeros((40, 40, 10))
        # 扩散的污染源
        source_x, source_y = 20, 20
        spread = 5 + i * 0.5  # 逐渐扩散

        for x in range(40):
            for y in range(40):
                for z in range(10):
                    dist = np.sqrt((x-source_x)**2 + (y-source_y)**2)
                    conc[x, y, z] = 50 * np.exp(-dist**2/(2*spread**2)) * np.exp(-z/3)

        concentration_series.append(conc)

    # 创建浓度场动画
    print("创建浓度场动画...")
    anim2 = anim_gen.create_concentration_animation(concentration_series, time_steps,
                                                    title="测试污染物扩散动画",
                                                    save_path="test_concentration_animation.gif")
    plt.show()

    # 粒子轨迹时间序列
    n_particles = 30
    particle_series = []

    # 初始粒子位置
    initial_pos = np.random.uniform(0.3, 0.7, (n_particles, 3))
    current_pos = initial_pos.copy()

    for i, t in enumerate(time_steps):
        # 简单的运动模型
        dt = 0.1
        current_pos[:, 0] += dt * np.sin(2*np.pi*t/2) + 0.01*np.random.randn(n_particles)
        current_pos[:, 1] += dt * np.cos(2*np.pi*t/2) + 0.01*np.random.randn(n_particles)
        current_pos[:, 2] += 0.001*np.random.randn(n_particles)

        # 边界处理
        current_pos = np.clip(current_pos, 0, 1)
        particle_series.append(current_pos.copy())

    # 创建粒子轨迹动画
    print("创建粒子轨迹动画...")
    anim3 = anim_gen.create_particle_trajectory_animation(particle_series, time_steps,
                                                          background_field=concentration_series,
                                                          title="测试粒子轨迹动画",
                                                          save_path="test_particle_animation.gif")
    plt.show()

    print("动画生成器测试完成!")

