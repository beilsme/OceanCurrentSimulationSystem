#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: chinese_config.py
模块: src.utils.chinese_config
功能: 提供全平台中文显示支持的配置工具
作者: 系统配置模块
版本: v1.0.1
创建时间: 2025-06-17
"""

import os
import sys
import platform
import locale
import warnings
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 配置日志
logger = logging.getLogger(__name__)

# 预定义的中文字体列表（按优先级排序）
CHINESE_FONTS = {
    'Windows': [
        'Microsoft YaHei',      # 微软雅黑
        'SimHei',              # 黑体
        'SimSun',              # 宋体
        'FangSong',            # 仿宋
        'KaiTi',               # 楷体
        'Microsoft JhengHei',   # 微软正黑
        'MingLiU',             # 细明体
        'PMingLiU',            # 新细明体
    ],
    'Darwin': [  # macOS
        'PingFang SC',         # 苹方简体
        'Heiti SC',            # 黑体-简
        'Songti SC',           # 宋体-简
        'STHeiti',             # 华文黑体
        'STSong',              # 华文宋体
        'STKaiti',             # 华文楷体
        'STFangsong',          # 华文仿宋
        'Arial Unicode MS',     # Arial Unicode
    ],
    'Linux': [
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'WenQuanYi Zen Hei',   # 文泉驿正黑
        'Noto Sans CJK SC',    # 思源黑体简体
        'Noto Serif CJK SC',   # 思源宋体简体
        'Source Han Sans SC',   # 思源黑体
        'Source Han Serif SC',  # 思源宋体
        'AR PL UMing CN',      # 文鼎PL细上海宋
        'AR PL UKai CN',       # 文鼎PL中楷
        'DejaVu Sans',         # DejaVu（后备）
    ]
}

# 字体文件路径模式
FONT_PATHS = {
    'Windows': [
        'C:/Windows/Fonts/',
        os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts'),
    ],
    'Darwin': [
        '/System/Library/Fonts/',
        '/Library/Fonts/',
        os.path.expanduser('~/Library/Fonts/'),
    ],
    'Linux': [
        '/usr/share/fonts/',
        '/usr/local/share/fonts/',
        os.path.expanduser('~/.fonts/'),
        os.path.expanduser('~/.local/share/fonts/'),
    ]
}

class ChineseConfig:
    """中文配置管理器"""

    def __init__(self):
        self.system = platform.system()
        self.available_fonts = []
        self.selected_font = None
        self.original_settings = {}

    def setup_chinese_support(self,
                              preferred_font: Optional[str] = None,
                              font_size: int = 12,
                              dpi: int = 100,
                              use_tex: bool = False,
                              force_rebuild_cache: bool = False) -> Dict[str, any]:
        """
        配置全局中文支持
        
        参数:
            preferred_font: 首选字体名称
            font_size: 字体大小
            dpi: DPI设置
            use_tex: 是否使用TeX渲染（通常不推荐用于中文）
            force_rebuild_cache: 是否强制重建字体缓存
            
        返回:
            配置结果字典
        """
        results = {
            'success': False,
            'font_used': None,
            'available_fonts': [],
            'encoding': None,
            'warnings': []
        }

        try:
            # 1. 设置系统编码
            self._setup_encoding()
            results['encoding'] = sys.getdefaultencoding()

            # 2. 重建字体缓存（如果需要）
            if force_rebuild_cache:
                self._rebuild_font_cache()

            # 3. 查找可用的中文字体
            self.available_fonts = self._find_chinese_fonts()
            results['available_fonts'] = self.available_fonts

            if not self.available_fonts:
                results['warnings'].append("未找到任何中文字体，中文可能无法正常显示")
                logger.warning("未找到任何中文字体")

            # 4. 选择字体
            if preferred_font and preferred_font in self.available_fonts:
                self.selected_font = preferred_font
            else:
                # 按系统优先级选择字体
                for font in CHINESE_FONTS.get(self.system, []):
                    if font in self.available_fonts:
                        self.selected_font = font
                        break

                # 如果仍未找到，使用第一个可用字体
                if not self.selected_font and self.available_fonts:
                    self.selected_font = self.available_fonts[0]

            # 5. 应用matplotlib配置
            if self.selected_font:
                self._configure_matplotlib(self.selected_font, font_size, dpi, use_tex)
                results['font_used'] = self.selected_font
                results['success'] = True
                logger.info(f"成功配置中文字体: {self.selected_font}")
            else:
                results['warnings'].append("未能设置中文字体")

        except Exception as e:
            results['warnings'].append(f"配置过程中出现错误: {str(e)}")
            logger.error(f"配置中文支持时出错: {e}", exc_info=True)

        return results

    def _setup_encoding(self):
        """设置系统编码为UTF-8"""
        # Python 3 默认使用UTF-8，但确保环境变量正确设置
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 尝试设置locale
        try:
            if self.system == 'Windows':
                locale.setlocale(locale.LC_ALL, 'Chinese_China.936')
            else:
                # Linux和macOS
                for loc in ['zh_CN.UTF-8', 'zh_CN.utf8', 'en_US.UTF-8']:
                    try:
                        locale.setlocale(locale.LC_ALL, loc)
                        break
                    except locale.Error:
                        continue
        except Exception as e:
            logger.warning(f"设置locale失败: {e}")

    def _find_chinese_fonts(self) -> List[str]:
        """查找系统中可用的中文字体"""
        chinese_fonts = []

        # 获取matplotlib已知的所有字体
        available_fonts = set()
        for font in fm.fontManager.ttflist:
            available_fonts.add(font.name)

        # 检查预定义的中文字体
        for font_name in CHINESE_FONTS.get(self.system, []):
            if font_name in available_fonts:
                chinese_fonts.append(font_name)

        # 如果预定义字体不够，扫描字体文件
        if len(chinese_fonts) < 3:
            chinese_fonts.extend(self._scan_font_files())

        # 去重并保持顺序
        seen = set()
        unique_fonts = []
        for font in chinese_fonts:
            if font not in seen:
                seen.add(font)
                unique_fonts.append(font)

        return unique_fonts

    def _scan_font_files(self) -> List[str]:
        """扫描字体文件查找中文字体"""
        additional_fonts = []

        for font_dir in FONT_PATHS.get(self.system, []):
            if not os.path.exists(font_dir):
                continue

            try:
                for font_file in Path(font_dir).rglob('*.ttf'):
                    try:
                        # 简单检查文件名是否包含中文字体关键词
                        filename = font_file.stem.lower()
                        if any(keyword in filename for keyword in
                               ['chinese', 'cjk', 'sc', 'cn', 'hei', 'song', 'kai', 'fang']):
                            font_prop = fm.FontProperties(fname=str(font_file))
                            font_name = font_prop.get_name()
                            if font_name and font_name not in additional_fonts:
                                additional_fonts.append(font_name)
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"扫描字体目录 {font_dir} 时出错: {e}")

        return additional_fonts

    def _rebuild_font_cache(self):
        """重建matplotlib字体缓存"""
        logger.info("正在重建字体缓存...")
        fm._rebuild()
        logger.info("字体缓存重建完成")

    def _configure_matplotlib(self, font_name: str, font_size: int, dpi: int, use_tex: bool):
        """配置matplotlib使用中文字体"""
        # 保存原始设置
        self.original_settings = {
            'font.family': rcParams['font.family'],
            'font.sans-serif': rcParams['font.sans-serif'].copy(),
            'font.size': rcParams['font.size'],
            'figure.dpi': rcParams['figure.dpi'],
            'axes.unicode_minus': rcParams['axes.unicode_minus'],
            'text.usetex': rcParams['text.usetex']
        }

        # 设置字体
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.size'] = font_size
        plt.rcParams['figure.dpi'] = dpi

        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # TeX设置（通常不推荐用于中文）
        plt.rcParams['text.usetex'] = use_tex

        # 设置后端（确保支持中文）
        if self.system == 'Linux' and 'DISPLAY' not in os.environ:
            matplotlib.use('Agg')  # 无显示器环境使用Agg后端

    def restore_settings(self):
        """恢复原始matplotlib设置"""
        if self.original_settings:
            for key, value in self.original_settings.items():
                plt.rcParams[key] = value
            logger.info("已恢复原始matplotlib设置")

    def test_chinese_display(self, save_path: Optional[str] = None) -> bool:
        """
        测试中文显示效果
        
        参数:
            save_path: 保存测试图片的路径（已注释掉保存功能）
            
        返回:
            是否测试成功
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 测试文本
            test_text = """中文显示测试
简体中文：我能吃玻璃而不伤身体
繁體中文：我能吃玻璃而不傷身體
日本语：私はガラスを食べられます
한국어: 나는 유리를 먹을 수 있어요
English: The quick brown fox jumps over the lazy dog
数学符号：∑∫∂ α β γ ± × ÷"""

            ax.text(0.5, 0.5, test_text,
                    transform=ax.transAxes,
                    fontsize=14,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_title(f'中文字体测试 - {self.selected_font or "默认字体"}', fontsize=16)
            ax.axis('off')

            # 注释掉保存和显示图片的逻辑
            # if save_path:
            #     plt.savefig(save_path, dpi=150, bbox_inches='tight')
            #     logger.info(f"测试图片已保存到: {save_path}")
            # else:
            #     plt.show()

            # 仅记录测试完成信息
            logger.info("中文显示测试已完成（未保存/显示图片）")

            plt.close()
            return True

        except Exception as e:
            logger.error(f"中文显示测试失败: {e}")
            return False

    def get_font_info(self) -> Dict[str, any]:
        """获取当前字体配置信息"""
        return {
            'system': self.system,
            'current_font': self.selected_font,
            'available_fonts': self.available_fonts,
            'matplotlib_backend': matplotlib.get_backend(),
            'encoding': sys.getdefaultencoding(),
            'locale': locale.getlocale(),
            'font_paths': FONT_PATHS.get(self.system, [])
        }


# 便捷函数
def setup_chinese_all(preferred_font: Optional[str] = None,
                      font_size: int = 12,
                      dpi: int = 100,
                      test: bool = True,
                      test_save_path: Optional[str] = None) -> bool:
    """
    一键配置全局中文支持
    
    参数:
        preferred_font: 首选字体名称
        font_size: 字体大小
        dpi: DPI设置
        test: 是否进行测试
        test_save_path: 测试图片保存路径（已不再使用）
        
    返回:
        是否配置成功
    """
    config = ChineseConfig()
    result = config.setup_chinese_support(
        preferred_font=preferred_font,
        font_size=font_size,
        dpi=dpi
    )

    if result['success']:
        logger.info(f"中文配置成功，使用字体: {result['font_used']}")

        if test:
            # 注释掉传递 test_save_path 参数
            success = config.test_chinese_display()  # 不再传递 test_save_path
            if not success:
                logger.warning("中文显示测试未通过")
                return False
    else:
        logger.error("中文配置失败")
        for warning in result['warnings']:
            logger.warning(warning)
        return False

    return True


def list_chinese_fonts() -> List[str]:
    """列出系统中所有可用的中文字体"""
    config = ChineseConfig()
    fonts = config._find_chinese_fonts()
    return fonts


def fix_chinese_display_issues():
    """修复常见的中文显示问题"""
    # 1. 清除字体缓存
    cache_dir = Path.home() / '.matplotlib'
    if cache_dir.exists():
        for cache_file in cache_dir.glob('fontlist-*.json'):
            try:
                cache_file.unlink()
                logger.info(f"已删除字体缓存: {cache_file}")
            except Exception as e:
                logger.warning(f"删除字体缓存失败: {e}")

    # 2. 重建字体缓存
    fm._rebuild()

    # 3. 重新配置中文
    return setup_chinese_all()


# 模块初始化时的自动配置（可选）
def auto_configure():
    """模块导入时自动配置中文支持"""
    try:
        setup_chinese_all(test=False)
    except Exception as e:
        logger.warning(f"自动配置中文支持失败: {e}")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("中文配置工具测试")
    print("=" * 60)

    # 1. 列出可用字体
    print("\n可用的中文字体:")
    fonts = list_chinese_fonts()
    for i, font in enumerate(fonts, 1):
        print(f"{i}. {font}")

    # 2. 配置中文支持
    print("\n正在配置中文支持...")
    # 注释掉测试图片保存路径
    success = setup_chinese_all(
        font_size=14,
        dpi=120,
        test=True
        # test_save_path="chinese_test.png"  # 已注释掉
    )

    if success:
        print("\n✓ 中文配置成功！")
    else:
        print("\n✗ 中文配置失败！")

    # 3. 显示配置信息
    config = ChineseConfig()
    config.setup_chinese_support()
    info = config.get_font_info()

    print("\n当前配置信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")