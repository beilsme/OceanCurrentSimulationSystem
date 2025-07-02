#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: setup_python_csharp_interface.py
功能: Python对C#接口一键安装配置脚本
作者: beilsm
版本: v1.0.0
创建时间: 2025-07-02

主要功能:
- 自动安装Python依赖包
- 配置Python-C#接口环境
- 检测C#运行时环境
- 创建配置文件
- 启动Python服务端点
"""

import os
import sys
import subprocess
import platform
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import venv
import urllib.request
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonCSharpSetup:
    """Python-C#接口安装配置器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.python_engine_path = self.project_root / "Source" / "PythonEngine"
        self.config_path = self.project_root / "Configuration"
        self.venv_path = self.project_root / "venv_oceansim"

        # 必需的Python包
        self.required_packages = [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
            "xarray>=0.19.0",
            "netcdf4>=1.5.0",
            "h5py>=3.1.0",
            "pyyaml>=5.4.0",
            "requests>=2.25.0",
            "flask>=2.0.0",
            "uvicorn>=0.15.0",
            "fastapi>=0.70.0",
            "websockets>=10.0",
            "aiohttp>=3.8.0",
            "python-dotnet>=3.0.0",  # 关键：Python.NET用于C#互操作
            "pywin32>=227;sys_platform=='win32'",  # Windows COM支持
        ]

        # C#运行时检测
        self.dotnet_versions = []
        self.system_info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version()
        }

    def detect_dotnet_runtime(self) -> bool:
        """检测.NET运行时环境"""
        logger.info("检测.NET运行时环境...")

        try:
            # 检测 .NET Core/5+
            result = subprocess.run(['dotnet', '--list-runtimes'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Microsoft.NETCore.App' in line or 'Microsoft.AspNetCore.App' in line:
                        version = line.split()[1]
                        self.dotnet_versions.append(version)
                logger.info(f"找到.NET运行时版本: {self.dotnet_versions}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("未找到.NET运行时")

        # Windows下检测.NET Framework
        if platform.system() == "Windows":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full") as key:
                    release, _ = winreg.QueryValueEx(key, "Release")
                    logger.info(f"找到.NET Framework，Release: {release}")
                    return True
            except Exception as e:
                logger.warning(f"检测.NET Framework失败: {e}")

        return False

    def create_virtual_environment(self) -> bool:
        """创建Python虚拟环境"""
        logger.info(f"创建Python虚拟环境: {self.venv_path}")

        if self.venv_path.exists():
            logger.info("虚拟环境已存在，删除重建...")
            shutil.rmtree(self.venv_path)

        try:
            venv.create(self.venv_path, with_pip=True)
            logger.info("虚拟环境创建成功")
            return True
        except Exception as e:
            logger.error(f"创建虚拟环境失败: {e}")
            return False

    def get_python_executable(self) -> Path:
        """获取虚拟环境中的Python可执行文件路径"""
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"

    def install_python_packages(self) -> bool:
        """安装Python依赖包"""
        logger.info("安装Python依赖包...")

        python_exe = self.get_python_executable()
        if not python_exe.exists():
            logger.error("Python可执行文件不存在")
            return False

        # 升级pip
        try:
            subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                           check=True, timeout=300)
            logger.info("pip升级成功")
        except Exception as e:
            logger.warning(f"pip升级失败: {e}")

        # 安装依赖包
        for package in self.required_packages:
            try:
                logger.info(f"安装: {package}")
                subprocess.run([str(python_exe), "-m", "pip", "install", package],
                               check=True, timeout=600)
            except subprocess.CalledProcessError as e:
                logger.error(f"安装{package}失败: {e}")
                # 对于关键包，直接返回失败
                if "python-dotnet" in package or "numpy" in package:
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"安装{package}超时")
                return False

        logger.info("Python依赖包安装完成")
        return True

    def create_python_csharp_bridge(self):
        """创建Python-C#桥接模块"""
        bridge_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python-C#桥接模块
提供Python调用C#服务的接口
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# 添加Python.NET支持
try:
    import clr
    sys.path.append(r"./Build/Release/CSharp")
    
    # 引用C#程序集
    clr.AddReference("OceanSimulation.Domain")
    clr.AddReference("OceanSimulation.Infrastructure")
    clr.AddReference("OceanSimulation.Application")
    
    # 导入C#命名空间
    from OceanSimulation.Application.Services import SimulationOrchestrationService
    from OceanSimulation.Domain.Entities import OceanDataEntity
    from OceanSimulation.Infrastructure.ComputeEngines import CppEngineWrapper
    
    CSHARP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"无法导入C#模块: {e}")
    CSHARP_AVAILABLE = False

class PythonCSharpBridge:
    """Python-C#桥接器"""
    
    def __init__(self, config_path: str = "./Configuration/python_service.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.csharp_service = None
        
        if CSHARP_AVAILABLE:
            self._initialize_csharp_service()
    
    def _initialize_csharp_service(self):
        """初始化C#服务"""
        try:
            self.csharp_service = SimulationOrchestrationService()
            self.logger.info("C#服务初始化成功")
        except Exception as e:
            self.logger.error(f"C#服务初始化失败: {e}")
            self.csharp_service = None
    
    def is_available(self) -> bool:
        """检查C#服务是否可用"""
        return CSHARP_AVAILABLE and self.csharp_service is not None
    
    def call_csharp_method(self, method_name: str, **kwargs) -> Optional[Any]:
        """调用C#方法"""
        if not self.is_available():
            self.logger.error("C#服务不可用")
            return None
        
        try:
            method = getattr(self.csharp_service, method_name)
            return method(**kwargs)
        except Exception as e:
            self.logger.error(f"调用C#方法{method_name}失败: {e}")
            return None
    
    def send_data_to_csharp(self, data: Dict[str, Any]) -> bool:
        """向C#发送数据"""
        try:
            if self.is_available():
                # 转换Python数据为C#格式
                json_data = json.dumps(data)
                result = self.csharp_service.ReceiveDataFromPython(json_data)
                return result
            return False
        except Exception as e:
            self.logger.error(f"向C#发送数据失败: {e}")
            return False
    
    def get_data_from_csharp(self, data_type: str) -> Optional[Dict[str, Any]]:
        """从C#获取数据"""
        try:
            if self.is_available():
                json_result = self.csharp_service.SendDataToPython(data_type)
                return json.loads(json_result) if json_result else None
            return None
        except Exception as e:
            self.logger.error(f"从C#获取数据失败: {e}")
            return None

# 全局桥接实例
bridge = PythonCSharpBridge()
'''

        bridge_file = self.python_engine_path / "core" / "csharp_bridge.py"
        bridge_file.parent.mkdir(parents=True, exist_ok=True)
        bridge_file.write_text(bridge_content, encoding='utf-8')
        logger.info(f"Python-C#桥接模块已创建: {bridge_file}")

    def create_configuration_files(self):
        """创建配置文件"""
        logger.info("创建配置文件...")

        # Python服务配置
        python_config = {
            "service": {
                "name": "OceanSimulation.PythonEngine",
                "version": "1.0.0",
                "host": "localhost",
                "port": 8888,
                "debug": True
            },
            "csharp_integration": {
                "assembly_path": "./Build/Release/CSharp",
                "assemblies": [
                    "OceanSimulation.Domain.dll",
                    "OceanSimulation.Infrastructure.dll",
                    "OceanSimulation.Application.dll"
                ],
                "communication": {
                    "method": "direct_clr",
                    "timeout": 30000,
                    "retry_count": 3
                }
            },
            "cpp_integration": {
                "library_path": "./Build/Release/Cpp",
                "library_name": "oceansim_csharp",
                "max_threads": 4
            },
            "data_processing": {
                "cache_enabled": True,
                "cache_path": "./Data/Cache/PythonCache",
                "max_memory_mb": 2048,
                "enable_gpu": False
            },
            "visualization": {
                "backend": "plotly",
                "output_format": ["html", "png", "json"],
                "dpi": 300,
                "style": "modern"
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/python_engine.log",
                "max_size": "10MB",
                "backup_count": 5
            }
        }

        config_file = self.config_path / "python_service.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(python_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Python服务配置文件已创建: {config_file}")

    def create_startup_script(self):
        """创建启动脚本"""
        startup_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python引擎启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "Source" / "PythonEngine"))

# 激活虚拟环境
venv_path = project_root / "venv_oceansim"
if {self.system_info["platform"] == "Windows"}:
    activate_script = venv_path / "Scripts" / "activate_this.py"
else:
    activate_script = venv_path / "bin" / "activate_this.py"

if activate_script.exists():
    exec(open(activate_script).read(), {{"__file__": str(activate_script)}})

# 启动Python引擎
if __name__ == "__main__":
    from core.csharp_bridge import bridge
    from services.python_engine_service import PythonEngineService
    
    print("启动Ocean Simulation Python引擎...")
    
    # 检查C#桥接
    if bridge.is_available():
        print("✓ C#桥接可用")
    else:
        print("✗ C#桥接不可用")
    
    # 启动服务
    service = PythonEngineService()
    service.start()
'''

        startup_file = self.python_engine_path / "start_python_engine.py"
        startup_file.write_text(startup_content, encoding='utf-8')
        startup_file.chmod(0o755)  # 可执行权限
        logger.info(f"启动脚本已创建: {startup_file}")

    def verify_installation(self) -> bool:
        """验证安装"""
        logger.info("验证安装...")

        python_exe = self.get_python_executable()

        # 测试Python.NET
        test_script = '''
import sys
try:
    import clr
    print("✓ Python.NET可用")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Python.NET不可用: {e}")
    sys.exit(1)
'''

        try:
            result = subprocess.run([str(python_exe), "-c", test_script],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("Python.NET验证成功")
                return True
            else:
                logger.error(f"Python.NET验证失败: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"验证过程失败: {e}")
            return False

    def run_setup(self) -> bool:
        """执行完整安装流程"""
        logger.info("开始Python-C#接口安装...")

        # 检查.NET运行时
        if not self.detect_dotnet_runtime():
            logger.error("未检测到.NET运行时，请先安装.NET 6.0或更高版本")
            return False

        # 创建虚拟环境
        if not self.create_virtual_environment():
            return False

        # 安装Python包
        if not self.install_python_packages():
            return False

        # 创建桥接模块
        self.create_python_csharp_bridge()

        # 创建配置文件
        self.create_configuration_files()

        # 创建启动脚本
        self.create_startup_script()

        # 验证安装
        if not self.verify_installation():
            logger.warning("安装验证失败，但基础组件已安装")

        logger.info("Python-C#接口安装完成！")
        logger.info(f"虚拟环境位置: {self.venv_path}")
        logger.info(f"启动脚本: {self.python_engine_path / 'start_python_engine.py'}")

        return True

def main():
    """主函数"""
    print("=== Ocean Simulation Python-C#接口安装器 ===")
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print("=" * 50)

    setup = PythonCSharpSetup()

    try:
        success = setup.run_setup()
        if success:
            print("\n🎉 安装成功！")
            print("\n下一步:")
            print("1. 构建C#项目")
            print("2. 运行启动脚本测试连接")
            print(f"3. 执行: python {setup.python_engine_path / 'start_python_engine.py'}")
        else:
            print("\n❌ 安装失败，请检查日志")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户取消安装")
        sys.exit(1)
    except Exception as e:
        logger.error(f"安装过程发生错误: {e}")
        print(f"\n❌ 安装失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()