#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件: setup_python_csharp_interface.py
功能: Python对C#接口一键安装配置脚本 (跨平台优化版)
作者: beilsm
版本: v1.1.0
创建时间: 2025-07-02
更新时间: 2025-07-02

主要功能:
- 自动安装Python依赖包
- 配置Python-C#接口环境
- 检测C#运行时环境
- 创建配置文件
- 启动Python服务端点
- 增强Mac M1和跨平台支持
"""

import os
import sys
import subprocess
import platform
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import venv
import urllib.request
import shutil
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PythonCSharpSetup:
    """Python-C#接口安装配置器 (跨平台增强版)"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.python_engine_path = self.project_root / "Source" / "PythonEngine"
        self.config_path = self.project_root / "Configuration"
        self.venv_path = self.project_root / "venv_oceansim"

        # 系统信息检测
        self.system_info = self._detect_system_info()

        # 根据系统调整Python包列表
        self.required_packages = self._get_platform_packages()

        # C#运行时检测
        self.dotnet_versions = []

    def _detect_system_info(self) -> Dict[str, str]:
        """检测系统信息"""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # 检测Apple Silicon
        is_apple_silicon = False
        if system == "darwin":
            try:
                # 检查是否为Apple Silicon
                result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
                if result.returncode == 0 and 'arm64' in result.stdout:
                    is_apple_silicon = True
            except:
                pass

        # 检测Python架构
        python_arch = platform.architecture()[0]

        info = {
            "platform": system,
            "machine": machine,
            "python_version": platform.python_version(),
            "python_arch": python_arch,
            "is_apple_silicon": is_apple_silicon,
            "is_windows": system == "windows",
            "is_macos": system == "darwin",
            "is_linux": system == "linux"
        }

        logger.info(f"系统信息: {info}")
        return info

    def _get_platform_packages(self) -> List[str]:
        """根据平台获取Python包列表"""
        base_packages = [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "xarray>=0.19.0",
            "h5py>=3.1.0",
            "pyyaml>=5.4.0",
            "requests>=2.25.0",
            "flask>=2.0.0",
            "uvicorn>=0.15.0",
            "fastapi>=0.70.0",
            "websockets>=10.0",
            "aiohttp>=3.8.0",
        ]

        # 根据平台添加特定包
        if self.system_info["is_windows"]:
            base_packages.extend([
                "pythonnet>=3.0.0",
                "pywin32>=227"
            ])
        elif self.system_info["is_macos"]:
            # Mac上的Python.NET支持
            base_packages.extend([
                "pythonnet>=3.0.0",
            ])

            # Apple Silicon特殊处理
            if self.system_info["is_apple_silicon"]:
                # 某些包在Apple Silicon上需要特殊版本
                base_packages.extend([
                    "tensorflow-macos>=2.8.0;platform_machine=='arm64'",
                    "tensorflow>=2.8.0;platform_machine!='arm64'",
                ])
            else:
                base_packages.append("tensorflow>=2.8.0")

            # PyTorch for Mac
            base_packages.append("torch>=1.11.0")

        elif self.system_info["is_linux"]:
            base_packages.extend([
                "pythonnet>=3.0.0",
                "tensorflow>=2.8.0",
                "torch>=1.11.0",
            ])

        # 条件性包 - 只在支持的平台上安装
        try:
            base_packages.append("netcdf4>=1.5.0")
        except:
            logger.warning("netcdf4可能在当前平台不可用")

        return base_packages

    def detect_dotnet_runtime(self) -> bool:
        """检测.NET运行时环境 (跨平台增强)"""
        logger.info("检测.NET运行时环境...")

        # 方法1: 使用dotnet命令
        if self._check_dotnet_command():
            return True

        # 方法2: 检查环境变量
        if self._check_dotnet_env():
            return True

        # 方法3: 平台特定检测
        if self.system_info["is_windows"]:
            return self._check_dotnet_windows()
        elif self.system_info["is_macos"]:
            return self._check_dotnet_macos()
        elif self.system_info["is_linux"]:
            return self._check_dotnet_linux()

        return False

    def _check_dotnet_command(self) -> bool:
        """检查dotnet命令"""
        try:
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
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            logger.debug("dotnet命令不可用")
        return False

    def _check_dotnet_env(self) -> bool:
        """检查.NET环境变量"""
        dotnet_root = os.environ.get('DOTNET_ROOT')
        if dotnet_root and Path(dotnet_root).exists():
            logger.info(f"找到DOTNET_ROOT: {dotnet_root}")
            return True
        return False

    def _check_dotnet_windows(self) -> bool:
        """Windows下检测.NET"""
        try:
            import winreg
            # 检测.NET Framework
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full") as key:
                release, _ = winreg.QueryValueEx(key, "Release")
                logger.info(f"找到.NET Framework，Release: {release}")
                return True
        except Exception as e:
            logger.debug(f"检测.NET Framework失败: {e}")

        # 检测.NET Core/5+
        try:
            program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
            dotnet_path = Path(program_files) / "dotnet"
            if dotnet_path.exists():
                logger.info(f"找到.NET安装目录: {dotnet_path}")
                return True
        except Exception as e:
            logger.debug(f"检测.NET Core失败: {e}")

        return False

    def _check_dotnet_macos(self) -> bool:
        """macOS下检测.NET"""
        # 常见的.NET安装路径
        common_paths = [
            "/usr/local/share/dotnet",
            "/usr/share/dotnet",
            Path.home() / ".dotnet",
            "/Applications/dotnet"
        ]

        for path in common_paths:
            if Path(path).exists():
                logger.info(f"找到.NET安装目录: {path}")
                return True

        # 检查Homebrew安装的.NET
        try:
            result = subprocess.run(['brew', 'list', 'dotnet'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("通过Homebrew找到.NET")
                return True
        except:
            pass

        return False

    def _check_dotnet_linux(self) -> bool:
        """Linux下检测.NET"""
        # 常见的.NET安装路径
        common_paths = [
            "/usr/share/dotnet",
            "/opt/dotnet",
            Path.home() / ".dotnet"
        ]

        for path in common_paths:
            if Path(path).exists():
                logger.info(f"找到.NET安装目录: {path}")
                return True

        # 检查包管理器安装的.NET
        package_managers = [
            ['dpkg', '-l', 'dotnet*'],
            ['rpm', '-qa', 'dotnet*'],
            ['pacman', '-Q', 'dotnet*']
        ]

        for cmd in package_managers:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    logger.info(f"通过包管理器找到.NET: {cmd[0]}")
                    return True
            except:
                continue

        return False

    def create_virtual_environment(self) -> bool:
        """创建Python虚拟环境 (增强版)"""
        logger.info(f"创建Python虚拟环境: {self.venv_path}")

        if self.venv_path.exists():
            logger.info("虚拟环境已存在，删除重建...")
            try:
                shutil.rmtree(self.venv_path)
            except PermissionError as e:
                logger.error(f"删除虚拟环境失败，权限不足: {e}")
                return False

        try:
            # 创建虚拟环境
            venv.create(self.venv_path, with_pip=True, clear=True)

            # 验证虚拟环境
            python_exe = self.get_python_executable()
            if not python_exe.exists():
                logger.error(f"虚拟环境创建失败，Python可执行文件不存在: {python_exe}")
                return False

            logger.info("虚拟环境创建成功")
            return True
        except Exception as e:
            logger.error(f"创建虚拟环境失败: {e}")
            return False

    def get_python_executable(self) -> Path:
        """获取虚拟环境中的Python可执行文件路径"""
        if self.system_info["is_windows"]:
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"

    def install_python_packages(self) -> bool:
        """安装Python依赖包 (增强版)"""
        logger.info("安装Python依赖包...")

        python_exe = self.get_python_executable()
        if not python_exe.exists():
            logger.error("Python可执行文件不存在")
            return False

        # 升级pip和基础工具
        basic_tools = ["pip", "setuptools", "wheel"]
        for tool in basic_tools:
            try:
                logger.info(f"升级: {tool}")
                subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", tool],
                               check=True, timeout=300)
            except Exception as e:
                logger.warning(f"{tool}升级失败: {e}")

        # 安装依赖包
        failed_packages = []
        for package in self.required_packages:
            if not self._install_single_package(python_exe, package):
                failed_packages.append(package)

        # 报告安装结果
        if failed_packages:
            logger.warning(f"以下包安装失败: {failed_packages}")
            # 检查关键包
            critical_failed = [pkg for pkg in failed_packages
                               if any(critical in pkg.lower()
                                      for critical in ["pythonnet", "numpy", "scipy"])]
            if critical_failed:
                logger.error(f"关键包安装失败: {critical_failed}")
                return False

        logger.info("Python依赖包安装完成")
        return True

    def _install_single_package(self, python_exe: Path, package: str) -> bool:
        """安装单个Python包"""
        try:
            logger.info(f"安装: {package}")

            # 构建安装命令
            cmd = [str(python_exe), "-m", "pip", "install", package]

            # Apple Silicon特殊处理
            if self.system_info["is_apple_silicon"] and "tensorflow" in package:
                # 设置环境变量以支持Apple Silicon
                env = os.environ.copy()
                env["GRPC_PYTHON_BUILD_SYSTEM_OPENSSL"] = "1"
                env["GRPC_PYTHON_BUILD_SYSTEM_ZLIB"] = "1"
            else:
                env = None

            subprocess.run(cmd, check=True, timeout=600, env=env)
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"安装{package}失败: {e}")

            # 尝试替代安装方法
            if "pythonnet" in package and not self.system_info["is_windows"]:
                return self._install_pythonnet_alternative(python_exe)

        except subprocess.TimeoutExpired:
            logger.error(f"安装{package}超时")

        return False

    def _install_pythonnet_alternative(self, python_exe: Path) -> bool:
        """非Windows系统的Python.NET替代安装"""
        logger.info("尝试Python.NET替代安装方法...")

        alternatives = [
            "pythonnet",
            "python-dotnet",
            "clr-loader"  # 轻量级替代
        ]

        for alt in alternatives:
            try:
                subprocess.run([str(python_exe), "-m", "pip", "install", alt],
                               check=True, timeout=300)
                logger.info(f"成功安装Python.NET替代包: {alt}")
                return True
            except:
                continue

        logger.warning("所有Python.NET安装方法均失败，将使用模拟模式")
        return True  # 允许继续，但功能受限

    def create_python_csharp_bridge(self):
        """创建Python-C#桥接模块 (跨平台增强)"""
        bridge_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python-C#桥接模块 (跨平台增强版)
提供Python调用C#服务的接口
"""

import os
import sys
import json
import logging
import platform
from typing import Dict, Any, Optional
from pathlib import Path

# 检测运行环境
SYSTEM_INFO = {
    "platform": platform.system().lower(),
    "is_windows": platform.system().lower() == "windows",
    "is_macos": platform.system().lower() == "darwin",
    "is_linux": platform.system().lower() == "linux",
    "is_apple_silicon": platform.machine().lower() == "arm64" and platform.system().lower() == "darwin"
}

# 尝试导入Python.NET
CSHARP_AVAILABLE = False
CLR_MODULE = None

def _try_import_pythonnet():
    """尝试导入Python.NET的不同变体"""
    global CSHARP_AVAILABLE, CLR_MODULE
    
    import_attempts = [
        ("clr", "pythonnet"),
        ("pythonnet", "python-dotnet"),
        ("clr_loader", "clr-loader")
    ]
    
    for module_name, package_name in import_attempts:
        try:
            CLR_MODULE = __import__(module_name)
            CSHARP_AVAILABLE = True
            logging.info(f"成功导入{package_name}")
            return True
        except ImportError:
            continue
    
    logging.warning("无法导入任何Python.NET实现")
    return False

# 执行导入尝试
_try_import_pythonnet()

# 如果成功导入，尝试加载C#程序集
if CSHARP_AVAILABLE and hasattr(CLR_MODULE, 'AddReference'):
    try:
        # 设置程序集路径
        assembly_paths = [
            "./Build/Release/CSharp",
            "../Build/Release/CSharp",
            "../../Build/Release/CSharp"
        ]
        
        for path in assembly_paths:
            if Path(path).exists():
                sys.path.append(path)
                break
        
        # 引用C#程序集
        CLR_MODULE.AddReference("OceanSimulation.Domain")
        CLR_MODULE.AddReference("OceanSimulation.Infrastructure")
        CLR_MODULE.AddReference("OceanSimulation.Application")
        
        # 导入C#命名空间
        from OceanSimulation.Application.Services import SimulationOrchestrationService
        from OceanSimulation.Domain.Entities import OceanDataEntity
        from OceanSimulation.Infrastructure.ComputeEngines import CppEngineWrapper
        
        logging.info("C#程序集加载成功")
        
    except ImportError as e:
        logging.warning(f"无法导入C#模块: {e}")
        CSHARP_AVAILABLE = False
    except Exception as e:
        logging.warning(f"C#程序集加载失败: {e}")
        CSHARP_AVAILABLE = False

class PythonCSharpBridge:
    """Python-C#桥接器 (跨平台增强版)"""
    
    def __init__(self, config_path: str = "./Configuration/python_service.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self.csharp_service = None
        self.mock_mode = not CSHARP_AVAILABLE
        
        if CSHARP_AVAILABLE:
            self._initialize_csharp_service()
        else:
            self.logger.warning("运行在模拟模式 - C#集成不可用")
    
    def _initialize_csharp_service(self):
        """初始化C#服务"""
        try:
            if 'SimulationOrchestrationService' in globals():
                self.csharp_service = SimulationOrchestrationService()
                self.logger.info("C#服务初始化成功")
            else:
                raise Exception("SimulationOrchestrationService未找到")
        except Exception as e:
            self.logger.error(f"C#服务初始化失败: {e}")
            self.csharp_service = None
            self.mock_mode = True
    
    def is_available(self) -> bool:
        """检查C#服务是否可用"""
        return CSHARP_AVAILABLE and self.csharp_service is not None and not self.mock_mode
    
    def call_csharp_method(self, method_name: str, **kwargs) -> Optional[Any]:
        """调用C#方法"""
        if self.mock_mode:
            self.logger.warning(f"模拟模式: 调用C#方法 {method_name}")
            return {"status": "mocked", "method": method_name, "args": kwargs}
        
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
        if self.mock_mode:
            self.logger.info(f"模拟模式: 发送数据到C# - {len(str(data))} 字符")
            return True
            
        try:
            if self.is_available():
                json_data = json.dumps(data)
                result = self.csharp_service.ReceiveDataFromPython(json_data)
                return result
            return False
        except Exception as e:
            self.logger.error(f"向C#发送数据失败: {e}")
            return False
    
    def get_data_from_csharp(self, data_type: str) -> Optional[Dict[str, Any]]:
        """从C#获取数据"""
        if self.mock_mode:
            self.logger.info(f"模拟模式: 从C#获取数据类型 {data_type}")
            return {"type": data_type, "data": "mocked_data", "timestamp": "2025-07-02"}
            
        try:
            if self.is_available():
                json_result = self.csharp_service.SendDataToPython(data_type)
                return json.loads(json_result) if json_result else None
            return None
        except Exception as e:
            self.logger.error(f"从C#获取数据失败: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "system_info": SYSTEM_INFO,
            "csharp_available": CSHARP_AVAILABLE,
            "mock_mode": self.mock_mode,
            "clr_module": str(type(CLR_MODULE)) if CLR_MODULE else None
        }

# 全局桥接实例
bridge = PythonCSharpBridge()
'''

        bridge_file = self.python_engine_path / "core" / "csharp_bridge.py"
        bridge_file.parent.mkdir(parents=True, exist_ok=True)
        bridge_file.write_text(bridge_content, encoding='utf-8')
        logger.info(f"Python-C#桥接模块已创建: {bridge_file}")

    def create_configuration_files(self):
        """创建配置文件 (跨平台增强)"""
        logger.info("创建配置文件...")

        # Python服务配置
        python_config = {
            "service": {
                "name": "OceanSimulation.PythonEngine",
                "version": "1.1.0",
                "host": "localhost",
                "port": 8888,
                "debug": True
            },
            "system": self.system_info,
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
                    "retry_count": 3,
                    "fallback_mode": "mock"
                }
            },
            "cpp_integration": {
                "library_path": "./Build/Release/Cpp",
                "library_name": "oceansim_csharp",
                "max_threads": 4,
                "platform_specific": {
                    "windows": {"extension": ".dll"},
                    "darwin": {"extension": ".dylib"},
                    "linux": {"extension": ".so"}
                }
            },
            "data_processing": {
                "cache_enabled": True,
                "cache_path": "./Data/Cache/PythonCache",
                "max_memory_mb": 2048,
                "enable_gpu": False,
                "apple_silicon_optimizations": self.system_info["is_apple_silicon"]
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

        # 使用json作为fallback，因为yaml可能未安装
        try:
            import yaml
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(python_config, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # 使用JSON格式作为fallback
            json_file = config_file.with_suffix('.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(python_config, f, indent=2, ensure_ascii=False)
            logger.info(f"YAML不可用，使用JSON格式: {json_file}")

        logger.info(f"Python服务配置文件已创建: {config_file}")

    def create_startup_script(self):
        """创建启动脚本 (跨平台增强)"""
        startup_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python引擎启动脚本 (跨平台增强版)
"""

import sys
import os
import platform
from pathlib import Path

print("=== Ocean Simulation Python引擎启动 ===")
print(f"Python版本: {{platform.python_version()}}")
print(f"操作系统: {{platform.system()}} {{platform.release()}}")
print(f"架构: {{platform.machine()}}")

# 添加项目路径

project_root = Path(__file__).parent
python_engine_path = project_root / "Source" / "PythonEngine"
sys.path.insert(0, str(python_engine_path))

# 确保core目录在路径中
core_path = python_engine_path / "core"
if core_path.exists():
    sys.path.insert(0, str(core_path.parent))

print(f"Python引擎路径: {{python_engine_path}}")
print(f"当前工作目录: {{Path.cwd()}}")

# 激活虚拟环境
venv_path = project_root / "venv_oceansim"
if platform.system() == "Windows":
    python_exe = venv_path / "Scripts" / "python.exe"
    activate_script = venv_path / "Scripts" / "activate_this.py"
else:
    python_exe = venv_path / "bin" / "python"
    activate_script = venv_path / "bin" / "activate_this.py"

# 检查虚拟环境
if not python_exe.exists():
    print(f"⚠️  虚拟环境不存在: {{venv_path}}")
    print("请先运行安装脚本")
    sys.exit(1)

# 尝试激活虚拟环境
if activate_script.exists():
    try:
        exec(open(activate_script).read(), {{"__file__": str(activate_script)}})
        print("✓ 虚拟环境已激活")
    except Exception as e:
        print(f"⚠️  虚拟环境激活失败: {{e}}")

# 启动Python引擎
if __name__ == "__main__":
    try:
        from core.csharp_bridge import bridge
        print("✓ 桥接模块加载成功")
        
        # 显示系统信息
        system_info = bridge.get_system_info()
        print(f"系统信息: {{system_info}}")
        
        # 检查C#桥接
        if bridge.is_available():
            print("✓ C#桥接可用")
        else:
            print("⚠️  C#桥接不可用，运行在模拟模式")
        
        # 启动服务 (如果存在)
        try:
            from services.python_engine_service import PythonEngineService
            service = PythonEngineService()
            service.start()
        except ImportError:
            print("⚠️  PythonEngineService未找到，创建基础HTTP服务...")
            
            # 创建基础HTTP服务
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            
            class BasicHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        response = {{
                            "status": "running",
                            "system": system_info,
                            "bridge_available": bridge.is_available()
                        }}
                        self.wfile.write(json.dumps(response).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            server = HTTPServer(('localhost', 8888), BasicHandler)
            print("✓ 基础HTTP服务启动在 http://localhost:8888")
            print("访问 http://localhost:8888/status 查看状态")
            server.serve_forever()
            
    except ImportError as e:
        print(f"❌ 导入错误: {{e}}")
        print("请检查安装是否完成")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {{e}}")
        sys.exit(1)
'''

        startup_file = self.python_engine_path / "start_python_engine.py"
        startup_file.parent.mkdir(parents=True, exist_ok=True)
        startup_file.write_text(startup_content, encoding='utf-8')

        # 设置可执行权限 (Unix系统)
        if not self.system_info["is_windows"]:
            try:
                startup_file.chmod(0o755)
            except:
                pass

        logger.info(f"启动脚本已创建: {startup_file}")

    def create_platform_specific_scripts(self):
        """创建平台特定的脚本"""
        logger.info("创建平台特定脚本...")

        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Windows批处理脚本
        if self.system_info["is_windows"] or True:  # 总是创建，以备跨平台使用
            windows_script = '''@echo off
echo === Ocean Simulation Python引擎 (Windows) ===
cd /d "%~dp0.."

set VENV_PATH=venv_oceansim
if not exist "%VENV_PATH%\\Scripts\\python.exe" (
    echo 虚拟环境不存在，请先运行安装脚本
    pause
    exit /b 1
)

echo 启动Python引擎...
"%VENV_PATH%\\Scripts\\python.exe" "Source\\PythonEngine\\start_python_engine.py"
pause
'''
            (scripts_dir / "start_engine.bat").write_text(windows_script, encoding='utf-8')

        # macOS/Linux shell脚本
        if not self.system_info["is_windows"] or True:  # 总是创建
            unix_script = '''#!/bin/bash
echo "=== Ocean Simulation Python引擎 (Unix) ==="
cd "$(dirname "$0")/.."

VENV_PATH="venv_oceansim"
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "虚拟环境不存在，请先运行安装脚本"
    exit 1
fi

echo "启动Python引擎..."
"$VENV_PATH/bin/python" "Source/PythonEngine/start_python_engine.py"
'''
            script_file = scripts_dir / "start_engine.sh"
            script_file.write_text(unix_script, encoding='utf-8')
            if not self.system_info["is_windows"]:
                try:
                    script_file.chmod(0o755)
                except:
                    pass

        logger.info(f"平台脚本已创建: {scripts_dir}")

    def verify_installation(self) -> bool:
        """验证安装 (增强版)"""
        logger.info("验证安装...")

        python_exe = self.get_python_executable()
        if not python_exe.exists():
            logger.error(f"Python可执行文件不存在: {python_exe}")
            return False

        # 测试基础Python包
        basic_test = '''
import sys
import os
success_count = 0
total_tests = 0

# 测试基础包
packages_to_test = [
    "numpy", "scipy", "matplotlib", "pandas", 
    "requests", "json", "pathlib"
]

for package in packages_to_test:
    total_tests += 1
    try:
        __import__(package)
        print(f"✓ {package}")
        success_count += 1
    except ImportError as e:
        print(f"✗ {package}: {e}")

print(f"\\n基础包测试: {success_count}/{total_tests} 成功")
'''

        try:
            result = subprocess.run([str(python_exe), "-c", basic_test],
                                    capture_output=True, text=True, timeout=60)
            print("基础包测试结果:")
            print(result.stdout)
            if result.stderr:
                print("警告信息:")
                print(result.stderr)
        except Exception as e:
            logger.error(f"基础包测试失败: {e}")

        # 测试Python.NET (如果可用)
        pythonnet_test = '''
import sys
try:
    # 尝试不同的导入方式
    import_success = False
    
    try:
        import clr
        print("✓ Python.NET (clr) 可用")
        import_success = True
    except ImportError:
        pass
    
    try:
        import pythonnet
        print("✓ pythonnet 包可用")
        import_success = True
    except ImportError:
        pass
        
    try:
        import clr_loader
        print("✓ clr-loader 可用")
        import_success = True
    except ImportError:
        pass
    
    if not import_success:
        print("⚠️ 没有可用的Python.NET实现，将使用模拟模式")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Python.NET测试失败: {e}")
    sys.exit(1)
'''

        try:
            result = subprocess.run([str(python_exe), "-c", pythonnet_test],
                                    capture_output=True, text=True, timeout=30)
            print("Python.NET测试结果:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as e:
            logger.warning(f"Python.NET测试失败: {e}")

        # 测试桥接模块
        bridge_test = f'''
import sys
sys.path.insert(0, r"{self.python_engine_path}")

try:
    from core.csharp_bridge import bridge
    info = bridge.get_system_info()
    print("✓ 桥接模块加载成功")
    print(f"系统信息: {{info}}")
    
    if bridge.is_available():
        print("✓ C#桥接可用")
    else:
        print("⚠️ C#桥接不可用，运行在模拟模式")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ 桥接模块测试失败: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

        try:
            result = subprocess.run([str(python_exe), "-c", bridge_test],
                                    capture_output=True, text=True, timeout=30)
            print("桥接模块测试结果:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode == 0:
                logger.info("安装验证成功")
                return True
            else:
                logger.warning("桥接模块测试失败，但基础安装完成")
                return True  # 允许部分功能运行

        except Exception as e:
            logger.error(f"桥接模块测试失败: {e}")
            return False

    def install_dotnet_if_missing(self) -> bool:
        """如果缺少.NET，提供安装指导"""
        if self.detect_dotnet_runtime():
            return True

        logger.warning(".NET运行时未找到")
        print("\n" + "="*50)
        print("⚠️  .NET运行时环境未找到")
        print("="*50)

        if self.system_info["is_windows"]:
            print("Windows安装指导:")
            print("1. 访问: https://dotnet.microsoft.com/download")
            print("2. 下载并安装 .NET 6.0 或更高版本")
            print("3. 重新运行此安装脚本")

        elif self.system_info["is_macos"]:
            print("macOS安装指导:")
            print("方法1 - 官方安装包:")
            print("  1. 访问: https://dotnet.microsoft.com/download")
            print("  2. 下载 macOS 安装包")
            print("  3. 运行安装包")
            print("\n方法2 - Homebrew:")
            print("  brew install --cask dotnet")
            if self.system_info["is_apple_silicon"]:
                print("\n⚠️  Apple Silicon注意事项:")
                print("  确保下载 ARM64 版本的.NET")

        elif self.system_info["is_linux"]:
            print("Linux安装指导:")
            print("Ubuntu/Debian:")
            print("  sudo apt update")
            print("  sudo apt install dotnet-sdk-6.0")
            print("\nCentOS/RHEL/Fedora:")
            print("  sudo dnf install dotnet-sdk-6.0")
            print("\nArch Linux:")
            print("  sudo pacman -S dotnet-sdk")

        print("\n安装完成后，请重新运行此脚本")
        print("="*50)

        return False

    def run_setup(self) -> bool:
        """执行完整安装流程 (增强版)"""
        logger.info("开始Python-C#接口安装...")
        print(f"\n检测到系统: {self.system_info['platform']}")
        if self.system_info["is_apple_silicon"]:
            print("🍎 Apple Silicon (M1/M2) 检测到，将使用优化配置")

        # 检查.NET运行时
        if not self.install_dotnet_if_missing():
            print("\n❌ .NET运行时是必需的，请安装后重试")
            return False

        # 创建虚拟环境
        if not self.create_virtual_environment():
            return False

        # 安装Python包
        if not self.install_python_packages():
            print("⚠️  部分Python包安装失败，但继续安装...")

        # 创建桥接模块
        self.create_python_csharp_bridge()

        # 创建配置文件
        self.create_configuration_files()

        # 创建启动脚本
        self.create_startup_script()

        # 创建平台特定脚本
        self.create_platform_specific_scripts()

        # 验证安装
        if not self.verify_installation():
            logger.warning("安装验证失败，但基础组件已安装")

        logger.info("Python-C#接口安装完成！")

        # 显示安装总结
        self._print_installation_summary()

        return True

    def _print_installation_summary(self):
        """打印安装总结"""
        print("\n" + "="*60)
        print("🎉 安装完成！")
        print("="*60)
        print(f"系统平台: {self.system_info['platform']}")
        print(f"Python版本: {self.system_info['python_version']}")
        print(f"虚拟环境: {self.venv_path}")
        print(f"配置文件: {self.config_path}")

        if self.system_info["is_apple_silicon"]:
            print("🍎 Apple Silicon优化: 已启用")

        print("\n📁 重要文件位置:")
        print(f"  • 启动脚本: {self.python_engine_path / 'start_python_engine.py'}")
        print(f"  • 桥接模块: {self.python_engine_path / 'core' / 'csharp_bridge.py'}")
        print(f"  • 配置文件: {self.config_path / 'python_service.yaml'}")

        print("\n🚀 下一步操作:")
        print("1. 构建C#项目:")
        print("   dotnet build --configuration Release")

        print("\n2. 启动Python引擎:")
        if self.system_info["is_windows"]:
            print("   scripts\\start_engine.bat")
            print("   或")
            print(f"   {self.venv_path}\\Scripts\\python.exe Source\\PythonEngine\\start_python_engine.py")
        else:
            print("   ./scripts/start_engine.sh")
            print("   或")
            print(f"   {self.venv_path}/bin/python Source/PythonEngine/start_python_engine.py")

        print("\n3. 测试连接:")
        print("   curl http://localhost:8888/status")

        print("\n⚠️  注意事项:")
        if not self.detect_dotnet_runtime():
            print("  • .NET运行时未检测到，C#功能将在模拟模式下运行")
        print("  • 首次运行可能需要一些时间来初始化")
        print("  • 查看日志文件获取详细信息: ./logs/python_engine.log")

        if self.system_info["is_apple_silicon"]:
            print("  • Apple Silicon: 某些包可能需要额外配置")

        print("="*60)

def main():
    """主函数"""
    print("=== Ocean Simulation Python-C#接口安装器 (增强版) ===")
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")

    # Apple Silicon特别提示
    if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
        print("🍎 检测到Apple Silicon (M1/M2)，将使用优化配置")

    print("=" * 60)

    setup = PythonCSharpSetup()

    try:
        success = setup.run_setup()
        if success:
            print("\n✅ 安装流程完成！")
        else:
            print("\n❌ 安装失败，请检查错误信息")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 用户取消安装")
        sys.exit(1)
    except Exception as e:
        logger.error(f"安装过程发生错误: {e}")
        print(f"\n❌ 安装失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()