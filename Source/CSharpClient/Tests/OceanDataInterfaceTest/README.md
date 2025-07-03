# OceanDataInterface 测试程序

## 功能
测试 `OceanDataInterface.cs` 类的各种功能：
- Python引擎初始化
- 矢量场可视化生成
- NetCDF文件处理
- 数据统计分析
- Shapefile导出

## 运行方法

### 方法1: 使用脚本
```bash
./run_test.sh
```

### 方法2: 手动运行
```bash
dotnet build
dotnet run
```

## 准备工作
1. 确保Python环境已配置
2. 确保PythonEngine目录存在并包含必要的Python脚本
3. (可选) 准备测试NetCDF文件

## 输出
- 测试结果会在控制台显示
- 生成的图像和文件会保存在 `./TestOutput` 目录中
