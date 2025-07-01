// ==============================================================================
// 文件路径：Source/CSharpEngine/Examples/OceanSimExample.cs
// 作者：beilsm
// 版本号：v1.0.0
// 创建时间：2025-07-01
// 最新更改时间：2025-07-01
// ==============================================================================
// 📝 功能说明：
//   OceanSim C#接口使用示例
//   展示如何使用封装类进行海洋模拟计算
// ==============================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using OceanSim.Core;
using OceanSim.Core.Interop;

namespace OceanSim.Examples
{
    /// <summary>
    /// OceanSim使用示例类
    /// </summary>
    public class OceanSimExample
    {
        /// <summary>
        /// 主程序入口
        /// </summary>
        public static void Main(string[] args)
        {
            Console.WriteLine("=== OceanSim C# Interface Example ===");
            Console.WriteLine($"Library Version: {OceanSimLibrary.Version}");

            try
            {
                // 初始化库
                if (!OceanSimLibrary.Initialize())
                {
                    Console.WriteLine("Failed to initialize OceanSim library");
                    return;
                }

                Console.WriteLine("Library initialized successfully");
                OceanSimLibrary.SetLogLevel(LogLevel.Info);

                // 运行示例
                RunBasicGridExample();
                RunParticleTrackingExample();
                RunCurrentFieldExample();
                RunAdvectionDiffusionExample();
                RunVectorOperationsExample();
                RunPerformanceProfilingExample();

                Console.WriteLine("All examples completed successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            finally
            {
                // 清理资源
                OceanSimLibrary.Cleanup();
                Console.WriteLine("Library cleanup completed");
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        /// <summary>
        /// 基础网格操作示例
        /// </summary>
        private static void RunBasicGridExample()
        {
            Console.WriteLine("\n=== Basic Grid Example ===");

            using (var grid = OceanSimFactory.CreateDefaultGrid(50, 50, 25))
            {
                Console.WriteLine($"Grid created with dimensions: {grid.GetDimensions()}");

                // 创建测试数据
                var scalarData = CreateTestScalarField(50 * 50 * 25);
                grid.SetScalarField("temperature", scalarData);

                // 创建速度场
                var (uData, vData, wData) = CreateTestVelocityField(50 * 50 * 25);
                grid.SetVectorField("velocity", uData, vData, wData);

                // 测试插值
                var position = new Vector3(25.5f, 25.5f, 12.5f);
                var interpolatedValue = grid.Interpolate("temperature", position);
                Console.WriteLine($"Interpolated temperature at {position}: {interpolatedValue:F4}");

                // 获取标量场数据
                var retrievedData = grid.GetScalarField("temperature", scalarData.Length);
                Console.WriteLine($"Retrieved {retrievedData.Length} data points");
            }
        }

        /// <summary>
        /// 粒子追踪示例
        /// </summary>
        private static void RunParticleTrackingExample()
        {
            Console.WriteLine("\n=== Particle Tracking Example ===");

            using (var grid = OceanSimFactory.CreateDefaultGrid(50, 50, 25))
            using (var solver = new RungeKuttaSolverWrapper(4, 0.01))
            using (var particleSim = new ParticleSimulatorWrapper(grid, solver))
            {
                // 设置速度场
                var (uData, vData, wData) = CreateTestVelocityField(50 * 50 * 25);
                grid.SetVectorField("velocity", uData, vData, wData);

                // 初始化粒子
                var initialPositions = new List<Vector3>
                {
                    new Vector3(10, 10, 5),
                    new Vector3(20, 20, 10),
                    new Vector3(30, 30, 15),
                    new Vector3(40, 40, 20)
                };

                particleSim.InitializeParticles(initialPositions);
                Console.WriteLine($"Initialized {particleSim.ParticleCount} particles");

                // 模拟多个时间步
                for (int step = 0; step < 100; step++)
                {
                    particleSim.StepForward(0.01);
                    
                    if (step % 20 == 0)
                    {
                        var particles = particleSim.GetParticles();
                        Console.WriteLine($"Step {step}: Particle 0 position: ({particles[0].Position.X:F2}, {particles[0].Position.Y:F2}, {particles[0].Position.Z:F2})");
                    }
                }

                var finalParticles = particleSim.GetParticles();
                Console.WriteLine($"Final simulation: {finalParticles.Count(p => p.Active)} active particles");
            }
        }

        /// <summary>
        /// 洋流场求解示例
        /// </summary>
        private static void RunCurrentFieldExample()
        {
            Console.WriteLine("\n=== Current Field Solver Example ===");

            using (var grid = OceanSimFactory.CreateDefaultGrid(30, 30, 15))
            {
                var physParams = OceanSimFactory.CreateDefaultPhysicalParameters();
                using (var currentSolver = new CurrentFieldSolverWrapper(grid, physParams))
                {
                    // 计算速度场
                    currentSolver.ComputeVelocityField(0.01);
                    Console.WriteLine("Velocity field computed");

                    // 设置边界条件
                    var boundaryValues = new double[30]; // 简化的边界条件
                    for (int i = 0; i < boundaryValues.Length; i++)
                    {
                        boundaryValues[i] = 0.1 * Math.Sin(2 * Math.PI * i / boundaryValues.Length);
                    }
                    currentSolver.SetBoundaryConditions(0, boundaryValues);

                    // 计算动能
                    var kineticEnergy = currentSolver.ComputeKineticEnergy();
                    Console.WriteLine($"Total kinetic energy: {kineticEnergy:E4} J");

                    // 检查质量守恒
                    var massError = currentSolver.CheckMassConservation();
                    Console.WriteLine($"Mass conservation error: {massError:E6}");
                }
            }
        }

        /// <summary>
        /// 平流扩散求解示例
        /// </summary>
        private static void RunAdvectionDiffusionExample()
        {
            Console.WriteLine("\n=== Advection-Diffusion Solver Example ===");

            using (var grid = OceanSimFactory.CreateDefaultGrid(40, 40, 20))
            {
                var solverParams = OceanSimFactory.CreateDefaultSolverParameters();
                using (var advectionSolver = new AdvectionDiffusionSolverWrapper(grid, solverParams))
                {
                    // 设置初始条件 - 高斯分布
                    var initialField = CreateGaussianField(40, 40, 20, 20, 20, 10, 5.0);
                    advectionSolver.SetInitialCondition(initialField);
                    Console.WriteLine("Initial condition set");

                    // 设置速度场 - 简单的旋转流
                    var (uField, vField, wField) = CreateRotationalFlow(40, 40, 20);
                    advectionSolver.SetVelocityField(uField, vField, wField);
                    Console.WriteLine("Velocity field set");

                    // 求解
                    var result = advectionSolver.Solve(1.0, initialField.Length);
                    Console.WriteLine($"Advection-diffusion solved, result size: {result.Length}");

                    // 分析结果
                    var maxValue = result.Max();
                    var minValue = result.Min();
                    var totalMass = result.Sum();
                    Console.WriteLine($"Result statistics - Max: {maxValue:F4}, Min: {minValue:F4}, Total mass: {totalMass:F4}");
                }
            }
        }

        /// <summary>
        /// 向量运算示例
        /// </summary>
        private static void RunVectorOperationsExample()
        {
            Console.WriteLine("\n=== Vector Operations Example ===");

            var perfConfig = OceanSimFactory.CreateDefaultPerformanceConfig();
            using (var vectorOps = new VectorizedOperationsWrapper(perfConfig))
            {
                // 创建测试向量
                var size = 100000;
                var vectorA = Enumerable.Range(0, size).Select(i => Math.Sin(i * 0.01)).ToArray();
                var vectorB = Enumerable.Range(0, size).Select(i => Math.Cos(i * 0.01)).ToArray();

                Console.WriteLine($"Created test vectors of size {size}");

                // 向量加法
                var addResult = vectorOps.Add(vectorA, vectorB);
                Console.WriteLine($"Vector addition completed, first element: {addResult[0]:F6}");

                // 向量减法
                var subResult = vectorOps.Subtract(vectorA, vectorB);
                Console.WriteLine($"Vector subtraction completed, first element: {subResult[0]:F6}");

                // 点积
                var dotProduct = vectorOps.DotProduct(vectorA, vectorB);
                Console.WriteLine($"Dot product: {dotProduct:F6}");

                // 向量范数
                var normA = vectorOps.Norm(vectorA);
                var normB = vectorOps.Norm(vectorB);
                Console.WriteLine($"Vector norms - A: {normA:F6}, B: {normB:F6}");
            }
        }

        /// <summary>
        /// 性能分析示例
        /// </summary>
        private static void RunPerformanceProfilingExample()
        {
            Console.WriteLine("\n=== Performance Profiling Example ===");

            using (var profiler = new PerformanceProfilerWrapper())
            {
                // 测试多个计算段
                using (profiler.CreateTimingScope("GridCreation"))
                {
                    using (var grid = OceanSimFactory.CreateDefaultGrid(100, 100, 50))
                    {
                        var data = CreateTestScalarField(100 * 100 * 50);
                        grid.SetScalarField("test_field", data);
                    }
                }

                using (profiler.CreateTimingScope("VectorOperations"))
                {
                    var perfConfig = OceanSimFactory.CreateDefaultPerformanceConfig();
                    using (var vectorOps = new VectorizedOperationsWrapper(perfConfig))
                    {
                        var size = 1000000;
                        var a = new double[size];
                        var b = new double[size];
                        
                        for (int i = 0; i < size; i++)
                        {
                            a[i] = Math.Random.Shared.NextDouble();
                            b[i] = Math.Random.Shared.NextDouble();
                        }

                        var result = vectorOps.Add(a, b);
                        var dotProd = vectorOps.DotProduct(a, b);
                    }
                }

                using (profiler.CreateTimingScope("FiniteDifference"))
                {
                    using (var fdSolver = new FiniteDifferenceSolverWrapper(1000, 0.01))
                    {
                        var input = Enumerable.Range(0, 1000)
                            .Select(i => Math.Sin(i * 0.01))
                            .ToArray();
                        
                        var firstDeriv = fdSolver.ComputeFirstDerivative(input, 0);
                        var secondDeriv = fdSolver.ComputeSecondDerivative(input, 0);
                    }
                }

                // 输出性能统计
                Console.WriteLine("Performance Results:");
                Console.WriteLine($"  Grid Creation: {profiler.GetElapsedTime("GridCreation"):F2} ms");
                Console.WriteLine($"  Vector Operations: {profiler.GetElapsedTime("VectorOperations"):F2} ms");
                Console.WriteLine($"  Finite Difference: {profiler.GetElapsedTime("FiniteDifference"):F2} ms");

                // 生成详细报告
                try
                {
                    profiler.GenerateReport("performance_report.txt");
                    Console.WriteLine("Performance report saved to performance_report.txt");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to generate report: {ex.Message}");
                }
            }
        }

        // ===========================================
        // 辅助方法
        // ===========================================

        /// <summary>
        /// 创建测试标量场
        /// </summary>
        private static double[] CreateTestScalarField(int size)
        {
            var data = new double[size];
            var random = new Random(42); // 固定种子以获得可重复的结果

            for (int i = 0; i < size; i++)
            {
                data[i] = 20.0 + 5.0 * Math.Sin(i * 0.01) + 2.0 * random.NextDouble();
            }

            return data;
        }

        /// <summary>
        /// 创建测试速度场
        /// </summary>
        private static (double[] u, double[] v, double[] w) CreateTestVelocityField(int size)
        {
            var u = new double[size];
            var v = new double[size];
            var w = new double[size];

            for (int i = 0; i < size; i++)
            {
                double t = i * 0.01;
                u[i] = 0.1 * Math.Cos(t);
                v[i] = 0.1 * Math.Sin(t);
                w[i] = 0.05 * Math.Sin(2 * t);
            }

            return (u, v, w);
        }

        /// <summary>
        /// 创建高斯分布场
        /// </summary>
        private static double[] CreateGaussianField(int nx, int ny, int nz, 
            double centerX, double centerY, double centerZ, double amplitude)
        {
            var field = new double[nx * ny * nz];
            var sigma = 5.0; // 标准差

            for (int k = 0; k < nz; k++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        int index = k * nx * ny + j * nx + i;
                        
                        double dx = i - centerX;
                        double dy = j - centerY;
                        double dz = k - centerZ;
                        double distSq = dx * dx + dy * dy + dz * dz;
                        
                        field[index] = amplitude * Math.Exp(-distSq / (2 * sigma * sigma));
                    }
                }
            }

            return field;
        }

        /// <summary>
        /// 创建旋转流场
        /// </summary>
        private static (double[] u, double[] v, double[] w) CreateRotationalFlow(int nx, int ny, int nz)
        {
            var u = new double[nx * ny * nz];
            var v = new double[nx * ny * nz];
            var w = new double[nx * ny * nz];

            double centerX = nx / 2.0;
            double centerY = ny / 2.0;
            double omega = 0.1; // 角速度

            for (int k = 0; k < nz; k++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        int index = k * nx * ny + j * nx + i;
                        
                        double dx = i - centerX;
                        double dy = j - centerY;
                        
                        // 旋转流：u = -omega * y', v = omega * x'
                        u[index] = -omega * dy;
                        v[index] = omega * dx;
                        w[index] = 0.01 * Math.Sin(k * 0.2); // 小的垂直流动
                    }
                }
            }

            return (u, v, w);
        }
    }

    // ===========================================
    // 单元测试示例
    // ===========================================

    /// <summary>
    /// 简单的单元测试类
    /// </summary>
    public static class OceanSimTests
    {
        /// <summary>
        /// 运行所有测试
        /// </summary>
        public static void RunAllTests()
        {
            Console.WriteLine("\n=== Running Unit Tests ===");

            var tests = new Action[]
            {
                TestLibraryInitialization,
                TestGridCreationAndDestruction,
                TestVectorOperations,
                TestRungeKuttaSolver,
                TestFiniteDifferenceSolver
            };

            int passed = 0;
            int total = tests.Length;

            foreach (var test in tests)
            {
                try
                {
                    test();
                    Console.WriteLine($"✓ {test.Method.Name} passed");
                    passed++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"✗ {test.Method.Name} failed: {ex.Message}");
                }
            }

            Console.WriteLine($"\nTest Results: {passed}/{total} tests passed");
        }

        private static void TestLibraryInitialization()
        {
            if (!OceanSimLibrary.Initialize())
                throw new Exception("Failed to initialize library");

            var version = OceanSimLibrary.Version;
            if (string.IsNullOrEmpty(version))
                throw new Exception("Version string is null or empty");
        }

        private static void TestGridCreationAndDestruction()
        {
            using (var grid = new GridDataStructureWrapper(10, 10, 10, 1.0, 1.0, 1.0, Vector3.Zero))
            {
                var dims = grid.GetDimensions();
                if (dims.Nx != 10 || dims.Ny != 10 || dims.Nz != 10)
                    throw new Exception($"Unexpected grid dimensions: {dims}");
            }
        }

        private static void TestVectorOperations()
        {
            var config = OceanSimFactory.CreateDefaultPerformanceConfig();
            using (var vectorOps = new VectorizedOperationsWrapper(config))
            {
                var a = new double[] { 1.0, 2.0, 3.0 };
                var b = new double[] { 4.0, 5.0, 6.0 };

                var sum = vectorOps.Add(a, b);
                var expectedSum = new double[] { 5.0, 7.0, 9.0 };

                for (int i = 0; i < sum.Length; i++)
                {
                    if (Math.Abs(sum[i] - expectedSum[i]) > 1e-10)
                        throw new Exception($"Vector addition failed at index {i}");
                }

                var dotProduct = vectorOps.DotProduct(a, b);
                var expectedDot = 32.0; // 1*4 + 2*5 + 3*6 = 32
                if (Math.Abs(dotProduct - expectedDot) > 1e-10)
                    throw new Exception("Dot product calculation failed");
            }
        }

        private static void TestRungeKuttaSolver()
        {
            using (var solver = new RungeKuttaSolverWrapper(4, 0.01))
            {
                if (Math.Abs(solver.TimeStep - 0.01) > 1e-12)
                    throw new Exception("Time step not set correctly");

                solver.TimeStep = 0.02;
                if (Math.Abs(solver.TimeStep - 0.02) > 1e-12)
                    throw new Exception("Time step not updated correctly");
            }
        }

        private static void TestFiniteDifferenceSolver()
        {
            using (var solver = new FiniteDifferenceSolverWrapper(100, 0.01))
            {
                // 测试一个简单的二次函数 f(x) = x^2
                var input = new double[100];
                for (int i = 0; i < 100; i++)
                {
                    double x = i * 0.01;
                    input[i] = x * x;
                }

                var firstDeriv = solver.ComputeFirstDerivative(input, 0);
                
                // 检查导数的大致正确性（应该接近 2*x）
                // 在中间位置检查（避免边界效应）
                int midIndex = 50;
                double expectedDeriv = 2 * midIndex * 0.01; // 2*x
                double actualDeriv = firstDeriv[midIndex];
                
                if (Math.Abs(actualDeriv - expectedDeriv) > 0.1) // 允许一定的数值误差
                    throw new Exception($"First derivative test failed: expected {expectedDeriv}, got {actualDeriv}");
            }
        }
    }

    // ===========================================
    // 高级应用示例
    // ===========================================

    /// <summary>
    /// 高级应用示例 - 复杂的海洋模拟场景
    /// </summary>
    public class AdvancedOceanSimulation
    {
        private GridDataStructureWrapper _grid;
        private CurrentFieldSolverWrapper _currentSolver;
        private ParticleSimulatorWrapper _particleSimulator;
        private PerformanceProfilerWrapper _profiler;

        public AdvancedOceanSimulation()
        {
            InitializeSimulation();
        }

        private void InitializeSimulation()
        {
            Console.WriteLine("\n=== Advanced Ocean Simulation ===");

            _profiler = new PerformanceProfilerWrapper();

            using (_profiler.CreateTimingScope("Initialization"))
            {
                // 创建高分辨率网格
                _grid = new GridDataStructureWrapper(200, 200, 100, 0.5, 0.5, 0.2, Vector3.Zero);

                // 设置物理参数
                var physParams = new PhysicalParameters
                {
                    Density = 1025.0,
                    Viscosity = 1e-6,
                    Gravity = 9.81,
                    CoriolisParam = 1.4e-4, // 典型的中纬度值
                    WindStressX = 0.15,
                    WindStressY = 0.08
                };

                _currentSolver = new CurrentFieldSolverWrapper(_grid, physParams);

                // 创建龙格-库塔求解器和粒子模拟器
                var rkSolver = new RungeKuttaSolverWrapper(4, 0.005);
                _particleSimulator = new ParticleSimulatorWrapper(_grid, rkSolver);

                Console.WriteLine("Advanced simulation initialized");
            }
        }

        /// <summary>
        /// 运行完整的模拟
        /// </summary>
        public void RunSimulation(int timeSteps, double timeStep)
        {
            using (_profiler.CreateTimingScope("FullSimulation"))
            {
                // 设置初始条件
                SetupInitialConditions();

                // 释放粒子
                ReleaseParticles();

                // 主时间循环
                for (int step = 0; step < timeSteps; step++)
                {
                    using (_profiler.CreateTimingScope("TimeStep"))
                    {
                        // 更新流场
                        _currentSolver.ComputeVelocityField(timeStep);

                        // 推进粒子
                        _particleSimulator.StepForward(timeStep);

                        // 每隔一定步数输出状态
                        if (step % 100 == 0)
                        {
                            ReportSimulationStatus(step, timeStep);
                        }
                    }
                }

                Console.WriteLine($"Simulation completed: {timeSteps} time steps");
            }
        }

        private void SetupInitialConditions()
        {
            Console.WriteLine("Setting up initial conditions...");

            // 设置温度场 - 带有温度梯度
            var tempField = CreateTemperatureField();
            _grid.SetScalarField("temperature", tempField);

            // 设置盐度场
            var saltField = CreateSalinityField();
            _grid.SetScalarField("salinity", saltField);

            // 设置初始速度场
            var (u, v, w) = CreateInitialVelocityField();
            _grid.SetVectorField("velocity", u, v, w);
        }

        private void ReleaseParticles()
        {
            Console.WriteLine("Releasing particles...");

            // 在不同深度释放粒子群
            var positions = new List<Vector3>();

            // 表层粒子
            for (int i = 0; i < 50; i++)
            {
                positions.Add(new Vector3(
                    50 + i * 2,
                    100 + (float)(Math.Sin(i * 0.2) * 20),
                    5
                ));
            }

            // 中层粒子
            for (int i = 0; i < 30; i++)
            {
                positions.Add(new Vector3(
                    150,
                    50 + i * 3,
                    50
                ));
            }

            // 深层粒子
            for (int i = 0; i < 20; i++)
            {
                positions.Add(new Vector3(
                    100 + (float)(Math.Cos(i * 0.3) * 30),
                    150,
                    90
                ));
            }

            _particleSimulator.InitializeParticles(positions);
            Console.WriteLine($"Released {positions.Count} particles");
        }

        private void ReportSimulationStatus(int step, double timeStep)
        {
            var currentTime = step * timeStep;
            var particles = _particleSimulator.GetParticles();
            var activeParticles = particles.Count(p => p.Active);
            var kineticEnergy = _currentSolver.ComputeKineticEnergy();
            var massError = _currentSolver.CheckMassConservation();

            Console.WriteLine($"Step {step:D4} (t={currentTime:F2}s): " +
                            $"{activeParticles} active particles, " +
                            $"KE={kineticEnergy:E3} J, " +
                            $"Mass error={massError:E6}");
        }

        private double[] CreateTemperatureField()
        {
            var dims = _grid.GetDimensions();
            var field = new double[dims.Nx * dims.Ny * dims.Nz];

            for (int k = 0; k < dims.Nz; k++)
            {
                for (int j = 0; j < dims.Ny; j++)
                {
                    for (int i = 0; i < dims.Nx; i++)
                    {
                        int index = k * dims.Nx * dims.Ny + j * dims.Nx + i;
                        
                        // 表面温度最高，随深度递减
                        double surfaceTemp = 25.0;
                        double deepTemp = 4.0;
                        double depth = k * 0.2; // 深度（米）
                        
                        // 指数衰减温度剖面
                        double temp = deepTemp + (surfaceTemp - deepTemp) * Math.Exp(-depth / 50.0);
                        
                        // 添加水平变化
                        temp += 2.0 * Math.Sin(i * 0.05) * Math.Cos(j * 0.03);
                        
                        field[index] = temp;
                    }
                }
            }

            return field;
        }

        private double[] CreateSalinityField()
        {
            var dims = _grid.GetDimensions();
            var field = new double[dims.Nx * dims.Ny * dims.Nz];

            for (int k = 0; k < dims.Nz; k++)
            {
                for (int j = 0; j < dims.Ny; j++)
                {
                    for (int i = 0; i < dims.Nx; i++)
                    {
                        int index = k * dims.Nx * dims.Ny + j * dims.Nx + i;
                        
                        // 典型的海洋盐度分布
                        double baseSalinity = 35.0;
                        double depth = k * 0.2;
                        
                        // 深层盐度稍高
                        double salinity = baseSalinity + 0.5 * Math.Tanh(depth / 100.0);
                        
                        // 添加区域变化
                        salinity += 0.2 * Math.Sin(i * 0.02) * Math.Sin(j * 0.02);
                        
                        field[index] = salinity;
                    }
                }
            }

            return field;
        }

        private (double[] u, double[] v, double[] w) CreateInitialVelocityField()
        {
            var dims = _grid.GetDimensions();
            var size = dims.Nx * dims.Ny * dims.Nz;
            var u = new double[size];
            var v = new double[size];
            var w = new double[size];

            for (int k = 0; k < dims.Nz; k++)
            {
                for (int j = 0; j < dims.Ny; j++)
                {
                    for (int i = 0; i < dims.Nx; i++)
                    {
                        int index = k * dims.Nx * dims.Ny + j * dims.Nx + i;
                        
                        double x = i * 0.5;
                        double y = j * 0.5;
                        double z = k * 0.2;
                        
                        // 创建涡旋流场
                        double centerX = dims.Nx * 0.25;
                        double centerY = dims.Ny * 0.25;
                        double dx = x - centerX;
                        double dy = y - centerY;
                        double r = Math.Sqrt(dx * dx + dy * dy);
                        
                        if (r > 0)
                        {
                            double vortexStrength = 0.1 * Math.Exp(-r / 50.0);
                            u[index] = -vortexStrength * dy / r;
                            v[index] = vortexStrength * dx / r;
                        }
                        
                        // 添加垂直流动（上升流）
                        w[index] = 0.001 * Math.Sin(x * 0.02) * Math.Cos(y * 0.02) * Math.Exp(-z / 20.0);
                    }
                }
            }

            return (u, v, w);
        }

        public void GenerateReport(string filename)
        {
            try
            {
                _profiler.GenerateReport(filename);
                Console.WriteLine($"Detailed performance report saved to {filename}");

                // 输出性能摘要
                Console.WriteLine("\nPerformance Summary:");
                Console.WriteLine($"  Initialization: {_profiler.GetElapsedTime("Initialization"):F2} ms");
                Console.WriteLine($"  Full Simulation: {_profiler.GetElapsedTime("FullSimulation"):F2} ms");
                Console.WriteLine($"  Average Time Step: {_profiler.GetElapsedTime("TimeStep"):F4} ms");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to generate report: {ex.Message}");
            }
        }

        public void Dispose()
        {
            _particleSimulator?.Dispose();
            _currentSolver?.Dispose();
            _grid?.Dispose();
            _profiler?.Dispose();
        }
    }
}