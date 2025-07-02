// ==============================================================================
// 文件路径：Source/CSharpClient/Core/OceanSimWrapper.cs
// 作者：beilsm
// 版本号：v1.0.0
// 创建时间：2025-07-01
// 最新更改时间：2025-07-01
// ==============================================================================
// 📝 功能说明：
//   C++核心模块的C#高层封装类
//   提供面向对象的API接口，隐藏底层P/Invoke细节
// ==============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;
using OceanSim.Core.Interop;
using static OceanSim.Core.Interop.NativeMethods;
using System.Runtime.InteropServices;


namespace OceanSim.Core
{
    // ===========================================
    // 库管理器
    // ===========================================

    /// <summary>
    /// OceanSim库管理器
    /// </summary>
    public static class OceanSimLibrary
    {
        private static bool _isInitialized = false;
        private static readonly object _lockObject = new object();

        /// <summary>
        /// 初始化库
        /// </summary>
        public static bool Initialize()
        {
            lock (_lockObject)
            {
                if (!_isInitialized)
                {
                    int result = OceanSim_Initialize();
                    _isInitialized = result == 0;
                }
                return _isInitialized;
            }
        }

        /// <summary>
        /// 清理库资源
        /// </summary>
        public static void Cleanup()
        {
            lock (_lockObject)
            {
                if (_isInitialized)
                {
                    OceanSim_Cleanup();
                    _isInitialized = false;
                }
            }
        }

        /// <summary>
        /// 获取库版本
        /// </summary>
        public static string Version
        {
            get
            {
                var ptr = NativeMethods.OceanSim_GetVersion();
                var result = Marshal.PtrToStringAnsi(ptr);
                NativeMethods.OceanSim_FreeString(ptr);
                return result;
            }
        }


        /// <summary>
        /// 设置日志级别
        /// </summary>
        public static void SetLogLevel(LogLevel level)
        {
            OceanSim_SetLogLevel((int)level);
        }

        /// <summary>
        /// 检查库是否已初始化
        /// </summary>
        public static bool IsInitialized => _isInitialized;
    }

    /// <summary>
    /// 日志级别枚举
    /// </summary>
    public enum LogLevel
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }

    // ===========================================
    // 3D向量扩展
    // ===========================================

    /// <summary>
    /// 3D向量扩展方法
    /// </summary>
    public static class Vector3Extensions
    {
        public static Vector3D ToNative(this Vector3 vector)
        {
            return new Vector3D(vector.X, vector.Y, vector.Z);
        }

        public static Vector3 ToManaged(this Vector3D vector)
        {
            return new Vector3((float)vector.X, (float)vector.Y, (float)vector.Z);
        }
    }

    // ===========================================
    // 网格数据结构封装
    // ===========================================

    /// <summary>
    /// 网格数据结构封装类
    /// </summary>
    public class GridDataStructureWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public GridDataStructureWrapper(int nx, int ny, int nz, double dx, double dy, double dz,
            Vector3 origin, CoordinateSystemType coordSystem = CoordinateSystemType.Cartesian,
            GridTypeEnum gridType = GridTypeEnum.Regular) 
            : base(IntPtr.Zero)
        {
            var config = new GridConfig
            {
                Nx = nx, Ny = ny, Nz = nz,
                Dx = dx, Dy = dy, Dz = dz,
                Origin = origin.ToNative(),
                CoordinateSystem = (int)coordSystem,
                GridType = (int)gridType
            };

            _handle = Grid_Create(ref config);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create GridDataStructure");
        }

        /// <summary>
        /// 获取网格维度
        /// </summary>
        public (int Nx, int Ny, int Nz) GetDimensions()
        {
            ThrowIfDisposed();
            Grid_GetDimensions(_handle, out int nx, out int ny, out int nz);
            return (nx, ny, nz);
        }

        /// <summary>
        /// 设置标量场
        /// </summary>
        public void SetScalarField(string fieldName, double[] data)
        {
            ThrowIfDisposed();
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (string.IsNullOrEmpty(fieldName)) throw new ArgumentNullException(nameof(fieldName));
            
            Grid_SetScalarField(_handle, data, data.Length, fieldName);
        }

        /// <summary>
        /// 设置向量场
        /// </summary>
        public void SetVectorField(string fieldName, double[] uData, double[] vData, double[] wData)
        {
            ThrowIfDisposed();
            if (uData == null || vData == null || wData == null)
                throw new ArgumentNullException("Vector field components cannot be null");
            if (uData.Length != vData.Length || vData.Length != wData.Length)
                throw new ArgumentException("Vector field components must have the same length");
            
            Grid_SetVectorField(_handle, uData, vData, wData, uData.Length, fieldName);
        }

        /// <summary>
        /// 获取标量场
        /// </summary>
        public double[] GetScalarField(string fieldName, int size)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(fieldName)) throw new ArgumentNullException(nameof(fieldName));
            
            var data = new double[size];
            Grid_GetScalarField(_handle, fieldName, data, size);
            return data;
        }

        /// <summary>
        /// 插值计算
        /// </summary>
        public double Interpolate(string fieldName, Vector3 position, InterpolationMethod method = InterpolationMethod.Linear)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(fieldName)) throw new ArgumentNullException(nameof(fieldName));
            
            var pos = position.ToNative();
            return Grid_Interpolate(_handle, ref pos, fieldName, (int)method);
        }

        protected override void ReleaseNativeResource()
        {
            Grid_Destroy(_handle);
        }
    }

    // ===========================================
    // 粒子模拟器封装
    // ===========================================

    /// <summary>
    /// 粒子数据结构
    /// </summary>
    public class Particle
    {
        public int Id { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Velocity { get; set; }
        public double Age { get; set; }
        public bool Active { get; set; }

        public Particle() { }

        public Particle(ParticleData nativeData)
        {
            Id = nativeData.Id;
            Position = nativeData.Position.ToManaged();
            Velocity = nativeData.Velocity.ToManaged();
            Age = nativeData.Age;
            Active = nativeData.Active != 0;
        }

        public ParticleData ToNative()
        {
            return new ParticleData
            {
                Id = Id,
                Position = Position.ToNative(),
                Velocity = Velocity.ToNative(),
                Age = Age,
                Active = Active ? 1 : 0
            };
        }
    }

    /// <summary>
    /// 粒子模拟器封装类
    /// </summary>
    public class ParticleSimulatorWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public ParticleSimulatorWrapper(GridDataStructureWrapper grid, RungeKuttaSolverWrapper solver)
            : base(IntPtr.Zero)
        {
            if (grid == null) throw new ArgumentNullException(nameof(grid));
            if (solver == null) throw new ArgumentNullException(nameof(solver));
            
            _handle = ParticleSim_Create(grid.Handle, solver.Handle);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create ParticleSimulator");
        }

        /// <summary>
        /// 初始化粒子
        /// </summary>
        public void InitializeParticles(IEnumerable<Vector3> positions)
        {
            ThrowIfDisposed();
            if (positions == null) throw new ArgumentNullException(nameof(positions));
            
            var positionArray = positions.Select(p => p.ToNative()).ToArray();
            ParticleSim_InitializeParticles(_handle, positionArray, positionArray.Length);
        }

        /// <summary>
        /// 执行时间步进
        /// </summary>
        public void StepForward(double timeStep)
        {
            ThrowIfDisposed();
            ParticleSim_StepForward(_handle, timeStep);
        }

        /// <summary>
        /// 获取粒子数据
        /// </summary>
        public List<Particle> GetParticles()
        {
            ThrowIfDisposed();
            int count = ParticleSim_GetParticleCount(_handle);
            if (count == 0) return new List<Particle>();
            
            var nativeParticles = new ParticleData[count];
            ParticleSim_GetParticles(_handle, nativeParticles, count);
            
            return nativeParticles.Select(p => new Particle(p)).ToList();
        }

        /// <summary>
        /// 获取粒子数量
        /// </summary>
        public int ParticleCount
        {
            get
            {
                ThrowIfDisposed();
                return ParticleSim_GetParticleCount(_handle);
            }
        }

        protected override void ReleaseNativeResource()
        {
            ParticleSim_Destroy(_handle);
        }
    }

    // ===========================================
    // 洋流场求解器封装
    // ===========================================

    /// <summary>
    /// 洋流场求解器封装类
    /// </summary>
    public class CurrentFieldSolverWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public CurrentFieldSolverWrapper(GridDataStructureWrapper grid, PhysicalParameters parameters)
            : base(IntPtr.Zero)
        {
            if (grid == null) throw new ArgumentNullException(nameof(grid));
            
            _handle = CurrentSolver_Create(grid.Handle, ref parameters);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create CurrentFieldSolver");
        }

        /// <summary>
        /// 计算速度场
        /// </summary>
        public void ComputeVelocityField(double timeStep)
        {
            ThrowIfDisposed();
            CurrentSolver_ComputeVelocityField(_handle, timeStep);
        }

        /// <summary>
        /// 设置边界条件
        /// </summary>
        public void SetBoundaryConditions(int boundaryType, double[] values)
        {
            ThrowIfDisposed();
            if (values == null) throw new ArgumentNullException(nameof(values));
            
            CurrentSolver_SetBoundaryConditions(_handle, boundaryType, values, values.Length);
        }

        /// <summary>
        /// 计算动能
        /// </summary>
        public double ComputeKineticEnergy()
        {
            ThrowIfDisposed();
            return CurrentSolver_ComputeKineticEnergy(_handle);
        }

        /// <summary>
        /// 检查质量守恒
        /// </summary>
        public double CheckMassConservation()
        {
            ThrowIfDisposed();
            return CurrentSolver_CheckMassConservation(_handle);
        }

        protected override void ReleaseNativeResource()
        {
            CurrentSolver_Destroy(_handle);
        }
    }

    // ===========================================
    // 平流扩散求解器封装
    // ===========================================

    /// <summary>
    /// 平流扩散求解器封装类
    /// </summary>
    public class AdvectionDiffusionSolverWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public AdvectionDiffusionSolverWrapper(GridDataStructureWrapper grid, SolverParameters parameters)
            : base(IntPtr.Zero)
        {
            if (grid == null) throw new ArgumentNullException(nameof(grid));
            
            _handle = AdvectionSolver_Create(grid.Handle, ref parameters);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create AdvectionDiffusionSolver");
        }

        /// <summary>
        /// 设置初始条件
        /// </summary>
        public void SetInitialCondition(double[] initialField)
        {
            ThrowIfDisposed();
            if (initialField == null) throw new ArgumentNullException(nameof(initialField));
            
            AdvectionSolver_SetInitialCondition(_handle, initialField, initialField.Length);
        }

        /// <summary>
        /// 设置速度场
        /// </summary>
        public void SetVelocityField(double[] uField, double[] vField, double[] wField)
        {
            ThrowIfDisposed();
            if (uField == null || vField == null || wField == null)
                throw new ArgumentNullException("Velocity field components cannot be null");
            if (uField.Length != vField.Length || vField.Length != wField.Length)
                throw new ArgumentException("Velocity field components must have the same length");
            
            AdvectionSolver_SetVelocityField(_handle, uField, vField, wField, uField.Length);
        }

        /// <summary>
        /// 执行求解
        /// </summary>
        public double[] Solve(double timeEnd, int outputSize)
        {
            ThrowIfDisposed();
            var outputField = new double[outputSize];
            AdvectionSolver_Solve(_handle, timeEnd, outputField, outputSize);
            return outputField;
        }

        protected override void ReleaseNativeResource()
        {
            AdvectionSolver_Destroy(_handle);
        }
    }

    // ===========================================
    // 龙格-库塔求解器封装
    // ===========================================

    /// <summary>
    /// 龙格-库塔求解器封装类
    /// </summary>
    public class RungeKuttaSolverWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public RungeKuttaSolverWrapper(int order, double timeStep) : base(IntPtr.Zero)
        {
            if (order < 2 || order > 4)
                throw new ArgumentOutOfRangeException(nameof(order), "Order must be between 2 and 4");
            
            _handle = RungeKutta_Create(order, timeStep);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create RungeKuttaSolver");
        }

        /// <summary>
        /// 时间步长属性
        /// </summary>
        public double TimeStep
        {
            get
            {
                ThrowIfDisposed();
                return RungeKutta_GetTimeStep(_handle);
            }
            set
            {
                ThrowIfDisposed();
                RungeKutta_SetTimeStep(_handle, value);
            }
        }

        protected override void ReleaseNativeResource()
        {
            RungeKutta_Destroy(_handle);
        }
    }

    // ===========================================
    // 有限差分求解器封装
    // ===========================================

    /// <summary>
    /// 有限差分求解器封装类
    /// </summary>
    public class FiniteDifferenceSolverWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public FiniteDifferenceSolverWrapper(int gridSize, double spacing) : base(IntPtr.Zero)
        {
            _handle = FiniteDiff_Create(gridSize, spacing);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create FiniteDifferenceSolver");
        }

        /// <summary>
        /// 计算一阶导数
        /// </summary>
        public double[] ComputeFirstDerivative(double[] input, int direction)
        {
            ThrowIfDisposed();
            if (input == null) throw new ArgumentNullException(nameof(input));
            
            var output = new double[input.Length];
            FiniteDiff_ComputeFirstDerivative(_handle, input, output, input.Length, direction);
            return output;
        }

        /// <summary>
        /// 计算二阶导数
        /// </summary>
        public double[] ComputeSecondDerivative(double[] input, int direction)
        {
            ThrowIfDisposed();
            if (input == null) throw new ArgumentNullException(nameof(input));
            
            var output = new double[input.Length];
            FiniteDiff_ComputeSecondDerivative(_handle, input, output, input.Length, direction);
            return output;
        }

        protected override void ReleaseNativeResource()
        {
            FiniteDiff_Destroy(_handle);
        }
    }

    // ===========================================
    // 向量化运算封装
    // ===========================================

    /// <summary>
    /// 向量化运算封装类
    /// </summary>
    public class VectorizedOperationsWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public VectorizedOperationsWrapper(PerformanceConfig config) : base(IntPtr.Zero)
        {
            _handle = VectorOps_Create(ref config);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create VectorizedOperations");
        }

        /// <summary>
        /// 向量加法
        /// </summary>
        public double[] Add(double[] a, double[] b)
        {
            ThrowIfDisposed();
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException("Arrays must have the same length");
            
            var result = new double[a.Length];
            VectorOps_Add(_handle, a, b, result, a.Length);
            return result;
        }

        /// <summary>
        /// 向量减法
        /// </summary>
        public double[] Subtract(double[] a, double[] b)
        {
            ThrowIfDisposed();
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException("Arrays must have the same length");
            
            var result = new double[a.Length];
            VectorOps_Sub(_handle, a, b, result, a.Length);
            return result;
        }

        /// <summary>
        /// 点积运算
        /// </summary>
        public double DotProduct(double[] a, double[] b)
        {
            ThrowIfDisposed();
            if (a == null || b == null) throw new ArgumentNullException();
            if (a.Length != b.Length) throw new ArgumentException("Arrays must have the same length");
            
            return VectorOps_DotProduct(_handle, a, b, a.Length);
        }

        /// <summary>
        /// 向量范数
        /// </summary>
        public double Norm(double[] a)
        {
            ThrowIfDisposed();
            if (a == null) throw new ArgumentNullException(nameof(a));
            
            return VectorOps_Norm(_handle, a, a.Length);
        }

        protected override void ReleaseNativeResource()
        {
            VectorOps_Destroy(_handle);
        }
    }

    // ===========================================
    // 并行计算引擎封装
    // ===========================================

    /// <summary>
    /// 并行计算引擎封装类
    /// </summary>
    public class ParallelComputeEngineWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public ParallelComputeEngineWrapper(PerformanceConfig config) : base(IntPtr.Zero)
        {
            _handle = ParallelEngine_Create(ref config);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create ParallelComputeEngine");
        }

        /// <summary>
        /// 线程数量属性
        /// </summary>
        public int ThreadCount
        {
            get
            {
                ThrowIfDisposed();
                return ParallelEngine_GetThreadCount(_handle);
            }
            set
            {
                ThrowIfDisposed();
                ParallelEngine_SetThreadCount(_handle, value);
            }
        }

        protected override void ReleaseNativeResource()
        {
            ParallelEngine_Destroy(_handle);
        }
    }

    // ===========================================
    // 数据导出器封装
    // ===========================================

    /// <summary>
    /// 数据导出器封装类
    /// </summary>
    public class DataExporterWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public DataExporterWrapper() : base(IntPtr.Zero)
        {
            _handle = DataExporter_Create();
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create DataExporter");
        }

        /// <summary>
        /// 导出为NetCDF格式
        /// </summary>
        public bool ExportToNetCDF(GridDataStructureWrapper grid, string filename)
        {
            ThrowIfDisposed();
            if (grid == null) throw new ArgumentNullException(nameof(grid));
            if (string.IsNullOrEmpty(filename)) throw new ArgumentNullException(nameof(filename));
            
            return DataExporter_ExportToNetCDF(_handle, grid.Handle, filename) == 1;
        }

        /// <summary>
        /// 导出为VTK格式
        /// </summary>
        public bool ExportToVTK(GridDataStructureWrapper grid, string filename)
        {
            ThrowIfDisposed();
            if (grid == null) throw new ArgumentNullException(nameof(grid));
            if (string.IsNullOrEmpty(filename)) throw new ArgumentNullException(nameof(filename));
            
            return DataExporter_ExportToVTK(_handle, grid.Handle, filename) == 1;
        }

        protected override void ReleaseNativeResource()
        {
            DataExporter_Destroy(_handle);
        }
    }

    // ===========================================
    // 性能分析器封装
    // ===========================================

    /// <summary>
    /// 性能分析器封装类
    /// </summary>
    public class PerformanceProfilerWrapper : SafeNativeWrapper
    {
        /// <summary>
        /// 构造函数
        /// </summary>
        public PerformanceProfilerWrapper() : base(IntPtr.Zero)
        {
            _handle = Profiler_Create();
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create PerformanceProfiler");
        }

        /// <summary>
        /// 开始计时
        /// </summary>
        public void StartTiming(string sectionName)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(sectionName)) throw new ArgumentNullException(nameof(sectionName));
            
            Profiler_StartTiming(_handle, sectionName);
        }

        /// <summary>
        /// 结束计时
        /// </summary>
        public void EndTiming(string sectionName)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(sectionName)) throw new ArgumentNullException(nameof(sectionName));
            
            Profiler_EndTiming(_handle, sectionName);
        }

        /// <summary>
        /// 获取耗时
        /// </summary>
        public double GetElapsedTime(string sectionName)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(sectionName)) throw new ArgumentNullException(nameof(sectionName));
            
            return Profiler_GetElapsedTime(_handle, sectionName);
        }

        /// <summary>
        /// 生成性能报告
        /// </summary>
        public void GenerateReport(string filename)
        {
            ThrowIfDisposed();
            if (string.IsNullOrEmpty(filename)) throw new ArgumentNullException(nameof(filename));
            
            Profiler_GenerateReport(_handle, filename);
        }

        /// <summary>
        /// 计时器辅助类 - 支持using语法
        /// </summary>
        public class TimingScope : IDisposable
        {
            private readonly PerformanceProfilerWrapper _profiler;
            private readonly string _sectionName;

            public TimingScope(PerformanceProfilerWrapper profiler, string sectionName)
            {
                _profiler = profiler;
                _sectionName = sectionName;
                _profiler.StartTiming(_sectionName);
            }

            public void Dispose()
            {
                _profiler.EndTiming(_sectionName);
            }
        }

        /// <summary>
        /// 创建计时范围
        /// </summary>
        public TimingScope CreateTimingScope(string sectionName)
        {
            return new TimingScope(this, sectionName);
        }

        protected override void ReleaseNativeResource()
        {
            Profiler_Destroy(_handle);
        }
    }

    // ===========================================
    // 便捷工厂类
    // ===========================================

    /// <summary>
    /// OceanSim对象工厂类
    /// </summary>
    public static class OceanSimFactory
    {
        /// <summary>
        /// 创建默认网格
        /// </summary>
        public static GridDataStructureWrapper CreateDefaultGrid(int nx = 100, int ny = 100, int nz = 50)
        {
            return new GridDataStructureWrapper(nx, ny, nz, 1.0, 1.0, 1.0, Vector3.Zero);
        }

        /// <summary>
        /// 创建默认物理参数
        /// </summary>
        public static PhysicalParameters CreateDefaultPhysicalParameters()
        {
            return new PhysicalParameters
            {
                Density = 1025.0,
                Viscosity = 1e-6,
                Gravity = 9.81,
                CoriolisParam = 1e-4,
                WindStressX = 0.1,
                WindStressY = 0.05
            };
        }

        /// <summary>
        /// 创建默认求解器参数
        /// </summary>
        public static SolverParameters CreateDefaultSolverParameters()
        {
            return new SolverParameters
            {
                TimeStep = 0.01,
                DiffusionCoeff = 1e-3,
                SchemeType = (int)NumericalSchemeType.TvdSuperbee,
                IntegrationMethod = (int)TimeIntegrationType.RungeKutta4,
                CflNumber = 0.5
            };
        }

        /// <summary>
        /// 创建默认性能配置
        /// </summary>
        public static PerformanceConfig CreateDefaultPerformanceConfig()
        {
            return new PerformanceConfig
            {
                ExecutionPolicy = (int)ExecutionPolicyType.Parallel,
                NumThreads = Environment.ProcessorCount,
                SimdType = (int)SimdTypeEnum.AVX2,
                Priority = 0
            };
        }
    }
}