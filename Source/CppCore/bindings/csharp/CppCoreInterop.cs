// ==============================================================================
// 文件路径：Source/CppCore/bindings/csharp/CppCoreInterop.cs
// 作者：beilsm
// 版本号：v1.0.0
// 创建时间：2025-07-01
// 最新更改时间：2025-07-01
// ==============================================================================
// 📝 功能说明：
//   C++核心模块的C# P/Invoke接口定义
//   提供.NET调用C++动态库的底层接口
// ==============================================================================

using System;
using System.Runtime.InteropServices;

namespace OceanSim.Core.Interop
{
    /// <summary>
    /// C++核心模块的P/Invoke接口
    /// </summary>
    public static class NativeMethods
    {
        // 动态库名称配置
        #if WINDOWS || UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            private const string DllName = "oceansim_csharp.dll";
        #elif OSX || UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            private const string DllName = "oceansim_csharp"; // macOS自动补 lib + .dylib
        #elif LINUX || UNITY_STANDALONE_LINUX || UNITY_EDITOR_LINUX
            private const string DllName = "oceansim_csharp"; // Linux自动补 lib + .so
        #else
                private const string DllName = "oceansim_csharp";
        #endif


        // ===========================================
        // 数据结构定义
        // ===========================================

        /// <summary>
        /// 3D向量结构（双精度）
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Vector3D
        {
            public double X;
            public double Y;
            public double Z;

            public Vector3D(double x, double y, double z)
            {
                X = x;
                Y = y;
                Z = z;
            }
        }

        /// <summary>
        /// 3D向量结构（单精度）
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Vector3F
        {
            public float X;
            public float Y;
            public float Z;

            public Vector3F(float x, float y, float z)
            {
                X = x;
                Y = y;
                Z = z;
            }
        }

        /// <summary>
        /// 地理坐标结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct GeographicCoordinate
        {
            public double Latitude;
            public double Longitude;
            public double Depth;
        }

        /// <summary>
        /// 粒子数据结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ParticleData
        {
            public int Id;
            public Vector3D Position;
            public Vector3D Velocity;
            public double Age;
            public int Active; // 1 = true, 0 = false
        }

        /// <summary>
        /// 网格配置结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct GridConfig
        {
            public int Nx, Ny, Nz;           // 网格维度
            public double Dx, Dy, Dz;        // 网格间距
            public Vector3D Origin;          // 原点坐标
            public int CoordinateSystem;     // 坐标系类型
            public int GridType;            // 网格类型
        }

        /// <summary>
        /// 物理参数结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PhysicalParameters
        {
            public double Density;           // 密度
            public double Viscosity;         // 粘性系数
            public double Gravity;           // 重力
            public double CoriolisParam;     // 科里奥利参数
            public double WindStressX;       // 风应力X分量
            public double WindStressY;       // 风应力Y分量
        }

        /// <summary>
        /// 数值求解参数结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct SolverParameters
        {
            public double TimeStep;          // 时间步长
            public double DiffusionCoeff;    // 扩散系数
            public int SchemeType;          // 数值格式类型
            public int IntegrationMethod;    // 时间积分方法
            public double CflNumber;         // CFL数
        }

        /// <summary>
        /// 性能配置结构
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct PerformanceConfig
        {
            public int ExecutionPolicy;      // 执行策略
            public int NumThreads;          // 线程数量
            public int SimdType;            // SIMD类型
            public int Priority;            // 优先级
        }

        // ===========================================
        // 枚举类型定义
        // ===========================================

        /// <summary>
        /// 坐标系类型
        /// </summary>
        public enum CoordinateSystemType
        {
            Cartesian = 0,
            Spherical = 1,
            HybridSigma = 2,
            Isopycnal = 3
        }

        /// <summary>
        /// 网格类型
        /// </summary>
        public enum GridTypeEnum
        {
            Regular = 0,
            Curvilinear = 1,
            Unstructured = 2
        }

        /// <summary>
        /// 数值格式类型
        /// </summary>
        public enum NumericalSchemeType
        {
            Upwind = 0,
            Central = 1,
            TvdSuperbee = 2,
            Weno = 3
        }

        /// <summary>
        /// 时间积分方法
        /// </summary>
        public enum TimeIntegrationType
        {
            Euler = 0,
            RungeKutta2 = 1,
            RungeKutta3 = 2,
            RungeKutta4 = 3,
            AdamsBashforth = 4
        }

        /// <summary>
        /// 执行策略类型
        /// </summary>
        public enum ExecutionPolicyType
        {
            Sequential = 0,
            Parallel = 1,
            Vectorized = 2,
            HybridParallel = 3
        }

        /// <summary>
        /// SIMD类型
        /// </summary>
        public enum SimdTypeEnum
        {
            None = 0,
            SSE = 1,
            AVX = 2,
            AVX2 = 3,
            AVX512 = 4,
            NEON = 5
        }

        /// <summary>
        /// 插值方法
        /// </summary>
        public enum InterpolationMethod
        {
            Linear = 0,
            Cubic = 1,
            Bilinear = 2,
            Trilinear = 3,
            Conservative = 4
        }

        // ===========================================
        // 网格数据结构接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Grid_Create(ref GridConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Grid_Destroy(IntPtr grid);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Grid_GetDimensions(IntPtr grid, out int nx, out int ny, out int nz);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Grid_SetScalarField(IntPtr grid, double[] data, int size, 
            [MarshalAs(UnmanagedType.LPStr)] string fieldName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Grid_SetVectorField(IntPtr grid, double[] uData, double[] vData, 
            double[] wData, int size, [MarshalAs(UnmanagedType.LPStr)] string fieldName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Grid_GetScalarField(IntPtr grid, [MarshalAs(UnmanagedType.LPStr)] string fieldName, 
            double[] data, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double Grid_Interpolate(IntPtr grid, ref Vector3D position, 
            [MarshalAs(UnmanagedType.LPStr)] string fieldName, int method);

        // ===========================================
        // 粒子模拟器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ParticleSim_Create(IntPtr grid, IntPtr solverHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParticleSim_Destroy(IntPtr simulator);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParticleSim_InitializeParticles(IntPtr simulator, Vector3D[] positions, int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParticleSim_StepForward(IntPtr simulator, double timeStep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParticleSim_GetParticles(IntPtr simulator, ParticleData[] particles, int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ParticleSim_GetParticleCount(IntPtr simulator);

        // ===========================================
        // 洋流场求解器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CurrentSolver_Create(IntPtr grid, ref PhysicalParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CurrentSolver_Destroy(IntPtr solver);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CurrentSolver_ComputeVelocityField(IntPtr solver, double timeStep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CurrentSolver_SetBoundaryConditions(IntPtr solver, int boundaryType, 
            double[] values, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double CurrentSolver_ComputeKineticEnergy(IntPtr solver);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double CurrentSolver_CheckMassConservation(IntPtr solver);

        // ===========================================
        // 平流扩散求解器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr AdvectionSolver_Create(IntPtr grid, ref SolverParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void AdvectionSolver_Destroy(IntPtr solver);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void AdvectionSolver_SetInitialCondition(IntPtr solver, double[] initialField, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void AdvectionSolver_SetVelocityField(IntPtr solver, double[] uField, 
            double[] vField, double[] wField, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void AdvectionSolver_Solve(IntPtr solver, double timeEnd, double[] outputField, int size);

        // ===========================================
        // 龙格-库塔求解器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr RungeKutta_Create(int order, double timeStep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void RungeKutta_Destroy(IntPtr solver);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void RungeKutta_SetTimeStep(IntPtr solver, double timeStep);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double RungeKutta_GetTimeStep(IntPtr solver);

        // ===========================================
        // 有限差分求解器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr FiniteDiff_Create(int gridSize, double spacing);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void FiniteDiff_Destroy(IntPtr solver);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void FiniteDiff_ComputeFirstDerivative(IntPtr solver, double[] input, 
            double[] output, int size, int direction);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void FiniteDiff_ComputeSecondDerivative(IntPtr solver, double[] input, 
            double[] output, int size, int direction);

        // ===========================================
        // 向量化运算接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr VectorOps_Create(ref PerformanceConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void VectorOps_Destroy(IntPtr ops);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void VectorOps_Add(IntPtr ops, double[] a, double[] b, double[] result, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void VectorOps_Sub(IntPtr ops, double[] a, double[] b, double[] result, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double VectorOps_DotProduct(IntPtr ops, double[] a, double[] b, int size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double VectorOps_Norm(IntPtr ops, double[] a, int size);

        // ===========================================
        // 并行计算引擎接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ParallelEngine_Create(ref PerformanceConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParallelEngine_Destroy(IntPtr engine);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ParallelEngine_SetThreadCount(IntPtr engine, int numThreads);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ParallelEngine_GetThreadCount(IntPtr engine);

        // ===========================================
        // 数据导出器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr DataExporter_Create();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void DataExporter_Destroy(IntPtr exporter);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int DataExporter_ExportToNetCDF(IntPtr exporter, IntPtr grid, 
            [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int DataExporter_ExportToVTK(IntPtr exporter, IntPtr grid, 
            [MarshalAs(UnmanagedType.LPStr)] string filename);

        // ===========================================
        // 性能分析器接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr Profiler_Create();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Profiler_Destroy(IntPtr profiler);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Profiler_StartTiming(IntPtr profiler, [MarshalAs(UnmanagedType.LPStr)] string sectionName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Profiler_EndTiming(IntPtr profiler, [MarshalAs(UnmanagedType.LPStr)] string sectionName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double Profiler_GetElapsedTime(IntPtr profiler, [MarshalAs(UnmanagedType.LPStr)] string sectionName);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void Profiler_GenerateReport(IntPtr profiler, [MarshalAs(UnmanagedType.LPStr)] string filename);

        // ===========================================
        // 工具函数接口
        // ===========================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr OceanSim_GetVersion();
        
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void OceanSim_FreeString(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int OceanSim_Initialize();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void OceanSim_Cleanup();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void OceanSim_SetLogLevel(int level);
    }

    // ===========================================
    // 安全包装器基类
    // ===========================================

    /// <summary>
    /// 原生资源安全包装器基类
    /// </summary>
    public abstract class SafeNativeWrapper : IDisposable
    {
        protected IntPtr _handle;
        protected bool _disposed = false;

        protected SafeNativeWrapper(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary>
        /// 检查句柄是否有效
        /// </summary>
        public bool IsValid => _handle != IntPtr.Zero && !_disposed;

        /// <summary>
        /// 获取原生句柄
        /// </summary>
        public IntPtr Handle
        {
            get
            {
                ThrowIfDisposed();
                return _handle;
            }
        }

        /// <summary>
        /// 检查是否已释放，如果是则抛出异常
        /// </summary>
        protected void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().Name);
        }

        /// <summary>
        /// 释放原生资源
        /// </summary>
        protected abstract void ReleaseNativeResource();

        /// <summary>
        /// 实现IDisposable接口
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// 保护的Dispose方法
        /// </summary>
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    ReleaseNativeResource();
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// 析构函数
        /// </summary>
        ~SafeNativeWrapper()
        {
            Dispose(false);
        }
    }
}