// ==============================================================================
// 文件：Source/CSharpCore/Interop/EnKFInterop.cs
// 作者：beilsm
// 版本：v1.0.0
// 创建时间：2025-07-03
// 功能：集合卡尔曼滤波预测模块的C# P/Invoke接口
// 说明：提供TOPAZ系统EnKF算法的.NET调用接口
// ==============================================================================

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace OceanSimulation.Infrastructure.Interop
{
    /// <summary>
    /// 海洋状态向量结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct OceanState
    {
        public double Temperature;           // 温度 (°C)
        public double Salinity;              // 盐度 (PSU)
        public double VelocityU;             // U方向流速 (m/s)
        public double VelocityV;             // V方向流速 (m/s)
        public double VelocityW;             // W方向流速 (m/s)
        public double SeaIceConcentration;   // 海冰浓度 (0-1)
        public double SeaIceThickness;       // 海冰厚度 (m)
        public double SeaSurfaceHeight;      // 海表面高度 (m)

        /// <summary>
        /// 创建零状态
        /// </summary>
        public static OceanState Zero => new OceanState();

        /// <summary>
        /// 检查状态物理有效性
        /// </summary>
        public bool IsPhysicallyValid
        {
            get
            {
                return Temperature >= -2.0 && Temperature <= 35.0 &&
                       Salinity >= 0.0 && Salinity <= 45.0 &&
                       SeaIceConcentration >= 0.0 && SeaIceConcentration <= 1.0 &&
                       SeaIceThickness >= 0.0 && SeaIceThickness <= 20.0 &&
                       Math.Sqrt(VelocityU * VelocityU + VelocityV * VelocityV + VelocityW * VelocityW) <= 5.0;
            }
        }

        /// <summary>
        /// 应用物理约束
        /// </summary>
        public void ApplyPhysicalConstraints()
        {
            Temperature = Math.Max(-2.0, Math.Min(35.0, Temperature));
            Salinity = Math.Max(0.0, Math.Min(45.0, Salinity));
            SeaIceConcentration = Math.Max(0.0, Math.Min(1.0, SeaIceConcentration));
            SeaIceThickness = Math.Max(0.0, Math.Min(20.0, SeaIceThickness));

            // 流速约束
            double velocityMagnitude = Math.Sqrt(VelocityU * VelocityU + VelocityV * VelocityV + VelocityW * VelocityW);
            if (velocityMagnitude > 5.0)
            {
                double scale = 5.0 / velocityMagnitude;
                VelocityU *= scale;
                VelocityV *= scale;
                VelocityW *= scale;
            }

            // 海冰物理一致性
            if (SeaIceConcentration < 0.01)
            {
                SeaIceThickness = 0.0;
            }
        }

        public override string ToString()
        {
            return $"T={Temperature:F2}°C, S={Salinity:F2}PSU, V=({VelocityU:F3},{VelocityV:F3},{VelocityW:F3})m/s, " +
                   $"Ice={SeaIceConcentration:F3}({SeaIceThickness:F2}m), SSH={SeaSurfaceHeight:F3}m";
        }
    }

    /// <summary>
    /// EnKF配置参数
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct EnKFConfig
    {
        public int EnsembleSize;              // 集合大小（默认100）
        public double LocalizationRadius;     // 局地化半径 (m)
        public double InflationFactor;        // 充气因子
        public double RegularizationThreshold; // 正则化阈值
        [MarshalAs(UnmanagedType.U1)]
        public bool UseLocalization;          // 启用局地化
        [MarshalAs(UnmanagedType.U1)]
        public bool UseInflation;             // 启用充气
        public int NumThreads;                // 线程数
        [MarshalAs(UnmanagedType.U1)]
        public bool EnableVectorization;      // 启用向量化

        /// <summary>
        /// 获取TOPAZ标准配置
        /// </summary>
        public static EnKFConfig DefaultTOPAZ => new EnKFConfig
        {
            EnsembleSize = 100,              // TOPAZ标准配置
            LocalizationRadius = 150000.0,   // 150km
            InflationFactor = 1.02,          // 2%充气
            RegularizationThreshold = 1e-10,
            UseLocalization = true,
            UseInflation = true,
            NumThreads = Environment.ProcessorCount,
            EnableVectorization = true
        };

        /// <summary>
        /// 验证配置有效性
        /// </summary>
        public bool IsValid
        {
            get
            {
                return EnsembleSize > 0 && EnsembleSize <= 1000 &&
                       LocalizationRadius > 0.0 && LocalizationRadius <= 1000000.0 &&
                       InflationFactor >= 1.0 && InflationFactor <= 3.0 &&
                       RegularizationThreshold > 0.0 && RegularizationThreshold <= 1e-6 &&
                       NumThreads > 0 && NumThreads <= 64;
            }
        }
    }

    /// <summary>
    /// 观测数据结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ObservationData
    {
        public IntPtr Values;                 // 观测值数组
        public IntPtr ErrorVariances;        // 观测误差方差数组
        public IntPtr LocationsX;            // 观测位置X坐标
        public IntPtr LocationsY;            // 观测位置Y坐标
        public IntPtr ObservationTypes;      // 观测类型数组
        public IntPtr QualityFlags;          // 质量控制标记
        public int NumObservations;           // 观测数量
        public long Timestamp;                // 观测时间戳
    }

    /// <summary>
    /// 预报结果结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ForecastResult
    {
        public IntPtr EnsembleMean;           // 集合均值
        public IntPtr ForecastCovariance;    // 预报协方差（展平的矩阵）
        public double EnsembleSpread;         // 集合离散度
        public long ComputationTimeMs;       // 计算时间（毫秒）
        [MarshalAs(UnmanagedType.U1)]
        public bool Success;                  // 成功标志
        public int StateSize;                 // 状态大小
    }

    /// <summary>
    /// 分析结果结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct AnalysisResult
    {
        public IntPtr AnalysisMean;           // 分析均值
        public IntPtr AnalysisCovariance;    // 分析协方差（展平的矩阵）
        public IntPtr KalmanGain;            // 卡尔曼增益矩阵（展平）
        public double InnovationVariance;     // 创新方差
        public double ObservationImpact;      // 观测影响
        public long ComputationTimeMs;       // 计算时间（毫秒）
        [MarshalAs(UnmanagedType.U1)]
        public bool Success;                  // 成功标志
        public int StateSize;                 // 状态大小
        public int ObsSize;                   // 观测大小
    }

    /// <summary>
    /// 性能指标结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct PerformanceMetrics
    {
        public double EnsembleSpread;         // 集合离散度
        public double FilterDivergence;       // 滤波器发散度
        public double ObservationImpact;      // 观测影响
        public ulong TotalCycles;             // 总循环次数
        public long AverageCycleTimeMs;       // 平均循环时间
        public double MemoryUsageMb;          // 内存使用量（MB）
        public double AnalysisAccuracy;       // 分析精度
    }

    /// <summary>
    /// 网格参数结构
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct GridParameters
    {
        public int Nx, Ny, Nz;               // 网格维度
        public double Dx, Dy, Dz;            // 网格间距
        public double XMin, YMin, ZMin;      // 网格起始坐标
        public double XMax, YMax, ZMax;      // 网格结束坐标

        /// <summary>
        /// 创建北大西洋和北极海域网格（基于TOPAZ系统）
        /// </summary>
        public static GridParameters CreateTOPAZGrid()
        {
            return new GridParameters
            {
                Nx = 800, Ny = 880, Nz = 32,           // TOPAZ3典型分辨率
                Dx = 11000.0, Dy = 11000.0, Dz = 10.0, // 北极11km，垂直10m
                XMin = -2500000.0, YMin = -2500000.0, ZMin = 0.0,
                XMax = 2500000.0, YMax = 2500000.0, ZMax = 5000.0
            };
        }

        public long TotalGridPoints => (long)Nx * Ny * Nz;
        public double GridVolume => (XMax - XMin) * (YMax - YMin) * (ZMax - ZMin);
    }

    /// <summary>
    /// 观测类型枚举
    /// </summary>
    public enum ObservationType
    {
        SeaLevelAnomaly = 0,      // 海平面高度异常（DUACS）
        SeaSurfaceTemperature,    // 海表温度（Reynolds）
        SeaIceConcentration,      // 海冰浓度（SSM/I）
        ArgoTemperature,          // Argo温度剖面
        ArgoSalinity,             // Argo盐度剖面
        SeaIceDrift,              // 海冰漂移（CERSAT）
        Altimetry                 // 高度计数据
    }

    /// <summary>
    /// 充气类型枚举
    /// </summary>
    public enum InflationType
    {
        Multiplicative = 0,       // 乘性充气
        Additive,                 // 加性充气
        Adaptive,                 // 自适应充气
        Relaxation                // 松弛充气
    }

    /// <summary>
    /// EnKF预测模块的P/Invoke接口声明
    /// </summary>
    public static class EnKFNative
    {
        private const string DllName = "oceansim_csharp";

        // ==============================================================================
        // EnKF系统管理接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr EnKF_Create(ref EnKFConfig config, ref GridParameters gridParams);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_Initialize(IntPtr handle,
                                                  [In] OceanState[] initialState,
                                                  [In] double[] backgroundCovariance,
                                                  int stateSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetSystemInfo(IntPtr handle,
                                                    [Out] StringBuilder infoBuffer,
                                                    int bufferSize);

        // ==============================================================================
        // EnKF核心算法接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_ExecuteForecast(IntPtr handle,
                                                      double timeStep,
                                                      out ForecastResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_ExecuteAnalysis(IntPtr handle,
                                                      ref ObservationData observations,
                                                      out AnalysisResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_ExecuteAssimilationCycle(IntPtr handle,
                                                               double timeStep,
                                                               ref ObservationData observations,
                                                               out ForecastResult forecastResult,
                                                               out AnalysisResult analysisResult);

        // ==============================================================================
        // 状态访问接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetCurrentMean(IntPtr handle,
                                                     [Out] OceanState[] meanState,
                                                     int stateSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetCurrentCovariance(IntPtr handle,
                                                           [Out] double[] covariance,
                                                           int matrixSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetEnsembleMember(IntPtr handle,
                                                        int memberIndex,
                                                        [Out] OceanState[] memberState,
                                                        int stateSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetEnsembleStatistics(IntPtr handle,
                                                            out double ensembleSpread,
                                                            out double ensembleMeanNorm);

        // ==============================================================================
        // 观测算子接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr ObsOp_Create(ObservationType obsType, IntPtr gridHandle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void ObsOp_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int ObsOp_Apply(IntPtr handle,
                                             [In] OceanState[] modelState,
                                             int stateSize,
                                             [Out] double[] observations,
                                             int obsSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool ObsOp_Validate(IntPtr handle,
                                                 ref ObservationData observations);

        // ==============================================================================
        // 局地化和充气接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_SetLocalization(IntPtr handle,
                                                      double radius,
                                                      bool enable);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_SetInflation(IntPtr handle,
                                                   InflationType inflationType,
                                                   double inflationFactor);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_ComputeAdaptiveInflation(IntPtr handle,
                                                               ref ObservationData observations,
                                                               out double adaptiveFactor);

        // ==============================================================================
        // 性能监控接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetPerformanceMetrics(IntPtr handle,
                                                            out PerformanceMetrics metrics);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_ResetPerformanceStats(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_EnableProfiling(IntPtr handle, bool enable);

        // ==============================================================================
        // 验证和诊断接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_ValidateSystemState(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_CheckFilterDivergence(IntPtr handle,
                                                             double divergenceThreshold);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_ComputeAnalysisIncrement(IntPtr handle,
                                                               out double incrementNorm);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_ValidateLinearGaussian(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_ValidateLorenz96(IntPtr handle);

        // ==============================================================================
        // 内存管理接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_FreeForecastResult(ref ForecastResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_FreeAnalysisResult(ref AnalysisResult result);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_OptimizeMemory(IntPtr handle);

        // ==============================================================================
        // 配置和参数接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_GetDefaultConfig(out EnKFConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_UpdateConfig(IntPtr handle, ref EnKFConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_SaveState(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_LoadState(IntPtr handle, [MarshalAs(UnmanagedType.LPStr)] string filename);

        // ==============================================================================
        // 错误处理接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EnKF_GetLastError(IntPtr handle,
                                                   [Out] StringBuilder errorBuffer,
                                                   int bufferSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_ClearError(IntPtr handle);

        // ==============================================================================
        // 工具函数接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr EnKF_CreateObservationData(int numObservations);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_FreeObservationData(IntPtr obsData);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_SetObservation(IntPtr obsData,
                                                      int index,
                                                      double value,
                                                      double errorVariance,
                                                      double xLocation,
                                                      double yLocation,
                                                      ObservationType obsType);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr EnKF_CreateStateArray(int stateSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_FreeStateArray(IntPtr stateArray);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_CopyState(ref OceanState dest, ref OceanState src);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_InitializeStateZero(ref OceanState state);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_ApplyPhysicalConstraints(ref OceanState state);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_IsStatePhysicallyValid(ref OceanState state);

        // ==============================================================================
        // 版本和信息接口
        // ==============================================================================

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_GetVersion([Out] StringBuilder versionBuffer, int bufferSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void EnKF_GetBuildInfo([Out] StringBuilder buildInfoBuffer, int bufferSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool EnKF_CheckHardwareSupport([MarshalAs(UnmanagedType.LPStr)] string featureName);
    }
}
