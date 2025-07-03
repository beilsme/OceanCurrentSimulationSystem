namespace OceanSimulation.Domain.ValueObjects
{
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
    /// 执行策略
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

    /// <summary>
    /// 日志级别
    /// </summary>
    public enum LogLevel
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }
}
