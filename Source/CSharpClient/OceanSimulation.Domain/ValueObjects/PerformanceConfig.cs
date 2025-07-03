namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 性能配置
    /// </summary>
    public struct PerformanceConfig
    {
        public int ExecutionPolicy { get; set; }
        public int NumThreads { get; set; }
        public int SimdType { get; set; }
        public int Priority { get; set; }
    }
}
