namespace OceanSimulation.Domain.Entities
{
    /// <summary>
    /// 性能指标
    /// </summary>
    public class PerformanceMetrics
    {
        public double ComputationTime { get; set; }
        public double MemoryUsageMb { get; set; }
        public int IterationCount { get; set; }
        public double EfficiencyRatio { get; set; }
    }
}
