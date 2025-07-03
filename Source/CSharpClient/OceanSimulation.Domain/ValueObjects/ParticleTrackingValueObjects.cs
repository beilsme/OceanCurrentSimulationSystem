namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 粒子追踪配置参数
    /// </summary>
    public class ParticleTrackingConfig
    {
        public int ParticleCount { get; set; } = 50;
        public int TimeSteps { get; set; } = 100;
        public double TimeStepSize { get; set; } = 3600.0;
        public int TrailLength { get; set; } = 10;
        public string InitialPositionType { get; set; } = "random";
        public string? OutputPath { get; set; }
        public Dictionary<string, object> Bounds { get; set; } = new();
    }

    /// <summary>
    /// 粒子追踪结果
    /// </summary>
    public class ParticleTrackingResult
    {
        public bool Success { get; set; }
        public string OutputPath { get; set; } = "";
        public string ErrorMessage { get; set; } = "";
        public ParticleTrackingStatistics? Statistics { get; set; }
        public ParticleTrackingMetadata? Metadata { get; set; }
    }

    /// <summary>
    /// 粒子追踪统计信息
    /// </summary>
    public class ParticleTrackingStatistics
    {
        public DisplacementStatistics? DisplacementStatistics { get; set; }
        public TrajectoryStatistics? TrajectoryStatistics { get; set; }
        public ParticleStatus? ParticleStatus { get; set; }
    }

    /// <summary>
    /// 位移统计数据
    /// </summary>
    public class DisplacementStatistics
    {
        public double MeanDisplacement { get; set; }
        public double MaxDisplacement { get; set; }
        public double StdDisplacement { get; set; }
    }

    /// <summary>
    /// 轨迹统计数据
    /// </summary>
    public class TrajectoryStatistics
    {
        public double MeanTrajectoryLength { get; set; }
        public double MaxTrajectoryLength { get; set; }
    }

    /// <summary>
    /// 粒子状态统计
    /// </summary>
    public class ParticleStatus
    {
        public int TotalParticles { get; set; }
        public int ActiveParticles { get; set; }
        public int InactiveParticles { get; set; }
    }

    /// <summary>
    /// 粒子追踪元数据
    /// </summary>
    public class ParticleTrackingMetadata
    {
        public int ParticleCount { get; set; }
        public int TimeSteps { get; set; }
        public double SimulationDurationHours { get; set; }
        public int TrailLength { get; set; }
    }
}
