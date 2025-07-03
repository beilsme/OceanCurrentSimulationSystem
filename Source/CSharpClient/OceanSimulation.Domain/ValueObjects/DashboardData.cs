using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 综合仪表板数据
    /// </summary>
    public class DashboardData
    {
        public object VelocityData { get; set; } = new object();
        public object ConcentrationData { get; set; } = new object();
        public double[,] ParticlePositions { get; set; } = new double[0, 0];
        public Dictionary<string, object> TimeInfo { get; set; } = new();
        public Dictionary<string, object> Statistics { get; set; } = new();
    }
}
