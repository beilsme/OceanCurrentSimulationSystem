using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 综合仪表板数据
    /// </summary>
    public class DashboardData
    {
        public object VelocityData { get; set; }
        public object ConcentrationData { get; set; }
        public double[,] ParticlePositions { get; set; }
        public Dictionary<string, object> TimeInfo { get; set; }
        public Dictionary<string, object> Statistics { get; set; }
    }
}
