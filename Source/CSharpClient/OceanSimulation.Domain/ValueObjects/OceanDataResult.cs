using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 海洋数据处理结果
    /// </summary>
    public class OceanDataResult
    {
        public double[,] U { get; set; } = new double[0, 0];
        public double[,] V { get; set; } = new double[0, 0];
        public double[] Latitude { get; set; } = Array.Empty<double>();
        public double[] Longitude { get; set; } = Array.Empty<double>();
        public bool Success { get; set; }
        public string Message { get; set; } = string.Empty;
        public Dictionary<string, object> Metadata { get; set; } = new();
    }
}
