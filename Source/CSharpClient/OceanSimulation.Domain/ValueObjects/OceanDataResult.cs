using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 海洋数据处理结果
    /// </summary>
    public class OceanDataResult
    {
        public double[,] U { get; set; }
        public double[,] V { get; set; }
        public double[] Latitude { get; set; }
        public double[] Longitude { get; set; }
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> Metadata { get; set; }
    }
}
