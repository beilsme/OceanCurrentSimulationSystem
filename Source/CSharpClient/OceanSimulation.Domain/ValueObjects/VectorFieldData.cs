namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 二维矢量场数据
    /// </summary>
    public class VectorFieldData
    {
        public double[,] U { get; set; } = new double[0, 0];
        public double[,] V { get; set; } = new double[0, 0];
        public double[] Latitude { get; set; } = Array.Empty<double>();
        public double[] Longitude { get; set; } = Array.Empty<double>();
        public double? Depth { get; set; }
        public string TimeInfo { get; set; } = string.Empty;
    }
}
