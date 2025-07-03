namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 二维矢量场数据
    /// </summary>
    public class VectorFieldData
    {
        public double[,] U { get; set; }
        public double[,] V { get; set; }
        public double[] Latitude { get; set; }
        public double[] Longitude { get; set; }
        public double? Depth { get; set; }
        public string TimeInfo { get; set; }
    }
}
