namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 数据处理参数
    /// </summary>
    public class DataProcessingParameters
    {
        public int TimeIndex { get; set; } = 0;
        public int DepthIndex { get; set; } = 0;
        public double? LonMin { get; set; }
        public double? LonMax { get; set; }
        public double? LatMin { get; set; }
        public double? LatMax { get; set; }
    }
}
