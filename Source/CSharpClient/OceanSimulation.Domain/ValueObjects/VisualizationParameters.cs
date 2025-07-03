namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 可视化参数
    /// </summary>
    public class VisualizationParameters
    {
        public int Skip { get; set; } = 3;
        public double? LonMin { get; set; }
        public double? LonMax { get; set; }
        public double? LatMin { get; set; }
        public double? LatMax { get; set; }
        public int FontSize { get; set; } = 14;
        public int DPI { get; set; } = 120;
    }
}
