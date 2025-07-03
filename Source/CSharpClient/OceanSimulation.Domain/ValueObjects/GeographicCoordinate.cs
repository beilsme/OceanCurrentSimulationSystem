namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 地理坐标
    /// </summary>
    public struct GeographicCoordinate
    {
        public double Latitude { get; set; }
        public double Longitude { get; set; }
        public double Depth { get; set; }
    }
}
