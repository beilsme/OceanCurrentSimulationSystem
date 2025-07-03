namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 物理参数
    /// </summary>
    public struct PhysicalParameters
    {
        public double Density { get; set; }
        public double Viscosity { get; set; }
        public double Gravity { get; set; }
        public double CoriolisParam { get; set; }
        public double WindStressX { get; set; }
        public double WindStressY { get; set; }
    }
}
