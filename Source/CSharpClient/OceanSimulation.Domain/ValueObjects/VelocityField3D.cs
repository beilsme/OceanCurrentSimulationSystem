namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 三维速度场数据
    /// </summary>
    public class VelocityField3D
    {
        public double[,,] U { get; set; } = new double[0, 0, 0];
        public double[,,] V { get; set; } = new double[0, 0, 0];
        public double[,,] W { get; set; } = new double[0, 0, 0];
        public int NX { get; set; }
        public int NY { get; set; }
        public int NZ { get; set; }
        public double DX { get; set; }
        public double DY { get; set; }
        public double DZ { get; set; }
    }
}
