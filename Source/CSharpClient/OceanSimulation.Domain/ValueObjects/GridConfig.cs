namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 网格配置参数
    /// </summary>
    public struct GridConfig
    {
        public int Nx { get; set; }
        public int Ny { get; set; }
        public int Nz { get; set; }
        public double Dx { get; set; }
        public double Dy { get; set; }
        public double Dz { get; set; }
        public Vector3D Origin { get; set; }
        public CoordinateSystemType CoordinateSystem { get; set; }
        public GridTypeEnum GridType { get; set; }
    }
}
