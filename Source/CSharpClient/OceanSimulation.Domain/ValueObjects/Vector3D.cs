namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 双精度三维向量
    /// </summary>
    public struct Vector3D
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }

        public Vector3D(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }
    }
}
