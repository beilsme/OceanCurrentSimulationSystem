namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 单精度三维向量
    /// </summary>
    public struct Vector3F
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public Vector3F(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }
    }
}
