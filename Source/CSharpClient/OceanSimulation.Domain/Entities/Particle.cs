using OceanSimulation.Domain.ValueObjects;

namespace OceanSimulation.Domain.Entities
{
    /// <summary>
    /// 粒子实体
    /// </summary>
    public class Particle
    {
        public int Id { get; set; }
        public Vector3D Position { get; set; }
        public Vector3D Velocity { get; set; }
        public double Age { get; set; }
        public bool Active { get; set; }
    }
}
