using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 3D可视化参数
    /// </summary>
    public class Visualization3DParameters
    {
        public string Title { get; set; } = "3D海洋数据可视化";
        public Dictionary<string, int> SlicePositions { get; set; } = new();
    }
}
