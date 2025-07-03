using System.Collections.Generic;

namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 可视化结果
    /// </summary>
    public class VisualizationResult
    {
        public string ImagePath { get; set; }
        public bool Success { get; set; }
        public string Message { get; set; }
        public Dictionary<string, object> Metadata { get; set; }
    }
}
