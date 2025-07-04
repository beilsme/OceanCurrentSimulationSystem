namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 污染物扩散模拟结果
    /// </summary>
    public class PollutionDispersionResult
    {
        public bool Success { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public string ErrorMessage { get; set; } = string.Empty;
        public double MaxConcentration { get; set; }
        public double MeanConcentration { get; set; }
        public double TotalMass { get; set; }
    }
}
