namespace OceanSimulation.Domain.ValueObjects
{
    /// <summary>
    /// 数值求解器参数
    /// </summary>
    public struct SolverParameters
    {
        public double TimeStep { get; set; }
        public double DiffusionCoeff { get; set; }
        public NumericalSchemeType SchemeType { get; set; }
        public TimeIntegrationType IntegrationMethod { get; set; }
        public double CflNumber { get; set; }
    }
}
