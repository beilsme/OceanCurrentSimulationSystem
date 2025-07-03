namespace OceanSimulation.Domain.ValueObjects;

/// <summary>
/// 海洋学统计结果
/// </summary>
public class OceanographicStatistics
{
    public FlowStatistics? FlowStatistics { get; set; }
    public VorticityStatistics? VorticityStatistics { get; set; }
    public DivergenceStatistics? DivergenceStatistics { get; set; }
}

public class FlowStatistics
{
    public double MeanSpeed { get; set; }
    public double MaxSpeed { get; set; }
    public double SpeedStandardDeviation { get; set; }
    public double DominantDirection { get; set; }
    public double KineticEnergyDensity { get; set; }
}

public class VorticityStatistics
{
    public double MeanVorticity { get; set; }
    public double MaxVorticity { get; set; }
    public double MinVorticity { get; set; }
    public double VorticityVariance { get; set; }
    public int CycloneCount { get; set; }
    public int AnticycloneCount { get; set; }
}

public class DivergenceStatistics
{
    public double MeanDivergence { get; set; }
    public double MaxDivergence { get; set; }
    public double MinDivergence { get; set; }
    public int ConvergenceZones { get; set; }
    public int DivergenceZones { get; set; }
}
