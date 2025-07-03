namespace OceanSimulation.Domain.ValueObjects;

/// <summary>
/// Python执行结果
/// </summary>
public class PythonResult
{
    public bool Success { get; set; }
    public string Output { get; set; } = "";
    public string Error { get; set; } = "";
}
