using System.Collections.Generic;
using System.Threading.Tasks;
using OceanSimulation.Domain.ValueObjects;
using OceanSimulation.Domain.Entities;

namespace OceanSimulation.Domain.Interfaces
{
    /// <summary>
    /// Python 机器学习/可视化引擎接口
    /// </summary>
    public interface IPythonMLEngine
    {
        Task<bool> InitializeAsync();
        Task<string> GenerateVectorFieldVisualizationAsync(VectorFieldData vectorField, VisualizationParameters parameters);
        Task<string> Generate3DVelocityVisualizationAsync(VelocityField3D velocityField, Visualization3DParameters parameters);
        Task<string> GenerateComprehensiveDashboardAsync(DashboardData dashboardData, DashboardParameters parameters);
        Task<OceanDataResult> ProcessNetCDFDataAsync(string filePath, DataProcessingParameters parameters);
        Task<Dictionary<string, object>> GetPerformanceMetricsAsync();
    }
}
