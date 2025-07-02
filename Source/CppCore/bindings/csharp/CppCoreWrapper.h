// bindings/csharp/CppCoreWrapper.h
#pragma once
#include <cstddef>

#ifdef _WIN32
#ifdef OCEANSIM_CSHARP_EXPORTS
        #define OCEANSIM_API __declspec(dllexport)
    #else
        #define OCEANSIM_API __declspec(dllimport)
    #endif
#else
#define OCEANSIM_API __attribute__((visibility("default")))
#endif

extern "C" {

// ===========================================
// 基础数据类型定义
// ===========================================

typedef void* GridHandle;
typedef void* GridDataHandle;
typedef void* ParticleSimulatorHandle;
typedef void* CurrentFieldSolverHandle;
typedef void* AdvectionDiffusionSolverHandle;
typedef void* AdvectionDiffusionHandle;
typedef void* RungeKuttaSolverHandle;
typedef void* FiniteDifferenceSolverHandle;
typedef void* ParallelComputeEngineHandle;
typedef void* VectorizedOperationsHandle;
typedef void* MemoryManagerHandle;
typedef void* DataExporterHandle;
typedef void* PerformanceProfilerHandle;

// 坐标和向量数据结构
typedef struct {
    double x, y, z;
} Vector3D;

typedef struct {
    float x, y, z;
} Vector3F;

typedef struct {
    double latitude;
    double longitude;
    double depth;
} GeographicCoordinate;

// 粒子数据结构
typedef struct {
    int id;
    Vector3D position;
    Vector3D velocity;
    double age;
    int active;
} ParticleData;

// 网格配置结构
typedef struct {
    int nx, ny, nz;
    double dx, dy, dz;
    Vector3D origin;
    int coordinate_system;
    int grid_type;
} GridConfigType;

// 物理参数结构
typedef struct {
    double density;
    double viscosity;
    double gravity;
    double coriolis_param;
    double wind_stress_x;
    double wind_stress_y;
    // 洋流求解器专用参数
    double coriolis_f;
    double beta;
    double viscosity_h;
    double viscosity_v;
    double diffusivity_h;
    double diffusivity_v;
    double reference_density;
} PhysicalParameters;

// 求解器参数结构
typedef struct {
    int scheme_type;
    int integration_method;
    double diffusion_coeff;
} SolverParameters;

// 性能配置结构
typedef struct {
    int num_threads;
    int execution_policy;
    int simd_type;
} PerformanceConfig;

// 枚举类型定义
typedef enum {
    COORDINATE_CARTESIAN = 0,
    COORDINATE_SPHERICAL = 1,
    COORDINATE_HYBRID_SIGMA = 2,
    COORDINATE_ISOPYCNAL = 3
} CoordinateSystemType;

typedef enum {
    GRID_REGULAR = 0,
    GRID_CURVILINEAR = 1,
    GRID_UNSTRUCTURED = 2
} GridTypeEnum;

typedef enum {
    INTERP_LINEAR = 0,
    INTERP_CUBIC = 1,
    INTERP_BILINEAR = 2,
    INTERP_TRILINEAR = 3,
    INTERP_CONSERVATIVE = 4
} InterpolationMethodType;

typedef enum {
    SCHEME_UPWIND = 0,
    SCHEME_CENTRAL = 1,
    SCHEME_TVD_SUPERBEE = 2,
    SCHEME_WENO = 3
} NumericalSchemeType;

typedef enum {
    TIME_EULER = 0,
    TIME_RUNGE_KUTTA_2 = 1,
    TIME_RUNGE_KUTTA_3 = 2,
    TIME_RUNGE_KUTTA_4 = 3,
    TIME_ADAMS_BASHFORTH = 4
} TimeIntegrationType;

typedef enum {
    EXECUTION_SEQUENTIAL = 0,
    EXECUTION_PARALLEL = 1,
    EXECUTION_VECTORIZED = 2,
    EXECUTION_HYBRID_PARALLEL = 3
} ExecutionPolicyType;

typedef enum {
    SIMD_NONE = 0,
    SIMD_SSE = 1,
    SIMD_AVX = 2,
    SIMD_AVX2 = 3,
    SIMD_AVX512 = 4,
    SIMD_NEON = 5
} SimdTypeEnum;

typedef enum {
    BC_DIRICHLET = 0,
    BC_NEUMANN = 1,
    BC_ROBIN = 2,
    BC_PERIODIC = 3,
    BC_OUTFLOW = 4
} BoundaryConditionType;

// ========================= 粒子模拟器接口 =========================

OCEANSIM_API ParticleSimulatorHandle CreateParticleSimulator(
        int nx, int ny, int nz, double dx, double dy, double dz);

OCEANSIM_API void DestroyParticleSimulator(ParticleSimulatorHandle handle);

OCEANSIM_API void InitializeParticles(ParticleSimulatorHandle handle,
                                      const Vector3D* positions, int count);

OCEANSIM_API void InitializeRandomParticles(ParticleSimulatorHandle handle,
                                            int count, const Vector3D* bounds_min,
                                            const Vector3D* bounds_max);

OCEANSIM_API void StepParticlesForward(ParticleSimulatorHandle handle, double dt);

OCEANSIM_API int GetParticleCount(ParticleSimulatorHandle handle);

OCEANSIM_API void GetParticles(ParticleSimulatorHandle handle,
                               ParticleData* particles, int max_count);

OCEANSIM_API void SetParticleVelocityField(ParticleSimulatorHandle handle,
                                           const char* field_name);

OCEANSIM_API void EnableParticleDiffusion(ParticleSimulatorHandle handle,
                                          double diffusion_coefficient);

OCEANSIM_API void SetParticleBoundaryConditions(ParticleSimulatorHandle handle,
                                                const Vector3D* bounds_min,
                                                const Vector3D* bounds_max,
                                                int periodic);

OCEANSIM_API double GetParticleComputationTime(ParticleSimulatorHandle handle);

// ========================= 洋流场求解器接口 =========================

OCEANSIM_API CurrentFieldSolverHandle CreateCurrentFieldSolver(
        int nx, int ny, int nz, double dx, double dy, double dz,
        const PhysicalParameters* params);

OCEANSIM_API void DestroyCurrentFieldSolver(CurrentFieldSolverHandle handle);

OCEANSIM_API void SetCurrentFieldInitialCondition(CurrentFieldSolverHandle handle,
                                                  const char* field_name,
                                                  const double* data, int size);

OCEANSIM_API void SetBottomTopography(CurrentFieldSolverHandle handle,
                                      const double* bottom_depth, int nx, int ny);

OCEANSIM_API void SetWindStress(CurrentFieldSolverHandle handle,
                                const double* tau_x, const double* tau_y,
                                int nx, int ny);

OCEANSIM_API void StepCurrentFieldForward(CurrentFieldSolverHandle handle, double dt);

OCEANSIM_API void GetCurrentField(CurrentFieldSolverHandle handle,
                                  const char* field_name,
                                  double* data, int max_size);

OCEANSIM_API void GetVelocityField(CurrentFieldSolverHandle handle,
                                   double* u_data, double* v_data, double* w_data,
                                   int max_size);

OCEANSIM_API double GetTotalEnergy(CurrentFieldSolverHandle handle);

OCEANSIM_API int CheckMassConservation(CurrentFieldSolverHandle handle, double tolerance);

// ========================= 网格数据结构接口 =========================

OCEANSIM_API GridDataHandle CreateGridData(int nx, int ny, int nz,
                                           CoordinateSystemType coord_sys,
                                           GridTypeEnum grid_type);

OCEANSIM_API void DestroyGridData(GridDataHandle handle);

OCEANSIM_API void SetGridSpacing(GridDataHandle handle,
                                 double dx, double dy, const double* dz, int nz);

OCEANSIM_API void SetGridOrigin(GridDataHandle handle, const Vector3D* origin);

OCEANSIM_API void AddScalarField2D(GridDataHandle handle, const char* name,
                                   const double* data, int nx, int ny);

OCEANSIM_API void AddScalarField3D(GridDataHandle handle, const char* name,
                                   const double* data, int nx, int ny, int nz);

OCEANSIM_API void AddVectorField(GridDataHandle handle, const char* name,
                                 const double* u_data, const double* v_data,
                                 const double* w_data, int nx, int ny, int nz);

OCEANSIM_API double InterpolateScalar(GridDataHandle handle, const char* field_name,
                                      const Vector3D* position, InterpolationMethodType method);

OCEANSIM_API Vector3D InterpolateVector(GridDataHandle handle, const char* field_name,
                                        const Vector3D* position, InterpolationMethodType method);

OCEANSIM_API Vector3D ComputeGradient(GridDataHandle handle, const char* field_name,
                                      const Vector3D* position);

OCEANSIM_API void GetFieldData2D(GridDataHandle handle, const char* field_name,
                                 double* data, int max_size);

OCEANSIM_API void GetFieldData3D(GridDataHandle handle, const char* field_name,
                                 double* data, int max_size);

OCEANSIM_API int HasField(GridDataHandle handle, const char* field_name);

OCEANSIM_API void ClearField(GridDataHandle handle, const char* field_name);

OCEANSIM_API size_t GetGridMemoryUsage(GridDataHandle handle);

// ========================= 平流扩散求解器接口 =========================

OCEANSIM_API AdvectionDiffusionHandle CreateAdvectionDiffusionSolver(
        GridDataHandle grid_handle, NumericalSchemeType scheme, TimeIntegrationType time_method);

OCEANSIM_API void DestroyAdvectionDiffusionSolver(AdvectionDiffusionHandle handle);

OCEANSIM_API void SetADInitialCondition(AdvectionDiffusionHandle handle,
                                        const char* field_name,
                                        const double* initial_data, int size);

OCEANSIM_API void SetADVelocityField(AdvectionDiffusionHandle handle,
                                     const char* u_field, const char* v_field,
                                     const char* w_field);

OCEANSIM_API void SetDiffusionCoefficient(AdvectionDiffusionHandle handle,
                                          double diffusion_coeff);

OCEANSIM_API void SetBoundaryCondition(AdvectionDiffusionHandle handle,
                                       BoundaryConditionType type, int boundary_id,
                                       double value);

OCEANSIM_API void SolveAdvectionDiffusion(AdvectionDiffusionHandle handle, double dt);

OCEANSIM_API void GetADSolution(AdvectionDiffusionHandle handle,
                                double* solution_data, int max_size);

OCEANSIM_API double GetMaxConcentration(AdvectionDiffusionHandle handle);

OCEANSIM_API double GetTotalMass(AdvectionDiffusionHandle handle);

OCEANSIM_API int CheckADMassConservation(AdvectionDiffusionHandle handle, double tolerance);

OCEANSIM_API double ComputePecletNumber(AdvectionDiffusionHandle handle);

OCEANSIM_API double ComputeCourantNumber(AdvectionDiffusionHandle handle, double dt);

// ========================= 龙格-库塔求解器接口 =========================

OCEANSIM_API RungeKuttaSolverHandle CreateRungeKuttaSolver(int method);

OCEANSIM_API void DestroyRungeKuttaSolver(RungeKuttaSolverHandle handle);

// ========================= 有限差分求解器接口 =========================

OCEANSIM_API FiniteDifferenceSolverHandle CreateFiniteDifferenceSolver(int grid_size, double time_step);

OCEANSIM_API void DestroyFiniteDifferenceSolver(FiniteDifferenceSolverHandle handle);

OCEANSIM_API void SetFDBoundaryConditions(FiniteDifferenceSolverHandle handle,
                                          BoundaryConditionType type, const double* values, int size);

OCEANSIM_API void SolveFDAdvectionDiffusion(FiniteDifferenceSolverHandle handle,
                                            const double* initial_data, double* result,
                                            int size, double dt, int num_steps);

// ========================= 向量化运算接口 =========================

OCEANSIM_API VectorizedOperationsHandle CreateVectorizedOperations(const PerformanceConfig* config);

OCEANSIM_API void DestroyVectorizedOperations(VectorizedOperationsHandle handle);

OCEANSIM_API void VectorAdd(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size);

OCEANSIM_API void VectorSub(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size);

OCEANSIM_API void VectorMul(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size);

OCEANSIM_API double VectorDotProduct(VectorizedOperationsHandle handle, const double* a, const double* b, int size);

// ========================= 并行计算引擎接口 =========================

OCEANSIM_API ParallelComputeEngineHandle CreateParallelComputeEngine(const PerformanceConfig* config);

OCEANSIM_API void DestroyParallelComputeEngine(ParallelComputeEngineHandle handle);

OCEANSIM_API void StartParallelEngine(ParallelComputeEngineHandle handle);

OCEANSIM_API void StopParallelEngine(ParallelComputeEngineHandle handle);

OCEANSIM_API int GetAvailableThreads(ParallelComputeEngineHandle handle);

// ========================= 数据导出器接口 =========================

OCEANSIM_API DataExporterHandle CreateDataExporter(void);

OCEANSIM_API void DestroyDataExporter(DataExporterHandle handle);

OCEANSIM_API int ExportToNetCDF(DataExporterHandle handle, GridDataHandle grid, const char* filename);

OCEANSIM_API int ExportToVTK(DataExporterHandle handle, GridDataHandle grid, const char* filename);

// ========================= 性能分析器接口 =========================

OCEANSIM_API PerformanceProfilerHandle CreatePerformanceProfiler(void);

OCEANSIM_API void DestroyPerformanceProfiler(PerformanceProfilerHandle handle);

OCEANSIM_API void StartTiming(PerformanceProfilerHandle handle, const char* section_name);

OCEANSIM_API void EndTiming(PerformanceProfilerHandle handle, const char* section_name);

OCEANSIM_API double GetElapsedTime(PerformanceProfilerHandle handle, const char* section_name);

OCEANSIM_API void GenerateReport(PerformanceProfilerHandle handle, const char* filename);

// ========================= 性能分析接口 =========================

typedef struct {
    double computation_time;
    double memory_usage_mb;
    int iteration_count;
    double efficiency_ratio;
} PerformanceMetrics;

OCEANSIM_API void GetPerformanceMetrics(void* handle, PerformanceMetrics* metrics);

OCEANSIM_API void EnableProfiling(void* handle, int enable);

OCEANSIM_API void ResetPerformanceCounters(void* handle);

// ========================= 错误处理接口 =========================

typedef enum {
    ERROR_NONE = 0,
    ERROR_INVALID_HANDLE = 1,
    ERROR_INVALID_PARAMETER = 2,
    ERROR_MEMORY_ALLOCATION = 3,
    ERROR_FILE_IO = 4,
    ERROR_NUMERICAL_INSTABILITY = 5,
    ERROR_CONVERGENCE_FAILURE = 6
} ErrorCode;

OCEANSIM_API ErrorCode GetLastError();

OCEANSIM_API const char* GetErrorMessage(ErrorCode error_code);

OCEANSIM_API void ClearError();

// ========================= 初始化和清理 =========================

OCEANSIM_API void InitializeCppCore();

OCEANSIM_API void ShutdownCppCore();

OCEANSIM_API const char* GetVersionString();

OCEANSIM_API int GetThreadCount();

OCEANSIM_API void SetThreadCount(int thread_count);

// ===========================================
// 原始网格数据结构接口实现（向后兼容）
// ===========================================

OCEANSIM_API GridHandle Grid_Create(const GridConfigType* config);

OCEANSIM_API void Grid_Destroy(GridHandle grid);

OCEANSIM_API void Grid_GetDimensions(GridHandle grid, int* nx, int* ny, int* nz);

OCEANSIM_API void Grid_SetScalarField(GridHandle grid, const double* data, int size, const char* field_name);

OCEANSIM_API void Grid_SetVectorField(GridHandle grid, const double* u_data, const double* v_data,
                                      const double* w_data, int size, const char* field_name);

OCEANSIM_API void Grid_GetScalarField(GridHandle grid, const char* field_name, double* data, int size);

OCEANSIM_API double Grid_Interpolate(GridHandle grid, const Vector3D* position,
                                     const char* field_name, int method);

// ===========================================
// 原始粒子模拟器接口实现（向后兼容）
// ===========================================

OCEANSIM_API ParticleSimulatorHandle ParticleSim_Create(GridHandle grid, RungeKuttaSolverHandle solver_handle);

OCEANSIM_API void ParticleSim_Destroy(ParticleSimulatorHandle simulator);

OCEANSIM_API void ParticleSim_InitializeParticles(ParticleSimulatorHandle simulator,
                                                  const Vector3D* positions, int count);

OCEANSIM_API void ParticleSim_StepForward(ParticleSimulatorHandle simulator, double time_step);

OCEANSIM_API void ParticleSim_GetParticles(ParticleSimulatorHandle simulator, ParticleData* particles, int count);

OCEANSIM_API int ParticleSim_GetParticleCount(ParticleSimulatorHandle simulator);

// ===========================================
// 原始洋流场求解器接口实现（向后兼容）
// ===========================================

OCEANSIM_API CurrentFieldSolverHandle CurrentSolver_Create(GridHandle grid, const PhysicalParameters* params);

OCEANSIM_API void CurrentSolver_Destroy(CurrentFieldSolverHandle solver);

OCEANSIM_API void CurrentSolver_ComputeVelocityField(CurrentFieldSolverHandle solver, double time_step);

OCEANSIM_API void CurrentSolver_SetBoundaryConditions(CurrentFieldSolverHandle solver, int boundary_type,
                                                      const double* values, int size);

OCEANSIM_API double CurrentSolver_ComputeKineticEnergy(CurrentFieldSolverHandle solver);

OCEANSIM_API double CurrentSolver_CheckMassConservation(CurrentFieldSolverHandle solver);

// ===========================================
// 原始平流扩散求解器接口实现（向后兼容）
// ===========================================

OCEANSIM_API AdvectionDiffusionSolverHandle AdvectionSolver_Create(GridHandle grid, const SolverParameters* params);

OCEANSIM_API void AdvectionSolver_Destroy(AdvectionDiffusionSolverHandle solver);

OCEANSIM_API void AdvectionSolver_SetInitialCondition(AdvectionDiffusionSolverHandle solver,
                                                      const double* initial_field, int size);

OCEANSIM_API void AdvectionSolver_SetVelocityField(AdvectionDiffusionSolverHandle solver,
                                                   const double* u_field, const double* v_field,
                                                   const double* w_field, int size);

OCEANSIM_API void AdvectionSolver_Solve(AdvectionDiffusionSolverHandle solver, double time_end,
                                        double* output_field, int size);

// ===========================================
// 原始龙格-库塔求解器接口实现（向后兼容）
// ===========================================

OCEANSIM_API RungeKuttaSolverHandle RungeKutta_Create(int order, double time_step);

OCEANSIM_API void RungeKutta_Destroy(RungeKuttaSolverHandle solver);

OCEANSIM_API void RungeKutta_SetTimeStep(RungeKuttaSolverHandle solver, double time_step);

OCEANSIM_API double RungeKutta_GetTimeStep(RungeKuttaSolverHandle solver);

// ===========================================
// 原始有限差分求解器接口实现（向后兼容）
// ===========================================

OCEANSIM_API FiniteDifferenceSolverHandle FiniteDiff_Create(int grid_size, double spacing);

OCEANSIM_API void FiniteDiff_Destroy(FiniteDifferenceSolverHandle solver);

OCEANSIM_API void FiniteDiff_ComputeFirstDerivative(FiniteDifferenceSolverHandle solver,
                                                    const double* input, double* output,
                                                    int size, int direction);

OCEANSIM_API void FiniteDiff_ComputeSecondDerivative(FiniteDifferenceSolverHandle solver,
                                                     const double* input, double* output,
                                                     int size, int direction);

// ===========================================
// 原始向量化运算接口实现（向后兼容）
// ===========================================

OCEANSIM_API VectorizedOperationsHandle VectorOps_Create(const PerformanceConfig* config);

OCEANSIM_API void VectorOps_Destroy(VectorizedOperationsHandle ops);

OCEANSIM_API void VectorOps_Add(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size);

OCEANSIM_API void VectorOps_Sub(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size);

OCEANSIM_API double VectorOps_DotProduct(VectorizedOperationsHandle ops, const double* a, const double* b, int size);

OCEANSIM_API double VectorOps_Norm(VectorizedOperationsHandle ops, const double* a, int size);

// ===========================================
// 原始并行计算引擎接口实现（向后兼容）
// ===========================================

OCEANSIM_API ParallelComputeEngineHandle ParallelEngine_Create(const PerformanceConfig* config);

OCEANSIM_API void ParallelEngine_Destroy(ParallelComputeEngineHandle engine);

OCEANSIM_API void ParallelEngine_SetThreadCount(ParallelComputeEngineHandle engine, int num_threads);

OCEANSIM_API int ParallelEngine_GetThreadCount(ParallelComputeEngineHandle engine);

// ===========================================
// 原始数据导出器接口实现（向后兼容）
// ===========================================

OCEANSIM_API DataExporterHandle DataExporter_Create(void);

OCEANSIM_API void DataExporter_Destroy(DataExporterHandle exporter);

OCEANSIM_API int DataExporter_ExportToNetCDF(DataExporterHandle exporter, GridHandle grid, const char* filename);

OCEANSIM_API int DataExporter_ExportToVTK(DataExporterHandle exporter, GridHandle grid, const char* filename);

// ===========================================
// 原始性能分析器接口实现（向后兼容）
// ===========================================

OCEANSIM_API PerformanceProfilerHandle Profiler_Create(void);

OCEANSIM_API void Profiler_Destroy(PerformanceProfilerHandle profiler);

OCEANSIM_API void Profiler_StartTiming(PerformanceProfilerHandle profiler, const char* section_name);

OCEANSIM_API void Profiler_EndTiming(PerformanceProfilerHandle profiler, const char* section_name);

OCEANSIM_API double Profiler_GetElapsedTime(PerformanceProfilerHandle profiler, const char* section_name);

OCEANSIM_API void Profiler_GenerateReport(PerformanceProfilerHandle profiler, const char* filename);

// ===========================================
// 原始工具函数接口实现（向后兼容）
// ===========================================

OCEANSIM_API const char* OceanSim_GetVersion(void);

OCEANSIM_API int OceanSim_Initialize(void);

OCEANSIM_API void OceanSim_Cleanup(void);

OCEANSIM_API void OceanSim_SetLogLevel(int level);

} // extern "C"