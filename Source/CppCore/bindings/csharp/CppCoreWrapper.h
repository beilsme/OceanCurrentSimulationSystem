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
typedef void* ParticleSimulatorHandle;
typedef void* CurrentFieldSolverHandle;
typedef void* AdvectionDiffusionSolverHandle;
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

typedef void* CurrentFieldSolverHandle;

typedef struct {
    double gravity;
    double coriolis_f;
    double beta;
    double viscosity_h;
    double viscosity_v;
    double diffusivity_h;
    double diffusivity_v;
    double reference_density;
} PhysicalParameters;

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

typedef void* GridDataHandle;

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

typedef void* AdvectionDiffusionHandle;

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

} // extern "C"