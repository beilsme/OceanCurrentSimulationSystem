// ==============================================================================
// 文件路径：Source/CppCore/bindings/csharp/CppCoreWrapper.h
// 作者：beilsm
// 版本号：v1.0.0
// 创建时间：2025-07-01
// 最新更改时间：2025-07-01
// ==============================================================================
// 📝 功能说明：
//   C++核心模块的C#绑定包装器头文件
//   提供C风格的接口供C# P/Invoke调用
// ==============================================================================

#ifndef OCEANSIM_CSHARP_WRAPPER_H
#define OCEANSIM_CSHARP_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// 平台特定的导出宏定义
#ifdef _WIN32
#ifdef OCEANSIM_CSHARP_EXPORTS
        #define OCEANSIM_API __declspec(dllexport)
    #else
        #define OCEANSIM_API __declspec(dllimport)
    #endif
#else
#define OCEANSIM_API __attribute__((visibility("default")))
#endif

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

// 网格配置结构
typedef struct {
    int nx, ny, nz;           // 网格维度
    double dx, dy, dz;        // 网格间距
    Vector3D origin;          // 原点坐标
    int coordinate_system;    // 坐标系类型
    int grid_type;           // 网格类型
} GridConfig;

// 物理参数结构
typedef struct {
    double density;           // 密度
    double viscosity;         // 粘性系数
    double gravity;           // 重力
    double coriolis_param;    // 科里奥利参数
    double wind_stress_x;     // 风应力X分量
    double wind_stress_y;     // 风应力Y分量
} PhysicalParameters;

// 数值求解参数
typedef struct {
    double time_step;         // 时间步长
    double diffusion_coeff;   // 扩散系数
    int scheme_type;          // 数值格式类型
    int integration_method;   // 时间积分方法
    double cfl_number;        // CFL数
} SolverParameters;

// 执行策略和性能配置
typedef struct {
    int execution_policy;     // 执行策略
    int num_threads;          // 线程数量
    int simd_type;           // SIMD类型
    int priority;            // 优先级
} PerformanceConfig;

// 枚举类型定义
enum CoordinateSystemType {
    COORDINATE_CARTESIAN = 0,
    COORDINATE_SPHERICAL = 1,
    COORDINATE_HYBRID_SIGMA = 2,
    COORDINATE_ISOPYCNAL = 3
};

enum GridTypeEnum {
    GRID_REGULAR = 0,
    GRID_CURVILINEAR = 1,
    GRID_UNSTRUCTURED = 2
};

enum NumericalSchemeType {
    SCHEME_UPWIND = 0,
    SCHEME_CENTRAL = 1,
    SCHEME_TVD_SUPERBEE = 2,
    SCHEME_WENO = 3
};

enum TimeIntegrationType {
    TIME_EULER = 0,
    TIME_RUNGE_KUTTA_2 = 1,
    TIME_RUNGE_KUTTA_3 = 2,
    TIME_RUNGE_KUTTA_4 = 3,
    TIME_ADAMS_BASHFORTH = 4
};

enum ExecutionPolicyType {
    EXECUTION_SEQUENTIAL = 0,
    EXECUTION_PARALLEL = 1,
    EXECUTION_VECTORIZED = 2,
    EXECUTION_HYBRID_PARALLEL = 3
};

enum SimdTypeEnum {
    SIMD_NONE = 0,
    SIMD_SSE = 1,
    SIMD_AVX = 2,
    SIMD_AVX2 = 3,
    SIMD_AVX512 = 4,
    SIMD_NEON = 5
};

// ===========================================
// 网格数据结构接口
// ===========================================

/**
 * @brief 创建网格数据结构
 * @param config 网格配置参数
 * @return 网格句柄
 */
OCEANSIM_API GridHandle Grid_Create(const GridConfig* config);

/**
 * @brief 销毁网格数据结构
 * @param grid 网格句柄
 */
OCEANSIM_API void Grid_Destroy(GridHandle grid);

/**
 * @brief 获取网格维度
 * @param grid 网格句柄
 * @param nx X方向维度（输出）
 * @param ny Y方向维度（输出）
 * @param nz Z方向维度（输出）
 */
OCEANSIM_API void Grid_GetDimensions(GridHandle grid, int* nx, int* ny, int* nz);

/**
 * @brief 设置标量场数据
 * @param grid 网格句柄
 * @param data 标量场数据数组
 * @param size 数据数组大小
 * @param field_name 场名称
 */
OCEANSIM_API void Grid_SetScalarField(GridHandle grid, const double* data, int size, const char* field_name);

/**
 * @brief 设置向量场数据
 * @param grid 网格句柄
 * @param u_data U分量数据数组
 * @param v_data V分量数据数组
 * @param w_data W分量数据数组
 * @param size 数据数组大小
 * @param field_name 场名称
 */
OCEANSIM_API void Grid_SetVectorField(GridHandle grid, const double* u_data, const double* v_data,
                                      const double* w_data, int size, const char* field_name);

/**
 * @brief 获取标量场数据
 * @param grid 网格句柄
 * @param field_name 场名称
 * @param data 输出数据数组（需预分配）
 * @param size 数据数组大小
 */
OCEANSIM_API void Grid_GetScalarField(GridHandle grid, const char* field_name, double* data, int size);

/**
 * @brief 进行插值计算
 * @param grid 网格句柄
 * @param position 插值位置
 * @param field_name 场名称
 * @param method 插值方法
 * @return 插值结果
 */
OCEANSIM_API double Grid_Interpolate(GridHandle grid, const Vector3D* position,
                                     const char* field_name, int method);

// ===========================================
// 粒子模拟器接口
// ===========================================

/**
 * @brief 创建粒子模拟器
 * @param grid 网格句柄
 * @param solver_handle 求解器句柄
 * @return 粒子模拟器句柄
 */
OCEANSIM_API ParticleSimulatorHandle ParticleSim_Create(GridHandle grid, RungeKuttaSolverHandle solver_handle);

/**
 * @brief 销毁粒子模拟器
 * @param simulator 粒子模拟器句柄
 */
OCEANSIM_API void ParticleSim_Destroy(ParticleSimulatorHandle simulator);

/**
 * @brief 初始化粒子
 * @param simulator 粒子模拟器句柄
 * @param positions 初始位置数组
 * @param count 粒子数量
 */
OCEANSIM_API void ParticleSim_InitializeParticles(ParticleSimulatorHandle simulator,
                                                  const Vector3D* positions, int count);

/**
 * @brief 执行时间步进
 * @param simulator 粒子模拟器句柄
 * @param time_step 时间步长
 */
OCEANSIM_API void ParticleSim_StepForward(ParticleSimulatorHandle simulator, double time_step);

/**
 * @brief 获取粒子数据
 * @param simulator 粒子模拟器句柄
 * @param particles 粒子数据数组（输出）
 * @param count 粒子数量
 */
OCEANSIM_API void ParticleSim_GetParticles(ParticleSimulatorHandle simulator, ParticleData* particles, int count);

/**
 * @brief 获取粒子数量
 * @param simulator 粒子模拟器句柄
 * @return 粒子数量
 */
OCEANSIM_API int ParticleSim_GetParticleCount(ParticleSimulatorHandle simulator);

// ===========================================
// 洋流场求解器接口
// ===========================================

/**
 * @brief 创建洋流场求解器
 * @param grid 网格句柄
 * @param params 物理参数
 * @return 洋流场求解器句柄
 */
OCEANSIM_API CurrentFieldSolverHandle CurrentSolver_Create(GridHandle grid, const PhysicalParameters* params);

/**
 * @brief 销毁洋流场求解器
 * @param solver 求解器句柄
 */
OCEANSIM_API void CurrentSolver_Destroy(CurrentFieldSolverHandle solver);

/**
 * @brief 计算速度场
 * @param solver 求解器句柄
 * @param time_step 时间步长
 */
OCEANSIM_API void CurrentSolver_ComputeVelocityField(CurrentFieldSolverHandle solver, double time_step);

/**
 * @brief 设置边界条件
 * @param solver 求解器句柄
 * @param boundary_type 边界类型
 * @param values 边界值数组
 * @param size 数组大小
 */
OCEANSIM_API void CurrentSolver_SetBoundaryConditions(CurrentFieldSolverHandle solver, int boundary_type,
                                                      const double* values, int size);

/**
 * @brief 计算动能
 * @param solver 求解器句柄
 * @return 动能值
 */
OCEANSIM_API double CurrentSolver_ComputeKineticEnergy(CurrentFieldSolverHandle solver);

/**
 * @brief 检查质量守恒
 * @param solver 求解器句柄
 * @return 质量守恒误差
 */
OCEANSIM_API double CurrentSolver_CheckMassConservation(CurrentFieldSolverHandle solver);

// ===========================================
// 平流扩散求解器接口
// ===========================================

/**
 * @brief 创建平流扩散求解器
 * @param grid 网格句柄
 * @param params 求解器参数
 * @return 平流扩散求解器句柄
 */
OCEANSIM_API AdvectionDiffusionSolverHandle AdvectionSolver_Create(GridHandle grid, const SolverParameters* params);

/**
 * @brief 销毁平流扩散求解器
 * @param solver 求解器句柄
 */
OCEANSIM_API void AdvectionSolver_Destroy(AdvectionDiffusionSolverHandle solver);

/**
 * @brief 设置初始条件
 * @param solver 求解器句柄
 * @param initial_field 初始场数据
 * @param size 数据大小
 */
OCEANSIM_API void AdvectionSolver_SetInitialCondition(AdvectionDiffusionSolverHandle solver,
                                                      const double* initial_field, int size);

/**
 * @brief 设置速度场
 * @param solver 求解器句柄
 * @param u_field U速度分量
 * @param v_field V速度分量
 * @param w_field W速度分量
 * @param size 数据大小
 */
OCEANSIM_API void AdvectionSolver_SetVelocityField(AdvectionDiffusionSolverHandle solver,
                                                   const double* u_field, const double* v_field,
                                                   const double* w_field, int size);

/**
 * @brief 执行求解
 * @param solver 求解器句柄
 * @param time_end 结束时间
 * @param output_field 输出场数据（需预分配）
 * @param size 数据大小
 */
OCEANSIM_API void AdvectionSolver_Solve(AdvectionDiffusionSolverHandle solver, double time_end,
                                        double* output_field, int size);

// ===========================================
// 龙格-库塔求解器接口
// ===========================================

/**
 * @brief 创建龙格-库塔求解器
 * @param order 求解器阶数（2、3或4）
 * @param time_step 时间步长
 * @return 龙格-库塔求解器句柄
 */
OCEANSIM_API RungeKuttaSolverHandle RungeKutta_Create(int order, double time_step);

/**
 * @brief 销毁龙格-库塔求解器
 * @param solver 求解器句柄
 */
OCEANSIM_API void RungeKutta_Destroy(RungeKuttaSolverHandle solver);

/**
 * @brief 设置时间步长
 * @param solver 求解器句柄
 * @param time_step 新的时间步长
 */
OCEANSIM_API void RungeKutta_SetTimeStep(RungeKuttaSolverHandle solver, double time_step);

/**
 * @brief 获取当前时间步长
 * @param solver 求解器句柄
 * @return 当前时间步长
 */
OCEANSIM_API double RungeKutta_GetTimeStep(RungeKuttaSolverHandle solver);

// ===========================================
// 有限差分求解器接口
// ===========================================

/**
 * @brief 创建有限差分求解器
 * @param grid_size 网格大小
 * @param spacing 网格间距
 * @return 有限差分求解器句柄
 */
OCEANSIM_API FiniteDifferenceSolverHandle FiniteDiff_Create(int grid_size, double spacing);

/**
 * @brief 销毁有限差分求解器
 * @param solver 求解器句柄
 */
OCEANSIM_API void FiniteDiff_Destroy(FiniteDifferenceSolverHandle solver);

/**
 * @brief 计算一阶导数
 * @param solver 求解器句柄
 * @param input 输入数据
 * @param output 输出导数（需预分配）
 * @param size 数据大小
 * @param direction 求导方向（0=x, 1=y, 2=z）
 */
OCEANSIM_API void FiniteDiff_ComputeFirstDerivative(FiniteDifferenceSolverHandle solver,
                                                    const double* input, double* output,
                                                    int size, int direction);

/**
 * @brief 计算二阶导数
 * @param solver 求解器句柄
 * @param input 输入数据
 * @param output 输出导数（需预分配）
 * @param size 数据大小
 * @param direction 求导方向
 */
OCEANSIM_API void FiniteDiff_ComputeSecondDerivative(FiniteDifferenceSolverHandle solver,
                                                     const double* input, double* output,
                                                     int size, int direction);

// ===========================================
// 向量化运算接口
// ===========================================

/**
 * @brief 创建向量化运算引擎
 * @param config 性能配置
 * @return 向量化运算句柄
 */
OCEANSIM_API VectorizedOperationsHandle VectorOps_Create(const PerformanceConfig* config);

/**
 * @brief 销毁向量化运算引擎
 * @param ops 向量化运算句柄
 */
OCEANSIM_API void VectorOps_Destroy(VectorizedOperationsHandle ops);

/**
 * @brief 向量加法：result = a + b
 * @param ops 向量化运算句柄
 * @param a 输入向量A
 * @param b 输入向量B
 * @param result 输出向量
 * @param size 向量大小
 */
OCEANSIM_API void VectorOps_Add(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size);

/**
 * @brief 向量减法：result = a - b
 * @param ops 向量化运算句柄
 * @param a 输入向量A
 * @param b 输入向量B
 * @param result 输出向量
 * @param size 向量大小
 */
OCEANSIM_API void VectorOps_Sub(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size);

/**
 * @brief 向量点积
 * @param ops 向量化运算句柄
 * @param a 输入向量A
 * @param b 输入向量B
 * @param size 向量大小
 * @return 点积结果
 */
OCEANSIM_API double VectorOps_DotProduct(VectorizedOperationsHandle ops, const double* a, const double* b, int size);

/**
 * @brief 向量范数
 * @param ops 向量化运算句柄
 * @param a 输入向量
 * @param size 向量大小
 * @return 范数值
 */
OCEANSIM_API double VectorOps_Norm(VectorizedOperationsHandle ops, const double* a, int size);

// ===========================================
// 并行计算引擎接口
// ===========================================

/**
 * @brief 创建并行计算引擎
 * @param config 性能配置
 * @return 并行计算引擎句柄
 */
OCEANSIM_API ParallelComputeEngineHandle ParallelEngine_Create(const PerformanceConfig* config);

/**
 * @brief 销毁并行计算引擎
 * @param engine 并行计算引擎句柄
 */
OCEANSIM_API void ParallelEngine_Destroy(ParallelComputeEngineHandle engine);

/**
 * @brief 设置线程数量
 * @param engine 并行计算引擎句柄
 * @param num_threads 线程数量
 */
OCEANSIM_API void ParallelEngine_SetThreadCount(ParallelComputeEngineHandle engine, int num_threads);

/**
 * @brief 获取线程数量
 * @param engine 并行计算引擎句柄
 * @return 当前线程数量
 */
OCEANSIM_API int ParallelEngine_GetThreadCount(ParallelComputeEngineHandle engine);

// ===========================================
// 数据导出器接口
// ===========================================

/**
 * @brief 创建数据导出器
 * @return 数据导出器句柄
 */
OCEANSIM_API DataExporterHandle DataExporter_Create(void);

/**
 * @brief 销毁数据导出器
 * @param exporter 数据导出器句柄
 */
OCEANSIM_API void DataExporter_Destroy(DataExporterHandle exporter);

/**
 * @brief 导出为NetCDF格式
 * @param exporter 数据导出器句柄
 * @param grid 网格句柄
 * @param filename 文件名
 * @return 导出是否成功（1=成功，0=失败）
 */
OCEANSIM_API int DataExporter_ExportToNetCDF(DataExporterHandle exporter, GridHandle grid, const char* filename);

/**
 * @brief 导出为VTK格式
 * @param exporter 数据导出器句柄
 * @param grid 网格句柄
 * @param filename 文件名
 * @return 导出是否成功（1=成功，0=失败）
 */
OCEANSIM_API int DataExporter_ExportToVTK(DataExporterHandle exporter, GridHandle grid, const char* filename);

// ===========================================
// 性能分析器接口
// ===========================================

/**
 * @brief 创建性能分析器
 * @return 性能分析器句柄
 */
OCEANSIM_API PerformanceProfilerHandle Profiler_Create(void);

/**
 * @brief 销毁性能分析器
 * @param profiler 性能分析器句柄
 */
OCEANSIM_API void Profiler_Destroy(PerformanceProfilerHandle profiler);

/**
 * @brief 开始计时
 * @param profiler 性能分析器句柄
 * @param section_name 计时段名称
 */
OCEANSIM_API void Profiler_StartTiming(PerformanceProfilerHandle profiler, const char* section_name);

/**
 * @brief 结束计时
 * @param profiler 性能分析器句柄
 * @param section_name 计时段名称
 */
OCEANSIM_API void Profiler_EndTiming(PerformanceProfilerHandle profiler, const char* section_name);

/**
 * @brief 获取计时结果
 * @param profiler 性能分析器句柄
 * @param section_name 计时段名称
 * @return 耗时（毫秒）
 */
OCEANSIM_API double Profiler_GetElapsedTime(PerformanceProfilerHandle profiler, const char* section_name);

/**
 * @brief 生成性能报告
 * @param profiler 性能分析器句柄
 * @param filename 报告文件名
 */
OCEANSIM_API void Profiler_GenerateReport(PerformanceProfilerHandle profiler, const char* filename);

// ===========================================
// 工具函数接口
// ===========================================

/**
 * @brief 获取库版本信息
 * @return 版本字符串
 */
OCEANSIM_API const char* OceanSim_GetVersion(void);

/**
 * @brief 初始化库
 * @return 初始化是否成功（1=成功，0=失败）
 */
OCEANSIM_API int OceanSim_Initialize(void);

/**
 * @brief 清理库资源
 */
OCEANSIM_API void OceanSim_Cleanup(void);

/**
 * @brief 设置日志级别
 * @param level 日志级别（0=Debug, 1=Info, 2=Warning, 3=Error）
 */
OCEANSIM_API void OceanSim_SetLogLevel(int level);

#ifdef __cplusplus
}
#endif

#endif // OCEANSIM_CSHARP_WRAPPER_H