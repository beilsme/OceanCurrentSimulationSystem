// ==============================================================================
// 文件：Source/CppCore/bindings/csharp/EnKFWrapper.h
// 作者：beilsm
// 版本：v1.0.0
// 创建时间：2025-07-03
// 功能：集合卡尔曼滤波预测模块的C接口封装，供C#调用
// 说明：提供TOPAZ系统EnKF算法的标准化C接口
// ==============================================================================

#ifndef ENKF_WRAPPER_H
#define ENKF_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef OCEANSIM_EXPORTS
        #define OCEANSIM_API __declspec(dllexport)
    #else
        #define OCEANSIM_API __declspec(dllimport)
    #endif
#else
#define OCEANSIM_API __attribute__((visibility("default")))
#endif

#include <stdint.h>
#include <stdbool.h>

// ==============================================================================
// 类型定义和结构体
// ==============================================================================

/**
 * @brief 海洋状态向量结构
 */
typedef struct {
    double temperature;           // 温度 (°C)
    double salinity;             // 盐度 (PSU)
    double velocity_u;           // U方向流速 (m/s)
    double velocity_v;           // V方向流速 (m/s)
    double velocity_w;           // W方向流速 (m/s)
    double sea_ice_concentration; // 海冰浓度 (0-1)
    double sea_ice_thickness;    // 海冰厚度 (m)
    double sea_surface_height;   // 海表面高度 (m)
} OceanState;

/**
 * @brief EnKF配置参数
 */
typedef struct {
    int ensemble_size;              // 集合大小（默认100）
    double localization_radius;     // 局地化半径 (m)
    double inflation_factor;        // 充气因子
    double regularization_threshold; // 正则化阈值
    bool use_localization;          // 启用局地化
    bool use_inflation;             // 启用充气
    int num_threads;                // 线程数
    bool enable_vectorization;      // 启用向量化
} EnKFConfig;

/**
 * @brief 观测数据结构
 */
typedef struct {
    double* values;                 // 观测值数组
    double* error_variances;        // 观测误差方差数组
    double* locations_x;            // 观测位置X坐标
    double* locations_y;            // 观测位置Y坐标
    int* observation_types;         // 观测类型数组
    bool* quality_flags;            // 质量控制标记
    int num_observations;           // 观测数量
    int64_t timestamp;              // 观测时间戳
} ObservationData;

/**
 * @brief 预报结果结构
 */
typedef struct {
    OceanState* ensemble_mean;      // 集合均值
    double* forecast_covariance;    // 预报协方差（展平的矩阵）
    double ensemble_spread;         // 集合离散度
    int64_t computation_time_ms;    // 计算时间（毫秒）
    bool success;                   // 成功标志
    int state_size;                 // 状态大小
} ForecastResult;

/**
 * @brief 分析结果结构
 */
typedef struct {
    OceanState* analysis_mean;      // 分析均值
    double* analysis_covariance;    // 分析协方差（展平的矩阵）
    double* kalman_gain;            // 卡尔曼增益矩阵（展平）
    double innovation_variance;     // 创新方差
    double observation_impact;      // 观测影响
    int64_t computation_time_ms;    // 计算时间（毫秒）
    bool success;                   // 成功标志
    int state_size;                 // 状态大小
    int obs_size;                   // 观测大小
} AnalysisResult;

/**
 * @brief 性能指标结构
 */
typedef struct {
    double ensemble_spread;         // 集合离散度
    double filter_divergence;       // 滤波器发散度
    double observation_impact;      // 观测影响
    uint64_t total_cycles;          // 总循环次数
    int64_t average_cycle_time_ms;  // 平均循环时间
    double memory_usage_mb;         // 内存使用量（MB）
    double analysis_accuracy;       // 分析精度
} PerformanceMetrics;

/**
 * @brief 网格参数结构
 */
typedef struct {
    int nx, ny, nz;                 // 网格维度
    double dx, dy, dz;              // 网格间距
    double x_min, y_min, z_min;     // 网格起始坐标
    double x_max, y_max, z_max;     // 网格结束坐标
} GridParameters;

// 句柄类型定义
typedef void* EnKFHandle;
typedef void* GridHandle;
typedef void* ObservationOperatorHandle;

// 观测类型枚举
typedef enum {
    OBS_SEA_LEVEL_ANOMALY = 0,      // 海平面高度异常
    OBS_SEA_SURFACE_TEMPERATURE,    // 海表温度
    OBS_SEA_ICE_CONCENTRATION,      // 海冰浓度
    OBS_ARGO_TEMPERATURE,           // Argo温度剖面
    OBS_ARGO_SALINITY,              // Argo盐度剖面
    OBS_SEA_ICE_DRIFT,              // 海冰漂移
    OBS_ALTIMETRY                   // 高度计数据
} ObservationType;

// 充气类型枚举
typedef enum {
    INFLATION_MULTIPLICATIVE = 0,    // 乘性充气
    INFLATION_ADDITIVE,             // 加性充气
    INFLATION_ADAPTIVE,             // 自适应充气
    INFLATION_RELAXATION            // 松弛充气
} InflationType;

// ==============================================================================
// EnKF系统管理接口
// ==============================================================================

/**
 * @brief 创建EnKF系统实例
 * @param config EnKF配置参数
 * @param grid_params 网格参数
 * @return EnKF系统句柄
 */
OCEANSIM_API EnKFHandle EnKF_Create(const EnKFConfig* config, const GridParameters* grid_params);

/**
 * @brief 销毁EnKF系统实例
 * @param handle EnKF系统句柄
 */
OCEANSIM_API void EnKF_Destroy(EnKFHandle handle);

/**
 * @brief 初始化EnKF系统
 * @param handle EnKF系统句柄
 * @param initial_state 初始状态数组
 * @param background_covariance 背景误差协方差矩阵（展平）
 * @param state_size 状态大小
 * @return 成功返回true，失败返回false
 */
OCEANSIM_API bool EnKF_Initialize(EnKFHandle handle,
                                  const OceanState* initial_state,
                                  const double* background_covariance,
                                  int state_size);

/**
 * @brief 获取EnKF系统信息
 * @param handle EnKF系统句柄
 * @param info_buffer 信息缓冲区
 * @param buffer_size 缓冲区大小
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetSystemInfo(EnKFHandle handle, char* info_buffer, int buffer_size);

// ==============================================================================
// EnKF核心算法接口
// ==============================================================================

/**
 * @brief 执行预报步骤
 * @param handle EnKF系统句柄
 * @param time_step 时间步长（秒）
 * @param result 预报结果（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ExecuteForecast(EnKFHandle handle,
                                      double time_step,
                                      ForecastResult* result);

/**
 * @brief 执行分析步骤
 * @param handle EnKF系统句柄
 * @param observations 观测数据
 * @param result 分析结果（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ExecuteAnalysis(EnKFHandle handle,
                                      const ObservationData* observations,
                                      AnalysisResult* result);

/**
 * @brief 执行完整的同化循环
 * @param handle EnKF系统句柄
 * @param time_step 时间步长（秒）
 * @param observations 观测数据
 * @param forecast_result 预报结果（输出）
 * @param analysis_result 分析结果（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ExecuteAssimilationCycle(EnKFHandle handle,
                                               double time_step,
                                               const ObservationData* observations,
                                               ForecastResult* forecast_result,
                                               AnalysisResult* analysis_result);

// ==============================================================================
// 状态访问接口
// ==============================================================================

/**
 * @brief 获取当前集合均值
 * @param handle EnKF系统句柄
 * @param mean_state 均值状态数组（输出）
 * @param state_size 状态大小
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetCurrentMean(EnKFHandle handle,
                                     OceanState* mean_state,
                                     int state_size);

/**
 * @brief 获取当前协方差矩阵
 * @param handle EnKF系统句柄
 * @param covariance 协方差矩阵（展平，输出）
 * @param matrix_size 矩阵大小
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetCurrentCovariance(EnKFHandle handle,
                                           double* covariance,
                                           int matrix_size);

/**
 * @brief 获取指定集合成员
 * @param handle EnKF系统句柄
 * @param member_index 成员索引
 * @param member_state 成员状态数组（输出）
 * @param state_size 状态大小
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetEnsembleMember(EnKFHandle handle,
                                        int member_index,
                                        OceanState* member_state,
                                        int state_size);

/**
 * @brief 获取集合统计信息
 * @param handle EnKF系统句柄
 * @param ensemble_spread 集合离散度（输出）
 * @param ensemble_mean_norm 集合均值范数（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetEnsembleStatistics(EnKFHandle handle,
                                            double* ensemble_spread,
                                            double* ensemble_mean_norm);

// ==============================================================================
// 观测算子接口
// ==============================================================================

/**
 * @brief 创建观测算子
 * @param obs_type 观测类型
 * @param grid_handle 网格句柄
 * @return 观测算子句柄
 */
OCEANSIM_API ObservationOperatorHandle ObsOp_Create(ObservationType obs_type,
                                                    GridHandle grid_handle);

/**
 * @brief 销毁观测算子
 * @param handle 观测算子句柄
 */
OCEANSIM_API void ObsOp_Destroy(ObservationOperatorHandle handle);

/**
 * @brief 应用观测算子
 * @param handle 观测算子句柄
 * @param model_state 模式状态
 * @param state_size 状态大小
 * @param observations 观测值（输出）
 * @param obs_size 观测大小
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int ObsOp_Apply(ObservationOperatorHandle handle,
                             const OceanState* model_state,
                             int state_size,
                             double* observations,
                             int obs_size);

/**
 * @brief 验证观测数据
 * @param handle 观测算子句柄
 * @param observations 观测数据
 * @return 有效返回true，无效返回false
 */
OCEANSIM_API bool ObsOp_Validate(ObservationOperatorHandle handle,
                                 const ObservationData* observations);

// ==============================================================================
// 局地化和充气接口
// ==============================================================================

/**
 * @brief 设置局地化参数
 * @param handle EnKF系统句柄
 * @param radius 局地化半径（米）
 * @param enable 启用局地化
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_SetLocalization(EnKFHandle handle,
                                      double radius,
                                      bool enable);

/**
 * @brief 设置充气参数
 * @param handle EnKF系统句柄
 * @param inflation_type 充气类型
 * @param inflation_factor 充气因子
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_SetInflation(EnKFHandle handle,
                                   InflationType inflation_type,
                                   double inflation_factor);

/**
 * @brief 计算自适应充气因子
 * @param handle EnKF系统句柄
 * @param observations 观测数据
 * @param adaptive_factor 自适应因子（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ComputeAdaptiveInflation(EnKFHandle handle,
                                               const ObservationData* observations,
                                               double* adaptive_factor);

// ==============================================================================
// 性能监控接口
// ==============================================================================

/**
 * @brief 获取性能指标
 * @param handle EnKF系统句柄
 * @param metrics 性能指标（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_GetPerformanceMetrics(EnKFHandle handle,
                                            PerformanceMetrics* metrics);

/**
 * @brief 重置性能统计
 * @param handle EnKF系统句柄
 */
OCEANSIM_API void EnKF_ResetPerformanceStats(EnKFHandle handle);

/**
 * @brief 启用性能分析
 * @param handle EnKF系统句柄
 * @param enable 启用标志
 */
OCEANSIM_API void EnKF_EnableProfiling(EnKFHandle handle, bool enable);

// ==============================================================================
// 验证和诊断接口
// ==============================================================================

/**
 * @brief 验证EnKF系统状态
 * @param handle EnKF系统句柄
 * @return 有效返回true，无效返回false
 */
OCEANSIM_API bool EnKF_ValidateSystemState(EnKFHandle handle);

/**
 * @brief 检查滤波器发散
 * @param handle EnKF系统句柄
 * @param divergence_threshold 发散阈值
 * @return 发散返回true，正常返回false
 */
OCEANSIM_API bool EnKF_CheckFilterDivergence(EnKFHandle handle,
                                             double divergence_threshold);

/**
 * @brief 计算分析增量范数
 * @param handle EnKF系统句柄
 * @param increment_norm 增量范数（输出）
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ComputeAnalysisIncrement(EnKFHandle handle,
                                               double* increment_norm);

/**
 * @brief 执行线性高斯验证测试
 * @param handle EnKF系统句柄
 * @return 通过返回true，失败返回false
 */
OCEANSIM_API bool EnKF_ValidateLinearGaussian(EnKFHandle handle);

/**
 * @brief 执行Lorenz-96验证测试
 * @param handle EnKF系统句柄
 * @return 通过返回true，失败返回false
 */
OCEANSIM_API bool EnKF_ValidateLorenz96(EnKFHandle handle);

// ==============================================================================
// 内存管理接口
// ==============================================================================

/**
 * @brief 释放预报结果内存
 * @param result 预报结果指针
 */
OCEANSIM_API void EnKF_FreeForecastResult(ForecastResult* result);

/**
 * @brief 释放分析结果内存
 * @param result 分析结果指针
 */
OCEANSIM_API void EnKF_FreeAnalysisResult(AnalysisResult* result);

/**
 * @brief 优化内存使用
 * @param handle EnKF系统句柄
 */
OCEANSIM_API void EnKF_OptimizeMemory(EnKFHandle handle);

// ==============================================================================
// 配置和参数接口
// ==============================================================================

/**
 * @brief 获取默认EnKF配置
 * @param config 配置结构（输出）
 */
OCEANSIM_API void EnKF_GetDefaultConfig(EnKFConfig* config);

/**
 * @brief 更新EnKF配置
 * @param handle EnKF系统句柄
 * @param config 新配置参数
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_UpdateConfig(EnKFHandle handle, const EnKFConfig* config);

/**
 * @brief 保存EnKF状态到文件
 * @param handle EnKF系统句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_SaveState(EnKFHandle handle, const char* filename);

/**
 * @brief 从文件加载EnKF状态
 * @param handle EnKF系统句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_LoadState(EnKFHandle handle, const char* filename);

// ==============================================================================
// 错误处理接口
// ==============================================================================

/**
 * @brief 获取最后的错误信息
 * @param handle EnKF系统句柄
 * @param error_buffer 错误信息缓冲区
 * @param buffer_size 缓冲区大小
 * @return 错误代码
 */
OCEANSIM_API int EnKF_GetLastError(EnKFHandle handle, char* error_buffer, int buffer_size);

/**
 * @brief 清除错误状态
 * @param handle EnKF系统句柄
 */
OCEANSIM_API void EnKF_ClearError(EnKFHandle handle);

// ==============================================================================
// 工具函数接口
// ==============================================================================

/**
 * @brief 创建观测数据结构
 * @param num_observations 观测数量
 * @return 观测数据指针
 */
OCEANSIM_API ObservationData* EnKF_CreateObservationData(int num_observations);

/**
 * @brief 释放观测数据内存
 * @param obs_data 观测数据指针
 */
OCEANSIM_API void EnKF_FreeObservationData(ObservationData* obs_data);

/**
 * @brief 设置观测值
 * @param obs_data 观测数据
 * @param index 观测索引
 * @param value 观测值
 * @param error_variance 误差方差
 * @param x_location X坐标
 * @param y_location Y坐标
 * @param obs_type 观测类型
 */
OCEANSIM_API void EnKF_SetObservation(ObservationData* obs_data,
                                      int index,
                                      double value,
                                      double error_variance,
                                      double x_location,
                                      double y_location,
                                      ObservationType obs_type);

/**
 * @brief 创建海洋状态数组
 * @param state_size 状态大小
 * @return 状态数组指针
 */
OCEANSIM_API OceanState* EnKF_CreateStateArray(int state_size);

/**
 * @brief 释放海洋状态数组
 * @param state_array 状态数组指针
 */
OCEANSIM_API void EnKF_FreeStateArray(OceanState* state_array);

/**
 * @brief 复制海洋状态
 * @param dest 目标状态
 * @param src 源状态
 */
OCEANSIM_API void EnKF_CopyState(OceanState* dest, const OceanState* src);

/**
 * @brief 初始化海洋状态为零
 * @param state 状态指针
 */
OCEANSIM_API void EnKF_InitializeStateZero(OceanState* state);

/**
 * @brief 应用物理约束到状态
 * @param state 状态指针
 */
OCEANSIM_API void EnKF_ApplyPhysicalConstraints(OceanState* state);

/**
 * @brief 检查状态物理有效性
 * @param state 状态指针
 * @return 有效返回true，无效返回false
 */
OCEANSIM_API bool EnKF_IsStatePhysicallyValid(const OceanState* state);

// ==============================================================================
// 高级功能接口
// ==============================================================================

/**
 * @brief 设置自定义海洋模式积分器
 * @param handle EnKF系统句柄
 * @param integrator_func 积分器函数指针
 * @param user_data 用户数据
 * @return 成功返回0，失败返回错误代码
 */
typedef void (*OceanModelIntegrator)(OceanState* state, double time_step, void* user_data);
OCEANSIM_API int EnKF_SetCustomIntegrator(EnKFHandle handle,
                                          OceanModelIntegrator integrator_func,
                                          void* user_data);

/**
 * @brief 设置自定义观测算子
 * @param handle EnKF系统句柄
 * @param obs_operator_func 观测算子函数指针
 * @param user_data 用户数据
 * @return 成功返回0，失败返回错误代码
 */
typedef void (*CustomObservationOperator)(const OceanState* model_state,
                                          int state_size,
                                          double* observations,
                                          int obs_size,
                                          void* user_data);
OCEANSIM_API int EnKF_SetCustomObservationOperator(EnKFHandle handle,
                                                   CustomObservationOperator obs_operator_func,
                                                   void* user_data);

/**
 * @brief 启用/禁用并行计算
 * @param handle EnKF系统句柄
 * @param enable 启用标志
 * @param num_threads 线程数（0表示使用默认值）
 */
OCEANSIM_API void EnKF_SetParallelExecution(EnKFHandle handle, bool enable, int num_threads);

/**
 * @brief 设置数值稳定性参数
 * @param handle EnKF系统句柄
 * @param regularization_threshold 正则化阈值
 * @param condition_number_threshold 条件数阈值
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_SetStabilityParameters(EnKFHandle handle,
                                             double regularization_threshold,
                                             double condition_number_threshold);

// ==============================================================================
// 调试和日志接口
// ==============================================================================

/**
 * @brief 设置日志级别
 * @param handle EnKF系统句柄
 * @param log_level 日志级别 (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG)
 */
OCEANSIM_API void EnKF_SetLogLevel(EnKFHandle handle, int log_level);

/**
 * @brief 启用详细调试输出
 * @param handle EnKF系统句柄
 * @param enable 启用标志
 */
OCEANSIM_API void EnKF_EnableDebugOutput(EnKFHandle handle, bool enable);

/**
 * @brief 导出集合到文件
 * @param handle EnKF系统句柄
 * @param filename 文件名
 * @param format 文件格式 ("netcdf", "hdf5", "binary")
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ExportEnsemble(EnKFHandle handle,
                                     const char* filename,
                                     const char* format);

/**
 * @brief 导出协方差矩阵到文件
 * @param handle EnKF系统句柄
 * @param filename 文件名
 * @return 成功返回0，失败返回错误代码
 */
OCEANSIM_API int EnKF_ExportCovariance(EnKFHandle handle, const char* filename);

// ==============================================================================
// 版本和信息接口
// ==============================================================================

/**
 * @brief 获取EnKF库版本信息
 * @param version_buffer 版本信息缓冲区
 * @param buffer_size 缓冲区大小
 */
OCEANSIM_API void EnKF_GetVersion(char* version_buffer, int buffer_size);

/**
 * @brief 获取编译信息
 * @param build_info_buffer 编译信息缓冲区
 * @param buffer_size 缓冲区大小
 */
OCEANSIM_API void EnKF_GetBuildInfo(char* build_info_buffer, int buffer_size);

/**
 * @brief 检查硬件支持
 * @param feature_name 特性名称 ("avx", "sse", "openmp", "cuda")
 * @return 支持返回true，不支持返回false
 */
OCEANSIM_API bool EnKF_CheckHardwareSupport(const char* feature_name);

#ifdef __cplusplus
}
#endif

#endif // ENKF_WRAPPER_H