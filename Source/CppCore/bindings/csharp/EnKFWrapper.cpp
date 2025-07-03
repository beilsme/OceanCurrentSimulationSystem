// ==============================================================================
// 文件：Source/CppCore/bindings/csharp/EnKFWrapper.cpp
// 作者：beilsm
// 版本：v1.0.0
// 创建时间：2025-07-03
// 功能：集合卡尔曼滤波预测模块的C接口实现
// 说明：将C++的EnKF类包装为C接口供外部调用
// ==============================================================================

#include "EnKFWrapper.h"
#include "prediction/EnsembleKalmanFilter.h"
#include "data/GridDataStructure.h"
#include <memory>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <exception>
#include <chrono>
#include <unordered_map>

using namespace OceanSim::Prediction;
using namespace OceanSim::Data;

// ==============================================================================
// 内部数据结构和工具函数
// ==============================================================================

/**
 * @brief EnKF系统包装器结构
 */
struct EnKFWrapper {
    std::unique_ptr<EnsembleKalmanFilter> filter;
    std::shared_ptr<GridDataStructure> grid;
    EnKFConfiguration config;
    std::string last_error;
    bool is_initialized;

    // 性能统计
    std::chrono::steady_clock::time_point creation_time;
    uint64_t cycle_count;

    EnKFWrapper() : is_initialized(false), cycle_count(0) {
        creation_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief 观测算子包装器结构
 */
struct ObservationOperatorWrapper {
    std::unique_ptr<ObservationOperator> operator_;
    ObservationType type;
    std::shared_ptr<GridDataStructure> grid;
};

// 全局错误状态管理
static std::unordered_map<void*, std::string> g_error_messages;
static std::mutex g_error_mutex;

// 内部工具函数
namespace {
    /**
     * @brief 设置错误消息
     */
    void SetError(void* handle, const std::string& message) {
        std::lock_guard<std::mutex> lock(g_error_mutex);
        g_error_messages[handle] = message;
    }

    /**
     * @brief 获取错误消息
     */
    std::string GetError(void* handle) {
        std::lock_guard<std::mutex> lock(g_error_mutex);
        auto it = g_error_messages.find(handle);
        return (it != g_error_messages.end()) ? it->second : "未知错误";
    }

    /**
     * @brief 清除错误消息
     */
    void ClearError(void* handle) {
        std::lock_guard<std::mutex> lock(g_error_mutex);
        g_error_messages.erase(handle);
    }

    /**
     * @brief 转换C配置到C++配置
     */
    EnKFConfiguration ConvertConfig(const EnKFConfig* c_config) {
        EnKFConfiguration cpp_config;
        cpp_config.ensemble_size = c_config->ensemble_size;
        cpp_config.localization_radius = c_config->localization_radius;
        cpp_config.inflation_factor = c_config->inflation_factor;
        cpp_config.regularization_threshold = c_config->regularization_threshold;
        cpp_config.use_localization = c_config->use_localization;
        cpp_config.use_inflation = c_config->use_inflation;
        cpp_config.num_threads = c_config->num_threads;
        cpp_config.enable_vectorization = c_config->enable_vectorization;
        return cpp_config;
    }

    /**
     * @brief 转换C++状态向量到C状态向量
     */
    void ConvertStateVector(const OceanStateVector& cpp_state, OceanState* c_state) {
        c_state->temperature = cpp_state.temperature;
        c_state->salinity = cpp_state.salinity;
        c_state->velocity_u = cpp_state.velocity[0];
        c_state->velocity_v = cpp_state.velocity[1];
        c_state->velocity_w = cpp_state.velocity[2];
        c_state->sea_ice_concentration = cpp_state.sea_ice_concentration;
        c_state->sea_ice_thickness = cpp_state.sea_ice_thickness;
        c_state->sea_surface_height = cpp_state.sea_surface_height;
    }

    /**
     * @brief 转换C状态向量到C++状态向量
     */
    OceanStateVector ConvertStateVector(const OceanState* c_state) {
        OceanStateVector cpp_state;
        cpp_state.temperature = c_state->temperature;
        cpp_state.salinity = c_state->salinity;
        cpp_state.velocity = Eigen::Vector3d(c_state->velocity_u,
                                             c_state->velocity_v,
                                             c_state->velocity_w);
        cpp_state.sea_ice_concentration = c_state->sea_ice_concentration;
        cpp_state.sea_ice_thickness = c_state->sea_ice_thickness;
        cpp_state.sea_surface_height = c_state->sea_surface_height;
        return cpp_state;
    }

    /**
     * @brief 创建网格结构
     */
    std::shared_ptr<GridDataStructure> CreateGrid(const GridParameters* params) {
        auto grid = std::make_shared<GridDataStructure>();

        // 设置网格参数
        std::vector<int> dimensions = {params->nx, params->ny, params->nz};
        std::vector<double> spacing = {params->dx, params->dy, params->dz};
        std::vector<double> origin = {params->x_min, params->y_min, params->z_min};

        grid->initialize(dimensions, spacing, origin);

        return grid;
    }
}

// ==============================================================================
// EnKF系统管理接口实现
// ==============================================================================

OCEANSIM_API EnKFHandle EnKF_Create(const EnKFConfig* config, const GridParameters* grid_params) {
    try {
        if (!config || !grid_params) {
            return nullptr;
        }

        auto wrapper = std::make_unique<EnKFWrapper>();

        // 转换配置
        wrapper->config = ConvertConfig(config);

        // 验证配置
        if (!wrapper->config.validate()) {
            SetError(wrapper.get(), "EnKF配置参数无效");
            return nullptr;
        }

        // 创建网格
        wrapper->grid = CreateGrid(grid_params);

        // 创建EnKF滤波器
        wrapper->filter = std::make_unique<EnsembleKalmanFilter>(wrapper->config, wrapper->grid);

        return wrapper.release();

    } catch (const std::exception& e) {
        SetError(nullptr, std::string("创建EnKF系统失败: ") + e.what());
        return nullptr;
    }
}

OCEANSIM_API void EnKF_Destroy(EnKFHandle handle) {
    if (handle) {
        auto wrapper = static_cast<EnKFWrapper*>(handle);
        ClearError(handle);
        delete wrapper;
    }
}

OCEANSIM_API bool EnKF_Initialize(EnKFHandle handle,
                                  const OceanState* initial_state,
                                  const double* background_covariance,
                                  int state_size) {
    try {
        if (!handle || !initial_state || !background_covariance || state_size <= 0) {
            return false;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 转换初始状态
        std::vector<OceanStateVector> cpp_initial_state(state_size);
        for (int i = 0; i < state_size; ++i) {
            cpp_initial_state[i] = ConvertStateVector(&initial_state[i]);
        }

        // 转换背景误差协方差矩阵
        int covariance_size = state_size * 7; // 7个状态变量
        Eigen::MatrixXd cpp_covariance = Eigen::Map<const Eigen::MatrixXd>(
                background_covariance, covariance_size, covariance_size);

        // 初始化滤波器
        bool success = wrapper->filter->initialize(cpp_initial_state, cpp_covariance);

        if (success) {
            wrapper->is_initialized = true;
            ClearError(handle);
        } else {
            SetError(handle, "EnKF系统初始化失败");
        }

        return success;

    } catch (const std::exception& e) {
        SetError(handle, std::string("初始化失败: ") + e.what());
        return false;
    }
}

OCEANSIM_API int EnKF_GetSystemInfo(EnKFHandle handle, char* info_buffer, int buffer_size) {
    try {
        if (!handle || !info_buffer || buffer_size <= 0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        std::string info = "EnKF系统信息:\n";
        info += "集合大小: " + std::to_string(wrapper->config.ensemble_size) + "\n";
        info += "局地化半径: " + std::to_string(wrapper->config.localization_radius) + " m\n";
        info += "充气因子: " + std::to_string(wrapper->config.inflation_factor) + "\n";
        info += "线程数: " + std::to_string(wrapper->config.num_threads) + "\n";
        info += "初始化状态: " + (wrapper->is_initialized ? "已初始化" : "未初始化") + "\n";
        info += "同化循环次数: " + std::to_string(wrapper->cycle_count) + "\n";

        if (info.length() >= static_cast<size_t>(buffer_size)) {
            return static_cast<int>(info.length()) + 1; // 需要的缓冲区大小
        }

        std::strcpy(info_buffer, info.c_str());
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取系统信息失败: ") + e.what());
        return -1;
    }
}

// ==============================================================================
// EnKF核心算法接口实现
// ==============================================================================

OCEANSIM_API int EnKF_ExecuteForecast(EnKFHandle handle,
                                      double time_step,
                                      ForecastResult* result) {
    try {
        if (!handle || !result || time_step <= 0.0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        auto cpp_covariance = wrapper->filter->getCurrentCovariance();
        int expected_size = cpp_covariance.rows() * cpp_covariance.cols();

        if (expected_size != matrix_size) {
            SetError(handle, "协方差矩阵大小不匹配");
            return -3;
        }

        // 复制矩阵数据（按行展平）
        Eigen::Map<Eigen::MatrixXd>(covariance, cpp_covariance.rows(), cpp_covariance.cols()) = cpp_covariance;

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取当前协方差失败: ") + e.what());
        return -4;
    }
}

OCEANSIM_API int EnKF_GetEnsembleMember(EnKFHandle handle,
                                        int member_index,
                                        OceanState* member_state,
                                        int state_size) {
    try {
        if (!handle || !member_state || state_size <= 0 || member_index < 0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        if (member_index >= wrapper->config.ensemble_size) {
            SetError(handle, "集合成员索引超出范围");
            return -3;
        }

        const auto& ensemble = wrapper->filter->getCurrentEnsemble();
        const OceanStateVector* member = ensemble.getMember(member_index);

        if (ensemble.getStateSize() != static_cast<size_t>(state_size)) {
            SetError(handle, "状态大小不匹配");
            return -4;
        }

        for (int i = 0; i < state_size; ++i) {
            ConvertStateVector(member[i], &member_state[i]);
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取集合成员失败: ") + e.what());
        return -5;
    }
}

OCEANSIM_API int EnKF_GetEnsembleStatistics(EnKFHandle handle,
                                            double* ensemble_spread,
                                            double* ensemble_mean_norm) {
    try {
        if (!handle || !ensemble_spread || !ensemble_mean_norm) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        const auto& ensemble = wrapper->filter->getCurrentEnsemble();
        *ensemble_spread = ensemble.computeEnsembleSpread();

        // 计算集合均值的范数
        auto mean = wrapper->filter->getCurrentMean();
        double norm = 0.0;
        for (const auto& state : mean) {
            norm += state.temperature * state.temperature +
                    state.salinity * state.salinity +
                    state.velocity.squaredNorm() +
                    state.sea_ice_concentration * state.sea_ice_concentration;
        }
        *ensemble_mean_norm = std::sqrt(norm / mean.size());

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取集合统计失败: ") + e.what());
        return -3;
    }
}

// ==============================================================================
// 观测算子接口实现
// ==============================================================================

OCEANSIM_API ObservationOperatorHandle ObsOp_Create(ObservationType obs_type,
                                                    GridHandle grid_handle) {
    try {
        auto grid = static_cast<GridDataStructure*>(grid_handle);
        if (!grid) {
            return nullptr;
        }

        auto wrapper = std::make_unique<ObservationOperatorWrapper>();
        wrapper->type = obs_type;
        wrapper->grid = std::shared_ptr<GridDataStructure>(grid, [](GridDataStructure*){});

        switch (obs_type) {
            case OBS_SEA_LEVEL_ANOMALY:
                wrapper->operator_ = std::make_unique<SeaLevelAnomalyOperator>(wrapper->grid);
                break;
            case OBS_SEA_SURFACE_TEMPERATURE:
                wrapper->operator_ = std::make_unique<SeaSurfaceTemperatureOperator>(wrapper->grid);
                break;
            case OBS_SEA_ICE_CONCENTRATION:
                wrapper->operator_ = std::make_unique<SeaIceConcentrationOperator>(wrapper->grid);
                break;
            default:
                SetError(wrapper.get(), "不支持的观测类型");
                return nullptr;
        }

        return wrapper.release();

    } catch (const std::exception& e) {
        SetError(nullptr, std::string("创建观测算子失败: ") + e.what());
        return nullptr;
    }
}

OCEANSIM_API void ObsOp_Destroy(ObservationOperatorHandle handle) {
    if (handle) {
        auto wrapper = static_cast<ObservationOperatorWrapper*>(handle);
        ClearError(handle);
        delete wrapper;
    }
}

OCEANSIM_API int ObsOp_Apply(ObservationOperatorHandle handle,
                             const OceanState* model_state,
                             int state_size,
                             double* observations,
                             int obs_size) {
    try {
        if (!handle || !model_state || !observations || state_size <= 0 || obs_size <= 0) {
            return -1;
        }

        auto wrapper = static_cast<ObservationOperatorWrapper*>(handle);

        // 转换模式状态
        std::vector<OceanStateVector> cpp_state(state_size);
        for (int i = 0; i < state_size; ++i) {
            cpp_state[i] = ConvertStateVector(&model_state[i]);
        }

        // 应用观测算子
        Eigen::VectorXd obs_result = wrapper->operator_->apply(cpp_state);

        if (obs_result.size() != obs_size) {
            SetError(handle, "观测大小不匹配");
            return -2;
        }

        // 复制结果
        for (int i = 0; i < obs_size; ++i) {
            observations[i] = obs_result(i);
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("应用观测算子失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API bool ObsOp_Validate(ObservationOperatorHandle handle,
                                 const ObservationData* observations) {
    try {
        if (!handle || !observations) {
            return false;
        }

        auto wrapper = static_cast<ObservationOperatorWrapper*>(handle);

        Eigen::VectorXd obs_values = Eigen::Map<const Eigen::VectorXd>(
                observations->values, observations->num_observations);

        return wrapper->operator_->validateObservation(obs_values);

    } catch (const std::exception& e) {
        SetError(handle, std::string("验证观测失败: ") + e.what());
        return false;
    }
}

// ==============================================================================
// 局地化和充气接口实现
// ==============================================================================

OCEANSIM_API int EnKF_SetLocalization(EnKFHandle handle,
                                      double radius,
                                      bool enable) {
    try {
        if (!handle || radius <= 0.0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);
        wrapper->config.localization_radius = radius;
        wrapper->config.use_localization = enable;

        // 如果系统已初始化，更新配置
        if (wrapper->is_initialized) {
            // 这里可以添加实时更新配置的逻辑
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置局地化失败: ") + e.what());
        return -2;
    }
}

OCEANSIM_API int EnKF_SetInflation(EnKFHandle handle,
                                   InflationType inflation_type,
                                   double inflation_factor) {
    try {
        if (!handle || inflation_factor < 1.0 || inflation_factor > 3.0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);
        wrapper->config.inflation_factor = inflation_factor;
        wrapper->config.use_inflation = true;

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置充气失败: ") + e.what());
        return -2;
    }
}

OCEANSIM_API int EnKF_ComputeAdaptiveInflation(EnKFHandle handle,
                                               const ObservationData* observations,
                                               double* adaptive_factor) {
    try {
        if (!handle || !observations || !adaptive_factor) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 简化的自适应充气计算
        const auto& ensemble = wrapper->filter->getCurrentEnsemble();
        double ensemble_spread = ensemble.computeEnsembleSpread();
        double obs_variance = 0.0;

        for (int i = 0; i < observations->num_observations; ++i) {
            obs_variance += observations->error_variances[i];
        }
        obs_variance /= observations->num_observations;

        *adaptive_factor = std::max(1.0, std::min(2.0, obs_variance / ensemble_spread));

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("计算自适应充气失败: ") + e.what());
        return -3;
    }
}

// ==============================================================================
// 性能监控接口实现
// ==============================================================================

OCEANSIM_API int EnKF_GetPerformanceMetrics(EnKFHandle handle,
                                            PerformanceMetrics* metrics) {
    try {
        if (!handle || !metrics) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        auto cpp_metrics = wrapper->filter->getPerformanceMetrics();

        metrics->ensemble_spread = cpp_metrics.ensemble_spread;
        metrics->filter_divergence = cpp_metrics.filter_divergence;
        metrics->observation_impact = cpp_metrics.observation_impact;
        metrics->total_cycles = cpp_metrics.total_assimilation_cycles;
        metrics->average_cycle_time_ms = cpp_metrics.average_cycle_time.count();
        metrics->memory_usage_mb = cpp_metrics.memory_usage_mb;
        metrics->analysis_accuracy = 0.0; // 需要与真值比较才能计算

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取性能指标失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API void EnKF_ResetPerformanceStats(EnKFHandle handle) {
    if (handle) {
        auto wrapper = static_cast<EnKFWrapper*>(handle);
        wrapper->cycle_count = 0;
        wrapper->creation_time = std::chrono::steady_clock::now();
    }
}

OCEANSIM_API void EnKF_EnableProfiling(EnKFHandle handle, bool enable) {
    // 实现性能分析启用/禁用逻辑
    if (handle) {
        // 可以在这里添加性能分析的控制逻辑
    }
}

// ==============================================================================
// 验证和诊断接口实现
// ==============================================================================

OCEANSIM_API bool EnKF_ValidateSystemState(EnKFHandle handle) {
    try {
        if (!handle) {
            return false;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            return false;
        }

        return wrapper->filter->validateEnsembleConsistency();

    } catch (const std::exception&) {
        return false;
    }
}

OCEANSIM_API bool EnKF_CheckFilterDivergence(EnKFHandle handle,
                                             double divergence_threshold) {
    try {
        if (!handle || divergence_threshold <= 0.0) {
            return true; // 假设发散
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            return true;
        }

        const auto& ensemble = wrapper->filter->getCurrentEnsemble();
        return EnKFValidator::checkFilterDivergence(ensemble);

    } catch (const std::exception&) {
        return true;
    }
}

OCEANSIM_API int EnKF_ComputeAnalysisIncrement(EnKFHandle handle,
                                               double* increment_norm) {
    try {
        if (!handle || !increment_norm) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        *increment_norm = wrapper->filter->computeAnalysisIncrementNorm();

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("计算分析增量失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API bool EnKF_ValidateLinearGaussian(EnKFHandle handle) {
    try {
        if (!handle) {
            return false;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            return false;
        }

        return EnKFValidator::validateLinearGaussianCase(*wrapper->filter);

    } catch (const std::exception&) {
        return false;
    }
}

OCEANSIM_API bool EnKF_ValidateLorenz96(EnKFHandle handle) {
    try {
        if (!handle) {
            return false;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            return false;
        }

        return EnKFValidator::validateLorenz96System(*wrapper->filter);

    } catch (const std::exception&) {
        return false;
    }
}

// ==============================================================================
// 内存管理接口实现
// ==============================================================================

OCEANSIM_API void EnKF_FreeForecastResult(ForecastResult* result) {
    if (result) {
        delete[] result->ensemble_mean;
        delete[] result->forecast_covariance;
        result->ensemble_mean = nullptr;
        result->forecast_covariance = nullptr;
    }
}

OCEANSIM_API void EnKF_FreeAnalysisResult(AnalysisResult* result) {
    if (result) {
        delete[] result->analysis_mean;
        delete[] result->analysis_covariance;
        delete[] result->kalman_gain;
        result->analysis_mean = nullptr;
        result->analysis_covariance = nullptr;
        result->kalman_gain = nullptr;
    }
}

OCEANSIM_API void EnKF_OptimizeMemory(EnKFHandle handle) {
    // 实现内存优化逻辑
    if (handle) {
        // 可以在这里添加内存压缩、垃圾收集等优化操作
    }
}

// ==============================================================================
// 配置和参数接口实现
// ==============================================================================

OCEANSIM_API void EnKF_GetDefaultConfig(EnKFConfig* config) {
    if (config) {
        config->ensemble_size = 100;              // TOPAZ标准配置
        config->localization_radius = 150000.0;   // 150km
        config->inflation_factor = 1.02;          // 2%充气
        config->regularization_threshold = 1e-10;
        config->use_localization = true;
        config->use_inflation = true;
        config->num_threads = std::thread::hardware_concurrency();
        config->enable_vectorization = true;
    }
}

OCEANSIM_API int EnKF_UpdateConfig(EnKFHandle handle, const EnKFConfig* config) {
    try {
        if (!handle || !config) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);
        wrapper->config = ConvertConfig(config);

        if (!wrapper->config.validate()) {
            SetError(handle, "新配置参数无效");
            return -2;
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("更新配置失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API int EnKF_SaveState(EnKFHandle handle, const char* filename) {
    try {
        if (!handle || !filename) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 简化实现：保存基本状态信息
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            SetError(handle, "无法打开文件");
            return -3;
        }

        // 保存配置
        file.write(reinterpret_cast<const char*>(&wrapper->config), sizeof(wrapper->config));

        // 保存统计信息
        file.write(reinterpret_cast<const char*>(&wrapper->cycle_count), sizeof(wrapper->cycle_count));

        file.close();

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("保存状态失败: ") + e.what());
        return -4;
    }
}

OCEANSIM_API int EnKF_LoadState(EnKFHandle handle, const char* filename) {
    try {
        if (!handle || !filename) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            SetError(handle, "无法打开文件");
            return -2;
        }

        // 加载配置
        file.read(reinterpret_cast<char*>(&wrapper->config), sizeof(wrapper->config));

        // 加载统计信息
        file.read(reinterpret_cast<char*>(&wrapper->cycle_count), sizeof(wrapper->cycle_count));

        file.close();

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("加载状态失败: ") + e.what());
        return -3;
    }
}

// ==============================================================================
// 错误处理接口实现
// ==============================================================================

OCEANSIM_API int EnKF_GetLastError(EnKFHandle handle, char* error_buffer, int buffer_size) {
    try {
        if (!error_buffer || buffer_size <= 0) {
            return -1;
        }

        std::string error_msg = GetError(handle);

        if (error_msg.length() >= static_cast<size_t>(buffer_size)) {
            return static_cast<int>(error_msg.length()) + 1;
        }

        std::strcpy(error_buffer, error_msg.c_str());
        return 0;

    } catch (const std::exception&) {
        return -2;
    }
}

OCEANSIM_API void EnKF_ClearError(EnKFHandle handle) {
    ClearError(handle);
}

// ==============================================================================
// 工具函数接口实现
// ==============================================================================

OCEANSIM_API ObservationData* EnKF_CreateObservationData(int num_observations) {
    if (num_observations <= 0) {
        return nullptr;
    }

    auto obs_data = new ObservationData;
    obs_data->num_observations = num_observations;
    obs_data->values = new double[num_observations];
    obs_data->error_variances = new double[num_observations];
    obs_data->locations_x = new double[num_observations];
    obs_data->locations_y = new double[num_observations];
    obs_data->observation_types = new int[num_observations];
    obs_data->quality_flags = new bool[num_observations];
    obs_data->timestamp = 0;

    // 初始化为默认值
    for (int i = 0; i < num_observations; ++i) {
        obs_data->values[i] = 0.0;
        obs_data->error_variances[i] = 1.0;
        obs_data->locations_x[i] = 0.0;
        obs_data->locations_y[i] = 0.0;
        obs_data->observation_types[i] = OBS_SEA_SURFACE_TEMPERATURE;
        obs_data->quality_flags[i] = true;
    }

    return obs_data;
}

OCEANSIM_API void EnKF_FreeObservationData(ObservationData* obs_data) {
    if (obs_data) {
        delete[] obs_data->values;
        delete[] obs_data->error_variances;
        delete[] obs_data->locations_x;
        delete[] obs_data->locations_y;
        delete[] obs_data->observation_types;
        delete[] obs_data->quality_flags;
        delete obs_data;
    }
}

OCEANSIM_API void EnKF_SetObservation(ObservationData* obs_data,
                                      int index,
                                      double value,
                                      double error_variance,
                                      double x_location,
                                      double y_location,
                                      ObservationType obs_type) {
    if (obs_data && index >= 0 && index < obs_data->num_observations) {
        obs_data->values[index] = value;
        obs_data->error_variances[index] = error_variance;
        obs_data->locations_x[index] = x_location;
        obs_data->locations_y[index] = y_location;
        obs_data->observation_types[index] = obs_type;
        obs_data->quality_flags[index] = true;
    }
}

OCEANSIM_API OceanState* EnKF_CreateStateArray(int state_size) {
    if (state_size <= 0) {
        return nullptr;
    }

    auto state_array = new OceanState[state_size];

    // 初始化为零
    for (int i = 0; i < state_size; ++i) {
        EnKF_InitializeStateZero(&state_array[i]);
    }

    return state_array;
}

OCEANSIM_API void EnKF_FreeStateArray(OceanState* state_array) {
    delete[] state_array;
}

OCEANSIM_API void EnKF_CopyState(OceanState* dest, const OceanState* src) {
    if (dest && src) {
        *dest = *src;
    }
}

OCEANSIM_API void EnKF_InitializeStateZero(OceanState* state) {
    if (state) {
        state->temperature = 0.0;
        state->salinity = 0.0;
        state->velocity_u = 0.0;
        state->velocity_v = 0.0;
        state->velocity_w = 0.0;
        state->sea_ice_concentration = 0.0;
        state->sea_ice_thickness = 0.0;
        state->sea_surface_height = 0.0;
    }
}

OCEANSIM_API void EnKF_ApplyPhysicalConstraints(OceanState* state) {
    if (state) {
        auto cpp_state = ConvertStateVector(state);
        cpp_state.applyPhysicalConstraints();
        ConvertStateVector(cpp_state, state);
    }
}

OCEANSIM_API bool EnKF_IsStatePhysicallyValid(const OceanState* state) {
    if (!state) {
        return false;
    }

    auto cpp_state = ConvertStateVector(state);
    return cpp_state.isPhysicallyValid();
}

// ==============================================================================
// 版本和信息接口实现
// ==============================================================================

OCEANSIM_API void EnKF_GetVersion(char* version_buffer, int buffer_size) {
    if (version_buffer && buffer_size > 0) {
        std::string version = "EnKF预测模块 v1.0.0 (基于TOPAZ系统)";
        if (version.length() < static_cast<size_t>(buffer_size)) {
            std::strcpy(version_buffer, version.c_str());
        }
    }
}

OCEANSIM_API void EnKF_GetBuildInfo(char* build_info_buffer, int buffer_size) {
    if (build_info_buffer && buffer_size > 0) {
        std::string build_info = "编译时间: " __DATE__ " " __TIME__
                                 ", 编译器: "
                                 #ifdef __GNUC__
                                 "GCC " __VERSION__;
#elif defined(_MSC_VER)
        "MSVC " + std::to_string(_MSC_VER);
#else
                                "Unknown";
#endif

        if (build_info.length() < static_cast<size_t>(buffer_size)) {
            std::strcpy(build_info_buffer, build_info.c_str());
        }
    }
}

OCEANSIM_API bool EnKF_CheckHardwareSupport(const char* feature_name) {
    if (!feature_name) {
        return false;
    }

    std::string feature(feature_name);

    if (feature == "openmp") {
#ifdef _OPENMP
        return true;
#else
        return false;
#endif
    } else if (feature == "avx" || feature == "sse") {
        // 简化实现，实际应该检查CPU特性
        return true;
    } else if (feature == "cuda") {
#ifdef ENABLE_CUDA
        return true;
#else
        return false;
#endif
    }

    return false;
} "EnKF系统未初始化");
return -2;
}

// 执行预报步骤
auto cpp_result = wrapper->filter->executeForecastStep(time_step);

if (!cpp_result.success) {
SetError(handle, "预报步骤执行失败");
return -3;
}

// 转换结果
result->state_size = static_cast<int>(cpp_result.ensemble_mean.size());
result->ensemble_mean = new OceanState[result->state_size];

for (int i = 0; i < result->state_size; ++i) {
ConvertStateVector(cpp_result.ensemble_mean[i], &result->ensemble_mean[i]);
}

// 复制协方差矩阵（展平）
int covariance_size = cpp_result.forecast_covariance.rows() * cpp_result.forecast_covariance.cols();
result->forecast_covariance = new double[covariance_size];
Eigen::Map<Eigen::MatrixXd>(result->forecast_covariance,
cpp_result.forecast_covariance.rows(),
        cpp_result.forecast_covariance.cols()) = cpp_result.forecast_covariance;

result->ensemble_spread = wrapper->filter->getCurrentEnsemble().computeEnsembleSpread();
result->computation_time_ms = cpp_result.computation_time.count();
result->success = true;

ClearError(handle);
return 0;

} catch (const std::exception& e) {
SetError(handle, std::string("预报执行失败: ") + e.what());
return -4;
}
}

OCEANSIM_API int EnKF_ExecuteAnalysis(EnKFHandle handle,
                                      const ObservationData* observations,
                                      AnalysisResult* result) {
    try {
        if (!handle || !observations || !result) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 转换观测数据
        ObservationData cpp_obs;
        cpp_obs.values = Eigen::Map<const Eigen::VectorXd>(observations->values, observations->num_observations);

        // 构建观测误差协方差矩阵
        cpp_obs.error_covariance = Eigen::MatrixXd::Zero(observations->num_observations, observations->num_observations);
        for (int i = 0; i < observations->num_observations; ++i) {
            cpp_obs.error_covariance(i, i) = observations->error_variances[i];
        }

        // 创建默认的海表温度观测算子（简化实现）
        cpp_obs.operator_ = std::make_shared<SeaSurfaceTemperatureOperator>(wrapper->grid);

        // 设置时间戳
        cpp_obs.timestamp = std::chrono::system_clock::from_time_t(observations->timestamp);

        // 执行分析步骤
        auto cpp_result = wrapper->filter->executeAnalysisStep(cpp_obs);

        if (!cpp_result.success) {
            SetError(handle, "分析步骤执行失败");
            return -3;
        }

        // 转换结果
        result->state_size = static_cast<int>(cpp_result.analysis_mean.size());
        result->obs_size = observations->num_observations;

        result->analysis_mean = new OceanState[result->state_size];
        for (int i = 0; i < result->state_size; ++i) {
            ConvertStateVector(cpp_result.analysis_mean[i], &result->analysis_mean[i]);
        }

        // 复制分析协方差矩阵
        int covariance_size = cpp_result.analysis_covariance.rows() * cpp_result.analysis_covariance.cols();
        result->analysis_covariance = new double[covariance_size];
        Eigen::Map<Eigen::MatrixXd>(result->analysis_covariance,
                                    cpp_result.analysis_covariance.rows(),
                                    cpp_result.analysis_covariance.cols()) = cpp_result.analysis_covariance;

        // 复制卡尔曼增益矩阵
        int gain_size = cpp_result.kalman_gain.rows() * cpp_result.kalman_gain.cols();
        result->kalman_gain = new double[gain_size];
        Eigen::Map<Eigen::MatrixXd>(result->kalman_gain,
                                    cpp_result.kalman_gain.rows(),
                                    cpp_result.kalman_gain.cols()) = cpp_result.kalman_gain;

        result->innovation_variance = cpp_result.innovation_variance;
        result->observation_impact = cpp_result.innovation_variance; // 简化
        result->computation_time_ms = cpp_result.computation_time.count();
        result->success = true;

        // 更新循环计数
        wrapper->cycle_count++;

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("分析执行失败: ") + e.what());
        return -4;
    }
}

OCEANSIM_API int EnKF_ExecuteAssimilationCycle(EnKFHandle handle,
                                               double time_step,
                                               const ObservationData* observations,
                                               ForecastResult* forecast_result,
                                               AnalysisResult* analysis_result) {
    try {
        // 执行预报步骤
        int forecast_status = EnKF_ExecuteForecast(handle, time_step, forecast_result);
        if (forecast_status != 0) {
            return forecast_status;
        }

        // 执行分析步骤
        int analysis_status = EnKF_ExecuteAnalysis(handle, observations, analysis_result);
        if (analysis_status != 0) {
            return analysis_status;
        }

        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("同化循环执行失败: ") + e.what());
        return -5;
    }
}

// ==============================================================================
// 状态访问接口实现
// ==============================================================================

OCEANSIM_API int EnKF_GetCurrentMean(EnKFHandle handle,
                                     OceanState* mean_state,
                                     int state_size) {
    try {
        if (!handle || !mean_state || state_size <= 0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        auto cpp_mean = wrapper->filter->getCurrentMean();

        if (cpp_mean.size() != static_cast<size_t>(state_size)) {
            SetError(handle, "状态大小不匹配");
            return -3;
        }

        for (int i = 0; i < state_size; ++i) {
            ConvertStateVector(cpp_mean[i], &mean_state[i]);
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取当前均值失败: ") + e.what());
        return -4;
    }
}

OCEANSIM_API int EnKF_GetCurrentCovariance(EnKFHandle handle,
                                           double* covariance,
                                           int matrix_size) {
    try {
        if (!handle || !covariance || matrix_size <= 0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }
        

OCEANSIM_API int EnKF_SetCustomIntegrator(EnKFHandle handle,
                                          OceanModelIntegrator integrator_func,
                                          void* user_data) {
    try {
        if (!handle || !integrator_func) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 在实际实现中，需要将自定义积分器集成到EnKF系统中
        // 这里提供接口框架，具体实现需要扩展EnsembleKalmanFilter类

        SetError(handle, "自定义积分器功能需要进一步实现");
        return -2; // 暂未完全实现

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置自定义积分器失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API int EnKF_SetCustomObservationOperator(EnKFHandle handle,
                                                   CustomObservationOperator obs_operator_func,
                                                   void* user_data) {
    try {
        if (!handle || !obs_operator_func) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 类似地，自定义观测算子需要扩展现有的观测算子系统

        SetError(handle, "自定义观测算子功能需要进一步实现");
        return -2; // 暂未完全实现

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置自定义观测算子失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API void EnKF_SetParallelExecution(EnKFHandle handle, bool enable, int num_threads) {
    try {
        if (!handle) {
            return;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (num_threads <= 0) {
            num_threads = std::thread::hardware_concurrency();
        }

        wrapper->config.num_threads = num_threads;

        // 在实际系统中，这里需要重新配置并行执行引擎
        // 可能需要重新创建线程池等资源

        ClearError(handle);

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置并行执行失败: ") + e.what());
    }
}

OCEANSIM_API int EnKF_SetStabilityParameters(EnKFHandle handle,
                                             double regularization_threshold,
                                             double condition_number_threshold) {
    try {
        if (!handle || regularization_threshold <= 0.0 || condition_number_threshold <= 0.0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);
        wrapper->config.regularization_threshold = regularization_threshold;

        // condition_number_threshold 需要在配置结构中添加相应字段
        // 这里演示如何处理这类扩展配置

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置稳定性参数失败: ") + e.what());
        return -2;
    }
}

// ==============================================================================
// 调试和日志接口实现
// ==============================================================================

OCEANSIM_API void EnKF_SetLogLevel(EnKFHandle handle, int log_level) {
    try {
        if (!handle || log_level < 0 || log_level > 3) {
            return;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 设置日志级别
        // 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG
        std::string level_name;
        switch (log_level) {
            case 0: level_name = "ERROR"; break;
            case 1: level_name = "WARN"; break;
            case 2: level_name = "INFO"; break;
            case 3: level_name = "DEBUG"; break;
        }

        // 在实际实现中，这里需要配置底层的日志系统
        // 例如设置spdlog或其他日志库的级别

    } catch (const std::exception&) {
        // 日志设置失败不应该影响系统运行
    }
}

OCEANSIM_API void EnKF_EnableDebugOutput(EnKFHandle handle, bool enable) {
    try {
        if (!handle) {
            return;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 启用或禁用详细调试输出
        // 在实际实现中，这可能涉及设置多个内部标志

    } catch (const std::exception&) {
        // 调试设置失败不应该影响系统运行
    }
}

OCEANSIM_API int EnKF_ExportEnsemble(EnKFHandle handle,
                                     const char* filename,
                                     const char* format) {
    try {
        if (!handle || !filename || !format) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        std::string format_str(format);
        std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::tolower);

        if (format_str == "binary") {
            // 导出为二进制格式
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                SetError(handle, "无法创建输出文件");
                return -3;
            }

            const auto& ensemble = wrapper->filter->getCurrentEnsemble();
            int ensemble_size = ensemble.getEnsembleSize();
            int state_size = static_cast<int>(ensemble.getStateSize());

            // 写入头信息
            file.write(reinterpret_cast<const char*>(&ensemble_size), sizeof(ensemble_size));
            file.write(reinterpret_cast<const char*>(&state_size), sizeof(state_size));

            // 写入集合数据
            for (int i = 0; i < ensemble_size; ++i) {
                const OceanStateVector* member = ensemble.getMember(i);
                for (int j = 0; j < state_size; ++j) {
                    file.write(reinterpret_cast<const char*>(&member[j]), sizeof(OceanStateVector));
                }
            }

            file.close();

        } else if (format_str == "netcdf") {
            // NetCDF格式导出（需要NetCDF库支持）
            SetError(handle, "NetCDF格式导出需要额外的库支持");
            return -4;

        } else if (format_str == "hdf5") {
            // HDF5格式导出（需要HDF5库支持）
            SetError(handle, "HDF5格式导出需要额外的库支持");
            return -5;

        } else {
            SetError(handle, "不支持的文件格式");
            return -6;
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("导出集合失败: ") + e.what());
        return -7;
    }
}

OCEANSIM_API int EnKF_ExportCovariance(EnKFHandle handle, const char* filename) {
    try {
        if (!handle || !filename) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        auto covariance = wrapper->filter->getCurrentCovariance();

        // 导出为二进制格式
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            SetError(handle, "无法创建输出文件");
            return -3;
        }

        // 写入矩阵维度
        int rows = static_cast<int>(covariance.rows());
        int cols = static_cast<int>(covariance.cols());
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        // 写入矩阵数据
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double value = covariance(i, j);
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }

        file.close();

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("导出协方差失败: ") + e.what());
        return -4;
    }
}

// ==============================================================================
// 高级工具函数接口实现补充
// ==============================================================================

OCEANSIM_API int EnKF_SetCustomInflationStrategy(EnKFHandle handle,
                                                 double (*inflation_func)(double current_spread, double target_spread, void* user_data),
                                                 void* user_data) {
    try {
        if (!handle || !inflation_func) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 在实际实现中，需要扩展InflationAlgorithm类以支持自定义函数
        // 这里提供接口框架

        SetError(handle, "自定义充气策略功能需要进一步实现");
        return -2; // 暂未完全实现

    } catch (const std::exception& e) {
        SetError(handle, std::string("设置自定义充气策略失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API int EnKF_ComputeFilterQuality(EnKFHandle handle,
                                           const ObservationData* truth_data,
                                           double* rmse,
                                           double* correlation,
                                           double* bias) {
    try {
        if (!handle || !truth_data || !rmse || !correlation || !bias) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 计算滤波器质量指标
        auto current_mean = wrapper->filter->getCurrentMean();

        // 简化实现：计算与真值的统计指标
        double sum_squared_error = 0.0;
        double sum_truth = 0.0;
        double sum_forecast = 0.0;
        double sum_truth_squared = 0.0;
        double sum_forecast_squared = 0.0;
        double sum_product = 0.0;

        int valid_points = 0;

        // 这里需要将ObservationData中的真值与current_mean进行比较
        // 简化实现，假设truth_data包含了对应位置的真值

        for (int i = 0; i < truth_data->num_observations && i < static_cast<int>(current_mean.size()); ++i) {
            double truth_value = truth_data->values[i];
            double forecast_value = current_mean[i].temperature; // 简化：只比较温度

            double error = forecast_value - truth_value;
            sum_squared_error += error * error;

            sum_truth += truth_value;
            sum_forecast += forecast_value;
            sum_truth_squared += truth_value * truth_value;
            sum_forecast_squared += forecast_value * forecast_value;
            sum_product += truth_value * forecast_value;

            valid_points++;
        }

        if (valid_points == 0) {
            SetError(handle, "没有有效的比较点");
            return -3;
        }

        // 计算RMSE
        *rmse = std::sqrt(sum_squared_error / valid_points);

        // 计算相关系数
        double mean_truth = sum_truth / valid_points;
        double mean_forecast = sum_forecast / valid_points;

        double numerator = sum_product - valid_points * mean_truth * mean_forecast;
        double denominator = std::sqrt((sum_truth_squared - valid_points * mean_truth * mean_truth) *
                                       (sum_forecast_squared - valid_points * mean_forecast * mean_forecast));

        *correlation = (denominator > 1e-10) ? numerator / denominator : 0.0;

        // 计算偏差
        *bias = mean_forecast - mean_truth;

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("计算滤波器质量失败: ") + e.what());
        return -4;
    }
}

OCEANSIM_API int EnKF_GetEnsembleVariability(EnKFHandle handle,
                                             double* temporal_variance,
                                             double* spatial_variance,
                                             double* ensemble_variance) {
    try {
        if (!handle || !temporal_variance || !spatial_variance || !ensemble_variance) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        const auto& ensemble = wrapper->filter->getCurrentEnsemble();

        // 计算集合方差
        *ensemble_variance = ensemble.computeEnsembleSpread();

        // 计算空间方差（简化实现）
        auto current_mean = ensemble.computeEnsembleMean();
        double spatial_var = 0.0;
        double mean_temp = 0.0;

        // 计算温度场的空间均值
        for (const auto& state : current_mean) {
            mean_temp += state.temperature;
        }
        mean_temp /= current_mean.size();

        // 计算空间方差
        for (const auto& state : current_mean) {
            double deviation = state.temperature - mean_temp;
            spatial_var += deviation * deviation;
        }
        *spatial_variance = spatial_var / current_mean.size();

        // 时间方差需要历史数据，这里简化为当前集合的时间变化
        // 在实际实现中，需要维护时间序列数据
        *temporal_variance = *ensemble_variance; // 简化

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("计算集合变异性失败: ") + e.what());
        return -3;
    }
}

// ==============================================================================
// 实时运行和调度接口实现
// ==============================================================================

OCEANSIM_API int EnKF_ScheduleAssimilationCycle(EnKFHandle handle,
                                                double start_time,
                                                double time_step,
                                                int num_cycles,
                                                void (*callback)(int cycle, double time, void* user_data),
                                                void* user_data) {
    try {
        if (!handle || time_step <= 0.0 || num_cycles <= 0) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 在实际实现中，这里应该创建一个调度器来管理定时执行
        // 这里提供同步执行的简化版本

        double current_time = start_time;

        for (int cycle = 0; cycle < num_cycles; ++cycle) {
            // 执行预报步骤
            ForecastResult forecast_result;
            int forecast_status = EnKF_ExecuteForecast(handle, time_step, &forecast_result);

            if (forecast_status != 0) {
                SetError(handle, "调度执行中预报步骤失败");
                return -3;
            }

            // 释放预报结果内存
            EnKF_FreeForecastResult(&forecast_result);

            // 调用回调函数
            if (callback) {
                callback(cycle, current_time, user_data);
            }

            current_time += time_step;

            // 更新循环计数
            wrapper->cycle_count++;
        }

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("调度同化循环失败: ") + e.what());
        return -4;
    }
}

// ==============================================================================
// 扩展诊断接口实现
// ==============================================================================

OCEANSIM_API int EnKF_ComputeObservationImpact(EnKFHandle handle,
                                               const ObservationData* observations,
                                               double* impact_magnitude,
                                               double* impact_efficiency) {
    try {
        if (!handle || !observations || !impact_magnitude || !impact_efficiency) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        // 计算观测影响
        // 这需要比较同化前后的状态差异

        // 保存当前状态
        auto current_mean_before = wrapper->filter->getCurrentMean();
        auto current_spread_before = wrapper->filter->getCurrentEnsemble().computeEnsembleSpread();

        // 执行分析步骤
        AnalysisResult analysis_result;
        int analysis_status = EnKF_ExecuteAnalysis(handle, observations, &analysis_result);

        if (analysis_status != 0) {
            SetError(handle, "计算观测影响时分析步骤失败");
            return -3;
        }

        // 计算状态变化
        auto current_mean_after = wrapper->filter->getCurrentMean();
        auto current_spread_after = wrapper->filter->getCurrentEnsemble().computeEnsembleSpread();

        // 计算影响大小
        double total_change = 0.0;
        for (size_t i = 0; i < current_mean_before.size() && i < current_mean_after.size(); ++i) {
            auto state_change = current_mean_after[i] - current_mean_before[i];
            total_change += state_change.temperature * state_change.temperature +
                            state_change.salinity * state_change.salinity +
                            state_change.velocity.squaredNorm();
        }

        *impact_magnitude = std::sqrt(total_change / current_mean_before.size());

        // 计算影响效率（集合离散度的相对变化）
        double spread_change = std::abs(current_spread_after - current_spread_before);
        *impact_efficiency = spread_change / (current_spread_before + 1e-10);

        // 释放分析结果内存
        EnKF_FreeAnalysisResult(&analysis_result);

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("计算观测影响失败: ") + e.what());
        return -4;
    }
}

// ==============================================================================
// 内存和资源管理的额外实现
// ==============================================================================

OCEANSIM_API int EnKF_GetMemoryUsage(EnKFHandle handle,
                                     size_t* ensemble_memory,
                                     size_t* covariance_memory,
                                     size_t* total_memory) {
    try {
        if (!handle || !ensemble_memory || !covariance_memory || !total_memory) {
            return -1;
        }

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        if (!wrapper->is_initialized) {
            SetError(handle, "EnKF系统未初始化");
            return -2;
        }

        const auto& ensemble = wrapper->filter->getCurrentEnsemble();
        auto covariance = wrapper->filter->getCurrentCovariance();

        // 计算集合内存使用
        *ensemble_memory = ensemble.getEnsembleSize() * ensemble.getStateSize() * sizeof(OceanStateVector);

        // 计算协方差矩阵内存使用
        *covariance_memory = covariance.rows() * covariance.cols() * sizeof(double);

        // 估算总内存使用（包括其他数据结构）
        *total_memory = *ensemble_memory + *covariance_memory +
                        sizeof(EnKFWrapper) + 1024 * 1024; // 额外1MB估算

        ClearError(handle);
        return 0;

    } catch (const std::exception& e) {
        SetError(handle, std::string("获取内存使用情况失败: ") + e.what());
        return -3;
    }
}

OCEANSIM_API void EnKF_ForceGarbageCollection(EnKFHandle handle) {
    try {
        if (!handle) {
            return;
        }

        // 在C++中没有垃圾收集器，但可以执行内存整理操作
        // 这里可以实现内存压缩、缓存清理等操作

        auto wrapper = static_cast<EnKFWrapper*>(handle);

        // 强制清理可能的内存碎片
        // 在实际实现中，可能涉及重新分配大的数据结构

    } catch (const std::exception&) {
        // 内存整理失败不应该影响系统运行
    }
}

// ==============================================================================
// 全局清理函数
// ==============================================================================

OCEANSIM_API void EnKF_GlobalCleanup() {
    // 清理全局资源
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_error_messages.clear();

    // 在这里可以添加其他全局资源的清理
    // 例如线程池、内存池等
}

OCEANSIM_API void EnKF_GlobalInitialize() {
    // 全局初始化
    // 在这里可以初始化全局资源
    // 例如数学库、并行计算环境等

    // 初始化Eigen
    Eigen::initParallel();

    // 设置线程数
    Eigen::setNbThreads(std::thread::hardware_concurrency());
}