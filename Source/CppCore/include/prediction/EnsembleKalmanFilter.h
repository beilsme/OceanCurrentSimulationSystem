// ==============================================================================
// 文件：Source/CppCore/include/prediction/EnsembleKalmanFilter.h
// 作者：beilsm
// 版本：v1.0.0
// 创建时间：2025-07-03
// 功能：基于TOPAZ系统的集合卡尔曼滤波预测模块核心实现
// 说明：实现100个集合成员配置、流依赖误差协方差传播和多源观测数据同化
// ==============================================================================

#ifndef ENSEMBLE_KALMAN_FILTER_H
#define ENSEMBLE_KALMAN_FILTER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <chrono>
#include "data/GridDataStructure.h"
#include "algorithms/RungeKuttaSolver.h"
#include "algorithms/ParallelComputeEngine.h"
#include "algorithms/VectorizedOperations.h"
#include "core/AdvectionDiffusionSolver.h"
#include "utils/Logger.h"

namespace OceanSim {
    namespace Prediction {

/**
 * @brief 海洋状态向量结构
 * 包含TOPAZ系统处理的主要海洋变量
 */
        struct OceanStateVector {
            double temperature;           // 温度 (°C)
            double salinity;             // 盐度 (PSU)
            Eigen::Vector3d velocity;    // 流速分量 (m/s)
            double sea_ice_concentration; // 海冰浓度 (0-1)
            double sea_ice_thickness;    // 海冰厚度 (m)
            double sea_surface_height;   // 海表面高度 (m)

            // 物理约束检查
            bool isPhysicallyValid() const;
            void applyPhysicalConstraints();

            // 运算符重载
            OceanStateVector operator+(const OceanStateVector& other) const;
            OceanStateVector operator-(const OceanStateVector& other) const;
            OceanStateVector operator*(double scalar) const;
        };

/**
 * @brief 集合成员容器
 * 高效管理100个集合成员的内存和计算
 */
        class EnsembleCollection {
        public:
            static constexpr int TOPAZ_ENSEMBLE_SIZE = 100;

            EnsembleCollection(size_t state_size);
            ~EnsembleCollection();

            // 集合访问接口
            OceanStateVector* getMember(int index);
            const OceanStateVector* getMember(int index) const;

            // 统计计算
            std::vector<OceanStateVector> computeEnsembleMean() const;
            Eigen::MatrixXd computeSampleCovariance() const;
            double computeEnsembleSpread() const;

            // 并行操作
            void parallelApply(std::function<void(OceanStateVector*, int)> operation);

            // 内存管理
            void resize(size_t new_state_size);
            size_t getStateSize() const { return state_size_; }
            int getEnsembleSize() const { return TOPAZ_ENSEMBLE_SIZE; }

        private:
            std::vector<std::unique_ptr<OceanStateVector[]>> members_;
            size_t state_size_;
            std::unique_ptr<Utils::ThreadPool> thread_pool_;
        };

/**
 * @brief 观测算子基类
 * 实现从模式状态到观测量的映射
 */
        class ObservationOperator {
        public:
            virtual ~ObservationOperator() = default;

            // 核心观测算子方法
            virtual Eigen::VectorXd apply(const std::vector<OceanStateVector>& model_state) = 0;
            virtual Eigen::MatrixXd computeJacobian(const std::vector<OceanStateVector>& model_state) = 0;

            // 观测类型识别
            virtual std::string getObservationType() const = 0;
            virtual bool validateObservation(const Eigen::VectorXd& observation) = 0;
        };

/**
 * @brief 海平面高度异常观测算子
 * 对应TOPAZ系统的DUACS产品
 */
        class SeaLevelAnomalyOperator : public ObservationOperator {
        public:
            SeaLevelAnomalyOperator(std::shared_ptr<Data::GridDataStructure> grid);

            Eigen::VectorXd apply(const std::vector<OceanStateVector>& model_state) override;
            Eigen::MatrixXd computeJacobian(const std::vector<OceanStateVector>& model_state) override;
            std::string getObservationType() const override { return "SeaLevelAnomaly"; }
            bool validateObservation(const Eigen::VectorXd& observation) override;

        private:
            std::shared_ptr<Data::GridDataStructure> grid_;
            Eigen::VectorXd reference_sea_level_;
        };

/**
 * @brief 海表温度观测算子
 * 对应TOPAZ系统的Reynolds数据
 */
        class SeaSurfaceTemperatureOperator : public ObservationOperator {
        public:
            SeaSurfaceTemperatureOperator(std::shared_ptr<Data::GridDataStructure> grid);

            Eigen::VectorXd apply(const std::vector<OceanStateVector>& model_state) override;
            Eigen::MatrixXd computeJacobian(const std::vector<OceanStateVector>& model_state) override;
            std::string getObservationType() const override { return "SeaSurfaceTemperature"; }
            bool validateObservation(const Eigen::VectorXd& observation) override;

        private:
            std::shared_ptr<Data::GridDataStructure> grid_;
            double min_valid_temp_ = -2.0;  // 海水冰点
            double max_valid_temp_ = 35.0;  // 最大海表温度
        };

/**
 * @brief 海冰浓度观测算子
 * 对应TOPAZ系统的SSM/I数据
 */
        class SeaIceConcentrationOperator : public ObservationOperator {
        public:
            SeaIceConcentrationOperator(std::shared_ptr<Data::GridDataStructure> grid);

            Eigen::VectorXd apply(const std::vector<OceanStateVector>& model_state) override;
            Eigen::MatrixXd computeJacobian(const std::vector<OceanStateVector>& model_state) override;
            std::string getObservationType() const override { return "SeaIceConcentration"; }
            bool validateObservation(const Eigen::VectorXd& observation) override;

        private:
            std::shared_ptr<Data::GridDataStructure> grid_;
        };

/**
 * @brief 观测数据容器
 */
        struct ObservationData {
            Eigen::VectorXd values;                           // 观测值
            Eigen::MatrixXd error_covariance;                // 观测误差协方差
            std::shared_ptr<ObservationOperator> operator_;   // 观测算子
            std::chrono::system_clock::time_point timestamp;  // 观测时间
            std::vector<Eigen::Vector2d> locations;          // 观测位置

            // 质量控制标记
            std::vector<bool> quality_flags;

            // 扰动观测生成（EnKF所需）
            std::vector<Eigen::VectorXd> generatePerturbedObservations(int ensemble_size) const;
        };

/**
 * @brief Gaspari-Cohn局地化算子
 * 实现TOPAZ系统的协方差局地化技术
 */
        class GaspariCohnLocalization {
        public:
            GaspariCohnLocalization(double localization_radius);

            // 局地化函数计算
            double computeCorrelationFunction(double distance) const;

            // 协方差矩阵局地化
            void applyLocalization(Eigen::MatrixXd& covariance_matrix,
                                   const std::shared_ptr<Data::GridDataStructure>& grid) const;

            // Schur乘积实现
            void applySchurProduct(Eigen::MatrixXd& matrix,
                                   const Eigen::MatrixXd& localization_matrix) const;

            // 设置局地化半径
            void setLocalizationRadius(double radius) { localization_radius_ = radius; }
            double getLocalizationRadius() const { return localization_radius_; }

        private:
            double localization_radius_;  // 局地化半径 (m)

            // Gaspari-Cohn五次多项式实现
            double gaspariCohnPolynomial(double normalized_distance) const;
        };

/**
 * @brief 充气算法类
 * 实现自适应充气以维护集合离散度
 */
        class InflationAlgorithm {
        public:
            enum class InflationType {
                Multiplicative,    // 乘性充气
                Additive,         // 加性充气  
                Adaptive,         // 自适应充气
                Relaxation        // 松弛充气
            };

            InflationAlgorithm(InflationType type, double base_factor = 1.02);

            // 充气操作
            void applyInflation(EnsembleCollection& ensemble,
                                const std::vector<OceanStateVector>& ensemble_mean);

            // 自适应充气因子计算
            double computeAdaptiveInflationFactor(const EnsembleCollection& ensemble,
                                                  const ObservationData& observations);

            // 充气参数设置
            void setInflationFactor(double factor) { inflation_factor_ = factor; }
            void setInflationType(InflationType type) { inflation_type_ = type; }

        private:
            InflationType inflation_type_;
            double inflation_factor_;
            double min_inflation_factor_ = 1.0;
            double max_inflation_factor_ = 2.0;

            // 内部充气方法
            void applyMultiplicativeInflation(EnsembleCollection& ensemble,
                                              const std::vector<OceanStateVector>& mean);
            void applyAdditiveInflation(EnsembleCollection& ensemble);
        };

/**
 * @brief EnKF预测引擎配置
 */
        struct EnKFConfiguration {
            // 基本配置
            int ensemble_size = 100;                    // TOPAZ标准配置
            double localization_radius = 150000.0;     // 150km局地化半径
            double inflation_factor = 1.02;            // 默认充气因子

            // 数值稳定性参数
            double regularization_threshold = 1e-10;   // 正则化阈值
            bool use_localization = true;              // 启用局地化
            bool use_inflation = true;                 // 启用充气

            // 并行计算配置
            int num_threads = std::thread::hardware_concurrency();
            bool enable_vectorization = true;

            // 验证参数
            bool validate() const;
        };

/**
 * @brief 集合卡尔曼滤波主类
 * 实现完整的TOPAZ系统EnKF算法
 */
        class EnsembleKalmanFilter {
        public:
            EnsembleKalmanFilter(const EnKFConfiguration& config,
                                 std::shared_ptr<Data::GridDataStructure> grid);
            ~EnsembleKalmanFilter();

            // 初始化
            bool initialize(const std::vector<OceanStateVector>& initial_state,
                            const Eigen::MatrixXd& background_error_covariance);

            // EnKF主循环
            struct ForecastResult {
                std::unique_ptr<EnsembleCollection> forecast_ensemble;
                std::vector<OceanStateVector> ensemble_mean;
                Eigen::MatrixXd forecast_covariance;
                std::chrono::milliseconds computation_time;
                bool success;
            };

            struct AnalysisResult {
                std::unique_ptr<EnsembleCollection> analysis_ensemble;
                std::vector<OceanStateVector> analysis_mean;
                Eigen::MatrixXd analysis_covariance;
                Eigen::MatrixXd kalman_gain;
                double innovation_variance;
                std::chrono::milliseconds computation_time;
                bool success;
            };

            // 预报步骤
            std::future<ForecastResult> executeforecastStepAsync(double time_step);
            ForecastResult executeForecastStep(double time_step);

            // 分析步骤
            std::future<AnalysisResult> executeAnalysisStepAsync(const ObservationData& observations);
            AnalysisResult executeAnalysisStep(const ObservationData& observations);

            // 完整同化循环
            struct AssimilationResult {
                ForecastResult forecast;
                AnalysisResult analysis;
                bool cycle_success;
            };

            std::future<AssimilationResult> executeAssimilationCycleAsync(
                    double time_step, const ObservationData& observations);

            // 状态访问
            const EnsembleCollection& getCurrentEnsemble() const { return *current_ensemble_; }
            std::vector<OceanStateVector> getCurrentMean() const;
            Eigen::MatrixXd getCurrentCovariance() const;

            // 统计信息
            struct PerformanceMetrics {
                double ensemble_spread;
                double filter_divergence;
                double observation_impact;
                size_t total_assimilation_cycles;
                std::chrono::milliseconds average_cycle_time;
                double memory_usage_mb;
            };

            PerformanceMetrics getPerformanceMetrics() const;

            // 验证和诊断
            bool validateEnsembleConsistency() const;
            double computeAnalysisIncrementNorm() const;

        private:
            // 配置和网格
            EnKFConfiguration config_;
            std::shared_ptr<Data::GridDataStructure> grid_;

            // 集合数据
            std::unique_ptr<EnsembleCollection> current_ensemble_;
            std::vector<OceanStateVector> current_mean_;
            Eigen::MatrixXd current_covariance_;

            // 算法组件
            std::unique_ptr<GaspariCohnLocalization> localization_;
            std::unique_ptr<InflationAlgorithm> inflation_;
            std::unique_ptr<Algorithms::RungeKuttaSolver> rk_solver_;
            std::unique_ptr<Algorithms::ParallelComputeEngine> parallel_engine_;
            std::unique_ptr<Algorithms::VectorizedOperations> vector_ops_;

            // 性能监控
            mutable std::mutex metrics_mutex_;
            mutable PerformanceMetrics performance_metrics_;

            // 内部方法

            // 集合生成
            void generateInitialEnsemble(const std::vector<OceanStateVector>& mean_state,
                                         const Eigen::MatrixXd& covariance);

            // 预报积分
            void integrateEnsembleMember(OceanStateVector* member, double time_step, int member_id);
            std::vector<std::future<void>> setupParallelForecast(double time_step);

            // 分析计算
            Eigen::MatrixXd computeKalmanGain(const Eigen::MatrixXd& forecast_covariance,
                                              const ObservationData& observations);
            void updateEnsembleMembers(const ObservationData& observations,
                                       const Eigen::MatrixXd& kalman_gain);

            // 数值稳定性
            Eigen::MatrixXd regularizeMatrix(const Eigen::MatrixXd& matrix) const;
            bool checkMatrixCondition(const Eigen::MatrixXd& matrix) const;

            // 统计计算
            void updateStatistics();
            void updatePerformanceMetrics(std::chrono::milliseconds execution_time);

            // 内存管理
            void optimizeMemoryUsage();

            // 日志记录
            std::shared_ptr<Utils::Logger> logger_;
            void logAssimilationCycle(const AssimilationResult& result);
        };

/**
 * @brief EnKF结果验证器
 * 提供算法正确性验证功能
 */
        class EnKFValidator {
        public:
            static bool validateLinearGaussianCase(const EnsembleKalmanFilter& filter);
            static bool validateLorenz96System(const EnsembleKalmanFilter& filter);
            static double computeAnalysisAccuracy(const AnalysisResult& result,
                                                  const std::vector<OceanStateVector>& truth);

            // 数值稳定性检查
            static bool checkFilterDivergence(const EnsembleCollection& ensemble);
            static bool checkCovariancePositiveDefinite(const Eigen::MatrixXd& covariance);
        };

    } // namespace Prediction
} // namespace OceanSim

#endif // ENSEMBLE_KALMAN_FILTER_H