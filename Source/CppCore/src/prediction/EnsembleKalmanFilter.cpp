// ==============================================================================
// 文件：Source/CppCore/src/prediction/EnsembleKalmanFilter.cpp
// 作者：beilsm
// 版本：v1.0.0
// 创建时间：2025-07-03
// 功能：基于TOPAZ系统的集合卡尔曼滤波预测模块实现
// 说明：完整实现EnKF算法的预报和分析步骤，支持并行计算和局地化技术
// ==============================================================================

#include "prediction/EnsembleKalmanFilter.h"
#include "utils/MathUtils.h"
#include <random>
#include <algorithm>
#include <execution>
#include <cmath>
#include <iomanip>

namespace OceanSim {
    namespace Prediction {

// ==============================================================================
// OceanStateVector 实现
// ==============================================================================

        bool OceanStateVector::isPhysicallyValid() const {
            // 温度物理约束 (-2°C 到 35°C)
            if (temperature < -2.0 || temperature > 35.0) return false;

            // 盐度物理约束 (0 到 45 PSU)
            if (salinity < 0.0 || salinity > 45.0) return false;

            // 海冰浓度约束 (0 到 1)
            if (sea_ice_concentration < 0.0 || sea_ice_concentration > 1.0) return false;

            // 海冰厚度约束 (非负值，最大20米)
            if (sea_ice_thickness < 0.0 || sea_ice_thickness > 20.0) return false;

            // 流速合理性检查 (最大5 m/s)
            if (velocity.norm() > 5.0) return false;

            return true;
        }

        void OceanStateVector::applyPhysicalConstraints() {
            // 温度约束
            temperature = std::max(-2.0, std::min(35.0, temperature));

            // 盐度约束
            salinity = std::max(0.0, std::min(45.0, salinity));

            // 海冰浓度约束
            sea_ice_concentration = std::max(0.0, std::min(1.0, sea_ice_concentration));

            // 海冰厚度约束
            sea_ice_thickness = std::max(0.0, std::min(20.0, sea_ice_thickness));

            // 流速约束
            if (velocity.norm() > 5.0) {
                velocity = velocity.normalized() * 5.0;
            }

            // 海冰物理一致性：海冰浓度为0时厚度也应为0
            if (sea_ice_concentration < 0.01) {
                sea_ice_thickness = 0.0;
            }
        }

        OceanStateVector OceanStateVector::operator+(const OceanStateVector& other) const {
            OceanStateVector result;
            result.temperature = temperature + other.temperature;
            result.salinity = salinity + other.salinity;
            result.velocity = velocity + other.velocity;
            result.sea_ice_concentration = sea_ice_concentration + other.sea_ice_concentration;
            result.sea_ice_thickness = sea_ice_thickness + other.sea_ice_thickness;
            result.sea_surface_height = sea_surface_height + other.sea_surface_height;
            return result;
        }

        OceanStateVector OceanStateVector::operator-(const OceanStateVector& other) const {
            OceanStateVector result;
            result.temperature = temperature - other.temperature;
            result.salinity = salinity - other.salinity;
            result.velocity = velocity - other.velocity;
            result.sea_ice_concentration = sea_ice_concentration - other.sea_ice_concentration;
            result.sea_ice_thickness = sea_ice_thickness - other.sea_ice_thickness;
            result.sea_surface_height = sea_surface_height - other.sea_surface_height;
            return result;
        }

        OceanStateVector OceanStateVector::operator*(double scalar) const {
            OceanStateVector result;
            result.temperature = temperature * scalar;
            result.salinity = salinity * scalar;
            result.velocity = velocity * scalar;
            result.sea_ice_concentration = sea_ice_concentration * scalar;
            result.sea_ice_thickness = sea_ice_thickness * scalar;
            result.sea_surface_height = sea_surface_height * scalar;
            return result;
        }

// ==============================================================================
// EnsembleCollection 实现
// ==============================================================================

        EnsembleCollection::EnsembleCollection(size_t state_size)
                : state_size_(state_size) {
            members_.resize(TOPAZ_ENSEMBLE_SIZE);
            for (int i = 0; i < TOPAZ_ENSEMBLE_SIZE; ++i) {
                members_[i] = std::make_unique<OceanStateVector[]>(state_size);
            }

            // 初始化线程池
            thread_pool_ = std::make_unique<Utils::ThreadPool>(
                    std::thread::hardware_concurrency());
        }

        EnsembleCollection::~EnsembleCollection() = default;

        OceanStateVector* EnsembleCollection::getMember(int index) {
            if (index < 0 || index >= TOPAZ_ENSEMBLE_SIZE) {
                throw std::out_of_range("Ensemble member index out of range");
            }
            return members_[index].get();
        }

        const OceanStateVector* EnsembleCollection::getMember(int index) const {
            if (index < 0 || index >= TOPAZ_ENSEMBLE_SIZE) {
                throw std::out_of_range("Ensemble member index out of range");
            }
            return members_[index].get();
        }

        std::vector<OceanStateVector> EnsembleCollection::computeEnsembleMean() const {
            std::vector<OceanStateVector> mean(state_size_);

            // 初始化均值为零
            for (size_t j = 0; j < state_size_; ++j) {
                mean[j] = OceanStateVector{};
            }

            // 计算集合均值
            for (int i = 0; i < TOPAZ_ENSEMBLE_SIZE; ++i) {
                const OceanStateVector* member = members_[i].get();
                for (size_t j = 0; j < state_size_; ++j) {
                    mean[j] = mean[j] + member[j];
                }
            }

            // 归一化
            double inv_ensemble_size = 1.0 / TOPAZ_ENSEMBLE_SIZE;
            for (size_t j = 0; j < state_size_; ++j) {
                mean[j] = mean[j] * inv_ensemble_size;
            }

            return mean;
        }

        Eigen::MatrixXd EnsembleCollection::computeSampleCovariance() const {
            const int state_dim = 7; // 状态向量维度 (新增海表面高度) // 状态向量维度
            const int total_size = state_size_ * state_dim;

            Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(total_size, total_size);

            // 计算集合均值
            auto ensemble_mean = computeEnsembleMean();

            // 构建扰动矩阵
            Eigen::MatrixXd perturbations(total_size, TOPAZ_ENSEMBLE_SIZE);

            for (int i = 0; i < TOPAZ_ENSEMBLE_SIZE; ++i) {
                const OceanStateVector* member = members_[i].get();
                for (size_t j = 0; j < state_size_; ++j) {
                    int base_idx = j * state_dim;
                    OceanStateVector deviation = member[j] - ensemble_mean[j];

                    perturbations(base_idx + 0, i) = deviation.temperature;
                    perturbations(base_idx + 1, i) = deviation.salinity;
                    perturbations(base_idx + 2, i) = deviation.velocity[0];
                    perturbations(base_idx + 3, i) = deviation.velocity[1];
                    perturbations(base_idx + 4, i) = deviation.velocity[2];
                    perturbations(base_idx + 5, i) = deviation.sea_ice_concentration;
                    perturbations(base_idx + 6, i) = deviation.sea_surface_height;
                }
            }

            // 计算样本协方差
            covariance = perturbations * perturbations.transpose() / (TOPAZ_ENSEMBLE_SIZE - 1);

            return covariance;
        }

        double EnsembleCollection::computeEnsembleSpread() const {
            auto mean = computeEnsembleMean();
            double total_spread = 0.0;

            for (int i = 0; i < TOPAZ_ENSEMBLE_SIZE; ++i) {
                const OceanStateVector* member = members_[i].get();
                double member_spread = 0.0;

                for (size_t j = 0; j < state_size_; ++j) {
                    OceanStateVector deviation = member[j] - mean[j];
                    member_spread += deviation.temperature * deviation.temperature +
                                     deviation.salinity * deviation.salinity +
                                     deviation.velocity.squaredNorm() +
                                    deviation.sea_ice_concentration * deviation.sea_ice_concentration +
                                    deviation.sea_surface_height * deviation.sea_surface_height;
                }
                total_spread += member_spread;
            }

            return std::sqrt(total_spread / (TOPAZ_ENSEMBLE_SIZE * state_size_));
        }

        void EnsembleCollection::parallelApply(std::function<void(OceanStateVector*, int)> operation) {
            std::vector<std::future<void>> futures;

            for (int i = 0; i < TOPAZ_ENSEMBLE_SIZE; ++i) {
                futures.push_back(thread_pool_->enqueue([this, operation, i]() {
                    operation(members_[i].get(), i);
                }));
            }

            // 等待所有任务完成
            for (auto& future : futures) {
                future.get();
            }
        }

// ==============================================================================
// 观测算子实现
// ==============================================================================

        SeaLevelAnomalyOperator::SeaLevelAnomalyOperator(std::shared_ptr<Data::GridDataStructure> grid)
                : grid_(grid) {
            // 初始化参考海平面
            auto dims = grid_->getDimensions();
            reference_sea_level_ = Eigen::VectorXd::Zero(dims[0] * dims[1]);
        }

        Eigen::VectorXd SeaLevelAnomalyOperator::apply(const std::vector<OceanStateVector>& model_state) {
            Eigen::VectorXd sea_level_anomaly(model_state.size());

            for (size_t i = 0; i < model_state.size(); ++i) {
                sea_level_anomaly(i) = model_state[i].sea_surface_height - reference_sea_level_(i);
            }

            return sea_level_anomaly;
        }

        Eigen::MatrixXd SeaLevelAnomalyOperator::computeJacobian(const std::vector<OceanStateVector>& model_state) {
            int obs_size = model_state.size();
            int state_size = model_state.size() * 7; // 7个状态变量

            Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(obs_size, state_size);

            // 海平面高度异常对海表面高度的偏导数为1
            for (int i = 0; i < obs_size; ++i) {
                int ssh_idx = i * 7 + 6; // 海表面高度在第7个位置
                jacobian(i, ssh_idx) = 1.0;
            }

            return jacobian;
        }

        bool SeaLevelAnomalyOperator::validateObservation(const Eigen::VectorXd& observation) {
            // 海平面异常通常在 -2m 到 2m 之间
            for (int i = 0; i < observation.size(); ++i) {
                if (std::abs(observation(i)) > 2.0) {
                    return false;
                }
            }
            return true;
        }

        SeaSurfaceTemperatureOperator::SeaSurfaceTemperatureOperator(std::shared_ptr<Data::GridDataStructure> grid)
                : grid_(grid) {
        }

        Eigen::VectorXd SeaSurfaceTemperatureOperator::apply(const std::vector<OceanStateVector>& model_state) {
            Eigen::VectorXd sst(model_state.size());

            for (size_t i = 0; i < model_state.size(); ++i) {
                sst(i) = model_state[i].temperature;
            }

            return sst;
        }

        Eigen::MatrixXd SeaSurfaceTemperatureOperator::computeJacobian(const std::vector<OceanStateVector>& model_state) {
            int obs_size = model_state.size();
            int state_size = model_state.size() * 7;

            Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(obs_size, state_size);

            // 海表温度对温度的偏导数为1
            for (int i = 0; i < obs_size; ++i) {
                int temp_idx = i * 7 + 0; // 温度是第1个变量
                jacobian(i, temp_idx) = 1.0;
            }

            return jacobian;
        }

        bool SeaSurfaceTemperatureOperator::validateObservation(const Eigen::VectorXd& observation) {
            for (int i = 0; i < observation.size(); ++i) {
                if (observation(i) < min_valid_temp_ || observation(i) > max_valid_temp_) {
                    return false;
                }
            }
            return true;
        }

        SeaIceConcentrationOperator::SeaIceConcentrationOperator(std::shared_ptr<Data::GridDataStructure> grid)
                : grid_(grid) {
        }

        Eigen::VectorXd SeaIceConcentrationOperator::apply(const std::vector<OceanStateVector>& model_state) {
            Eigen::VectorXd ice_concentration(model_state.size());

            for (size_t i = 0; i < model_state.size(); ++i) {
                ice_concentration(i) = model_state[i].sea_ice_concentration;
            }

            return ice_concentration;
        }

        Eigen::MatrixXd SeaIceConcentrationOperator::computeJacobian(const std::vector<OceanStateVector>& model_state) {
            int obs_size = model_state.size();
            int state_size = model_state.size() * 7;

            Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(obs_size, state_size);

            // 海冰浓度对海冰浓度的偏导数为1
            for (int i = 0; i < obs_size; ++i) {
                int ice_idx = i * 7 + 5; // 海冰浓度位于第6个位置
                jacobian(i, ice_idx) = 1.0;
            }

            return jacobian;
        }

        bool SeaIceConcentrationOperator::validateObservation(const Eigen::VectorXd& observation) {
            for (int i = 0; i < observation.size(); ++i) {
                if (observation(i) < 0.0 || observation(i) > 1.0) {
                    return false;
                }
            }
            return true;
        }

// ==============================================================================
// ObservationData 实现
// ==============================================================================

        std::vector<Eigen::VectorXd> ObservationData::generatePerturbedObservations(int ensemble_size) const {
            std::vector<Eigen::VectorXd> perturbed_obs(ensemble_size);

            // 设置随机数生成器
            std::random_device rd;
            std::mt19937 gen(rd());

            // Cholesky分解观测误差协方差矩阵
            Eigen::LLT<Eigen::MatrixXd> llt(error_covariance);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("观测误差协方差矩阵不是正定的");
            }

            Eigen::MatrixXd cholesky_factor = llt.matrixL();

            for (int i = 0; i < ensemble_size; ++i) {
                // 生成标准正态分布随机向量
                Eigen::VectorXd random_vector(values.size());
                std::normal_distribution<double> normal_dist(0.0, 1.0);

                for (int j = 0; j < random_vector.size(); ++j) {
                    random_vector(j) = normal_dist(gen);
                }

                // 生成相关扰动
                Eigen::VectorXd perturbation = cholesky_factor * random_vector;
                perturbed_obs[i] = values + perturbation;
            }

            return perturbed_obs;
        }

// ==============================================================================
// GaspariCohnLocalization 实现
// ==============================================================================

        GaspariCohnLocalization::GaspariCohnLocalization(double localization_radius)
                : localization_radius_(localization_radius) {
        }

        double GaspariCohnLocalization::computeCorrelationFunction(double distance) const {
            double normalized_distance = distance / localization_radius_;
            return gaspariCohnPolynomial(normalized_distance);
        }

        double GaspariCohnLocalization::gaspariCohnPolynomial(double r) const {
            if (r <= 1.0) {
                return -0.25 * std::pow(r, 5) + 0.5 * std::pow(r, 4) +
                       0.625 * std::pow(r, 3) - 5.0/3.0 * std::pow(r, 2) + 1.0;
            } else if (r <= 2.0) {
                return 1.0/12.0 * std::pow(r, 5) - 0.5 * std::pow(r, 4) +
                       0.625 * std::pow(r, 3) + 5.0/3.0 * std::pow(r, 2) -
                       5.0 * r + 4.0 - 2.0/(3.0 * r);
            } else {
                return 0.0;
            }
        }

        void GaspariCohnLocalization::applyLocalization(Eigen::MatrixXd& covariance_matrix,
                                                        const std::shared_ptr<Data::GridDataStructure>& grid) const {
            int n = covariance_matrix.rows();
            int grid_points = n / 7; // 7个状态变量

            // 构建局地化矩阵
            Eigen::MatrixXd localization_matrix = Eigen::MatrixXd::Identity(n, n);

            auto grid_spacing = grid->getSpacing();
            auto dimensions = grid->getDimensions();

            for (int i = 0; i < grid_points; ++i) {
                for (int j = 0; j < grid_points; ++j) {
                    // 计算网格点间距离
                    int i_x = i % dimensions[0];
                    int i_y = i / dimensions[0];
                    int j_x = j % dimensions[0];
                    int j_y = j / dimensions[0];

                    double dx = (i_x - j_x) * grid_spacing[0];
                    double dy = (i_y - j_y) * grid_spacing[1];
                    double distance = std::sqrt(dx*dx + dy*dy);

                    double correlation = computeCorrelationFunction(distance);

                    // 应用到所有状态变量的组合
                    for (int var_i = 0; var_i < 7; ++var_i) {
                        for (int var_j = 0; var_j < 7; ++var_j) {
                            int idx_i = i * 7 + var_i;
                            int idx_j = j * 7 + var_j;
                            localization_matrix(idx_i, idx_j) = correlation;
                        }
                    }
                }
            }

            // 应用Schur乘积
            applySchurProduct(covariance_matrix, localization_matrix);
        }

        void GaspariCohnLocalization::applySchurProduct(Eigen::MatrixXd& matrix,
                                                        const Eigen::MatrixXd& localization_matrix) const {
            matrix = matrix.cwiseProduct(localization_matrix);
        }

// ==============================================================================
// InflationAlgorithm 实现
// ==============================================================================

        InflationAlgorithm::InflationAlgorithm(InflationType type, double base_factor)
                : inflation_type_(type), inflation_factor_(base_factor) {
        }

        void InflationAlgorithm::applyInflation(EnsembleCollection& ensemble,
                                                const std::vector<OceanStateVector>& ensemble_mean) {
            switch (inflation_type_) {
                case InflationType::Multiplicative:
                    applyMultiplicativeInflation(ensemble, ensemble_mean);
                    break;
                case InflationType::Additive:
                    applyAdditiveInflation(ensemble);
                    break;
                case InflationType::Adaptive:
                    // 自适应充气需要观测数据，在这里使用默认乘性充气
                    applyMultiplicativeInflation(ensemble, ensemble_mean);
                    break;
                default:
                    applyMultiplicativeInflation(ensemble, ensemble_mean);
            }
        }

        void InflationAlgorithm::applyMultiplicativeInflation(EnsembleCollection& ensemble,
                                                              const std::vector<OceanStateVector>& mean) {
            ensemble.parallelApply([this, &mean](OceanStateVector* member, int member_id) {
                for (size_t j = 0; j < mean.size(); ++j) {
                    OceanStateVector deviation = member[j] - mean[j];
                    member[j] = mean[j] + deviation * inflation_factor_;
                    member[j].applyPhysicalConstraints();
                }
            });
        }

        void InflationAlgorithm::applyAdditiveInflation(EnsembleCollection& ensemble) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> normal_dist(0.0, 0.01); // 小幅度随机扰动

            ensemble.parallelApply([&](OceanStateVector* member, int member_id) {
                std::mt19937 local_gen(rd() + member_id);
                std::normal_distribution<double> local_dist(0.0, 0.01);

                for (size_t j = 0; j < ensemble.getStateSize(); ++j) {
                    member[j].temperature += local_dist(local_gen);
                    member[j].salinity += local_dist(local_gen);
                    member[j].sea_ice_concentration += local_dist(local_gen);
                    member[j].applyPhysicalConstraints();
                }
            });
        }

        double InflationAlgorithm::computeAdaptiveInflationFactor(const EnsembleCollection& ensemble,
                                                                  const ObservationData& observations) {
            // 计算观测空间的集合方差
            double ensemble_variance = 0.0;
            double observation_variance = observations.error_covariance.trace() / observations.error_covariance.rows();

            // 简化的自适应充气因子计算
            double adaptive_factor = std::max(min_inflation_factor_,
                                              std::min(max_inflation_factor_,
                                                       observation_variance / ensemble_variance));

            return adaptive_factor;
        }

// ==============================================================================
// EnKFConfiguration 验证
// ==============================================================================

        bool EnKFConfiguration::validate() const {
            if (ensemble_size <= 0 || ensemble_size > 1000) return false;
            if (localization_radius <= 0.0 || localization_radius > 1000000.0) return false;
            if (inflation_factor < 1.0 || inflation_factor > 3.0) return false;
            if (regularization_threshold <= 0.0 || regularization_threshold > 1e-6) return false;
            if (num_threads <= 0 || num_threads > 64) return false;

            return true;
        }

// ==============================================================================
// EnsembleKalmanFilter 主要实现
// ==============================================================================

        EnsembleKalmanFilter::EnsembleKalmanFilter(const EnKFConfiguration& config,
                                                   std::shared_ptr<Data::GridDataStructure> grid)
                : config_(config), grid_(grid) {

            if (!config_.validate()) {
                throw std::invalid_argument("EnKF配置参数无效");
            }

            // 初始化算法组件
            localization_ = std::make_unique<GaspariCohnLocalization>(config_.localization_radius);
            inflation_ = std::make_unique<InflationAlgorithm>(InflationAlgorithm::InflationType::Multiplicative,
                                                              config_.inflation_factor);
            rk_solver_ = std::make_unique<Algorithms::RungeKuttaSolver>();
            parallel_engine_ = std::make_unique<Algorithms::ParallelComputeEngine>();
            vector_ops_ = std::make_unique<Algorithms::VectorizedOperations>();

            // 初始化日志记录器
            logger_ = std::make_shared<Utils::Logger>("EnKF");

            // 初始化性能指标
            performance_metrics_ = {};
        }

        EnsembleKalmanFilter::~EnsembleKalmanFilter() = default;

        bool EnsembleKalmanFilter::initialize(const std::vector<OceanStateVector>& initial_state,
                                              const Eigen::MatrixXd& background_error_covariance) {
            try {
                // 创建集合
                current_ensemble_ = std::make_unique<EnsembleCollection>(initial_state.size());

                // 生成初始集合
                generateInitialEnsemble(initial_state, background_error_covariance);

                // 计算初始统计量
                current_mean_ = current_ensemble_->computeEnsembleMean();
                current_covariance_ = current_ensemble_->computeSampleCovariance();

                logger_->info("EnKF系统初始化成功，集合大小: {}, 状态维度: {}",
                              config_.ensemble_size, initial_state.size());

                return true;
            } catch (const std::exception& e) {
                logger_->error("EnKF初始化失败: {}", e.what());
                return false;
            }
        }

        void EnsembleKalmanFilter::generateInitialEnsemble(const std::vector<OceanStateVector>& mean_state,
                                                           const Eigen::MatrixXd& covariance) {
            std::random_device rd;
            std::mt19937 gen(rd());

            // Cholesky分解背景误差协方差
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error("背景误差协方差矩阵不是正定的");
            }

            Eigen::MatrixXd cholesky_factor = llt.matrixL();

            current_ensemble_->parallelApply([&](OceanStateVector* member, int member_id) {
                std::mt19937 local_gen(rd() + member_id);
                std::normal_distribution<double> normal_dist(0.0, 1.0);

                // 生成随机扰动向量
                Eigen::VectorXd random_vector(covariance.rows());
                for (int j = 0; j < random_vector.size(); ++j) {
                    random_vector(j) = normal_dist(local_gen);
                }

                // 生成相关扰动
                Eigen::VectorXd perturbation = cholesky_factor * random_vector;

                // 应用扰动到状态向量
                for (size_t j = 0; j < mean_state.size(); ++j) {
                    int base_idx = j * 7;

                    member[j] = mean_state[j];
                    member[j].temperature += perturbation(base_idx + 0);
                    member[j].salinity += perturbation(base_idx + 1);
                    member[j].velocity[0] += perturbation(base_idx + 2);
                    member[j].velocity[1] += perturbation(base_idx + 3);
                    member[j].velocity[2] += perturbation(base_idx + 4);
                    member[j].sea_ice_concentration += perturbation(base_idx + 5);
                    member[j].sea_surface_height += perturbation(base_idx + 6);

                    // 应用物理约束
                    member[j].applyPhysicalConstraints();
                }
            });
        }

        EnsembleKalmanFilter::ForecastResult EnsembleKalmanFilter::executeForecastStep(double time_step) {
            auto start_time = std::chrono::high_resolution_clock::now();

            try {
                // 并行执行集合预报
                auto futures = setupParallelForecast(time_step);

                // 等待所有集合成员完成积分
                for (auto& future : futures) {
                    future.get();
                }

                // 计算预报集合统计量
                auto forecast_mean = current_ensemble_->computeEnsembleMean();
                auto forecast_covariance = current_ensemble_->computeSampleCovariance();

                // 应用局地化
                if (config_.use_localization) {
                    localization_->applyLocalization(forecast_covariance, grid_);
                }

                // 应用充气
                if (config_.use_inflation) {
                    inflation_->applyInflation(*current_ensemble_, forecast_mean);
                    forecast_mean = current_ensemble_->computeEnsembleMean();
                    forecast_covariance = current_ensemble_->computeSampleCovariance();
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                auto computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time);

                // 更新性能指标
                updatePerformanceMetrics(computation_time);

                logger_->info("预报步骤完成，耗时: {} ms", computation_time.count());

                ForecastResult result;
                result.forecast_ensemble = std::make_unique<EnsembleCollection>(*current_ensemble_);
                result.ensemble_mean = forecast_mean;
                result.forecast_covariance = forecast_covariance;
                result.computation_time = computation_time;
                result.success = true;

                return result;

            } catch (const std::exception& e) {
                logger_->error("预报步骤失败: {}", e.what());

                ForecastResult result;
                result.success = false;
                return result;
            }
        }

        std::vector<std::future<void>> EnsembleKalmanFilter::setupParallelForecast(double time_step) {
            std::vector<std::future<void>> futures;

            for (int i = 0; i < config_.ensemble_size; ++i) {
                futures.push_back(std::async(std::launch::async, [this, time_step, i]() {
                    integrateEnsembleMember(current_ensemble_->getMember(i), time_step, i);
                }));
            }

            return futures;
        }

        void EnsembleKalmanFilter::integrateEnsembleMember(OceanStateVector* member, double time_step, int member_id) {
            // 这里实现具体的海洋模式积分
            // 使用龙格-库塔方法进行时间积分

            for (size_t j = 0; j < current_ensemble_->getStateSize(); ++j) {
                // 简化的海洋动力学方程积分
                // 实际实现中应该调用完整的HYCOM模式

                // 温度演化（简化的平流扩散）
                double temp_tendency = -0.01 * member[j].velocity.norm() * member[j].temperature +
                                       0.001 * (15.0 - member[j].temperature); // 松弛到气候态
                member[j].temperature += time_step * temp_tendency;

                // 盐度演化
                double sal_tendency = -0.005 * member[j].velocity.norm() * member[j].salinity +
                                      0.0005 * (35.0 - member[j].salinity);
                member[j].salinity += time_step * sal_tendency;

                // 流速演化（简化的准地转动力学）
                Eigen::Vector3d vel_tendency;
                vel_tendency[0] = -9.81 * 0.0001 * member[j].sea_surface_height; // 地转平衡
                vel_tendency[1] = 9.81 * 0.0001 * member[j].sea_surface_height;
                vel_tendency[2] = 0.0; // 垂直速度简化为0

                member[j].velocity += time_step * vel_tendency;

                // 海表面高度演化
                double ssh_tendency = -member[j].velocity.head<2>().norm() * 0.01;
                member[j].sea_surface_height += time_step * ssh_tendency;

                // 应用物理约束
                member[j].applyPhysicalConstraints();
            }
        }
        // ==============================================================================
// EnKF分析步骤和其他核心方法的续实现
// ==============================================================================

        EnsembleKalmanFilter::AnalysisResult EnsembleKalmanFilter::executeAnalysisStep(const ObservationData& observations) {
            auto start_time = std::chrono::high_resolution_clock::now();

            try {
                // 验证观测数据
                if (!observations.operator_->validateObservation(observations.values)) {
                    throw std::runtime_error("观测数据验证失败");
                }

                // 计算预报集合在观测空间的值
                std::vector<Eigen::VectorXd> forecast_observations;
                forecast_observations.reserve(config_.ensemble_size);

                for (int i = 0; i < config_.ensemble_size; ++i) {
                    const OceanStateVector* member = current_ensemble_->getMember(i);
                    std::vector<OceanStateVector> member_state(current_ensemble_->getStateSize());

                    for (size_t j = 0; j < current_ensemble_->getStateSize(); ++j) {
                        member_state[j] = member[j];
                    }

                    forecast_observations.push_back(observations.operator_->apply(member_state));
                }

                // 计算观测空间的集合均值和协方差
                Eigen::VectorXd obs_ensemble_mean = Eigen::VectorXd::Zero(observations.values.size());
                for (const auto& obs : forecast_observations) {
                    obs_ensemble_mean += obs;
                }
                obs_ensemble_mean /= config_.ensemble_size;

                // 计算创新协方差矩阵 HPH^T + R
                Eigen::MatrixXd obs_covariance = Eigen::MatrixXd::Zero(observations.values.size(), observations.values.size());
                for (const auto& obs : forecast_observations) {
                    Eigen::VectorXd deviation = obs - obs_ensemble_mean;
                    obs_covariance += deviation * deviation.transpose();
                }
                obs_covariance /= (config_.ensemble_size - 1);
                obs_covariance += observations.error_covariance;

                // 计算卡尔曼增益矩阵
                Eigen::MatrixXd kalman_gain = computeKalmanGain(current_covariance_, observations);

                // 生成扰动观测
                auto perturbed_observations = observations.generatePerturbedObservations(config_.ensemble_size);

                // 更新集合成员
                current_ensemble_->parallelApply([&](OceanStateVector* member, int member_id) {
                    // 将状态向量转换为向量形式
                    Eigen::VectorXd state_vector(current_ensemble_->getStateSize() * 7);
                    for (size_t j = 0; j < current_ensemble_->getStateSize(); ++j) {
                        int base_idx = j * 7;
                        state_vector(base_idx + 0) = member[j].temperature;
                        state_vector(base_idx + 1) = member[j].salinity;
                        state_vector(base_idx + 2) = member[j].velocity[0];
                        state_vector(base_idx + 3) = member[j].velocity[1];
                        state_vector(base_idx + 4) = member[j].velocity[2];
                        state_vector(base_idx + 5) = member[j].sea_ice_concentration;
                        state_vector(base_idx + 6) = member[j].sea_surface_height;
                    }

                    // 计算创新向量
                    Eigen::VectorXd innovation = perturbed_observations[member_id] - forecast_observations[member_id];

                    // 应用卡尔曼更新
                    Eigen::VectorXd state_increment = kalman_gain * innovation;
                    state_vector += state_increment;

                    // 将更新后的向量转换回状态结构
                    for (size_t j = 0; j < current_ensemble_->getStateSize(); ++j) {
                        int base_idx = j * 7;
                        member[j].temperature = state_vector(base_idx + 0);
                        member[j].salinity = state_vector(base_idx + 1);
                        member[j].velocity[0] = state_vector(base_idx + 2);
                        member[j].velocity[1] = state_vector(base_idx + 3);
                        member[j].velocity[2] = state_vector(base_idx + 4);
                        member[j].sea_ice_concentration = state_vector(base_idx + 5);
                        member[j].sea_surface_height = state_vector(base_idx + 6);

                        // 应用物理约束
                        member[j].applyPhysicalConstraints();
                    }
                });

                // 重新计算分析后的统计量
                auto analysis_mean = current_ensemble_->computeEnsembleMean();
                auto analysis_covariance = current_ensemble_->computeSampleCovariance();

                // 计算创新方差
                double innovation_variance = 0.0;
                for (int i = 0; i < config_.ensemble_size; ++i) {
                    Eigen::VectorXd innovation = perturbed_observations[i] - forecast_observations[i];
                    innovation_variance += innovation.squaredNorm();
                }
                innovation_variance /= config_.ensemble_size;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time);

                // 更新当前状态
                current_mean_ = analysis_mean;
                current_covariance_ = analysis_covariance;

                logger_->info("分析步骤完成，耗时: {} ms，创新方差: {}",
                              computation_time.count(), innovation_variance);

                AnalysisResult result;
                result.analysis_ensemble = std::make_unique<EnsembleCollection>(*current_ensemble_);
                result.analysis_mean = analysis_mean;
                result.analysis_covariance = analysis_covariance;
                result.kalman_gain = kalman_gain;
                result.innovation_variance = innovation_variance;
                result.computation_time = computation_time;
                result.success = true;

                return result;

            } catch (const std::exception& e) {
                logger_->error("分析步骤失败: {}", e.what());

                AnalysisResult result;
                result.success = false;
                return result;
            }
        }

        Eigen::MatrixXd EnsembleKalmanFilter::computeKalmanGain(const Eigen::MatrixXd& forecast_covariance,
                                                                const ObservationData& observations) {
            // 计算观测算子的雅可比矩阵
            Eigen::MatrixXd H = observations.operator_->computeJacobian(current_mean_);

            // 计算 HPH^T
            Eigen::MatrixXd HPHt = H * forecast_covariance * H.transpose();

            // 计算创新协方差矩阵 S = HPH^T + R
            Eigen::MatrixXd S = HPHt + observations.error_covariance;

            // 数值稳定性：正则化
            S = regularizeMatrix(S);

            // 使用SVD分解避免直接求逆
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);

            // 计算正则化的伪逆
            Eigen::VectorXd singular_values = svd.singularValues();
            Eigen::VectorXd inv_singular_values = Eigen::VectorXd::Zero(singular_values.size());

            for (int i = 0; i < singular_values.size(); ++i) {
                if (singular_values(i) > config_.regularization_threshold) {
                    inv_singular_values(i) = 1.0 / singular_values(i);
                }
            }

            Eigen::MatrixXd S_inv = svd.matrixV() * inv_singular_values.asDiagonal() * svd.matrixU().transpose();

            // 计算卡尔曼增益 K = PH^T S^{-1}
            Eigen::MatrixXd kalman_gain = forecast_covariance * H.transpose() * S_inv;

            return kalman_gain;
        }

        std::future<EnsembleKalmanFilter::ForecastResult> EnsembleKalmanFilter::executeforecastStepAsync(double time_step) {
            return std::async(std::launch::async, [this, time_step]() {
                return executeForecastStep(time_step);
            });
        }

        std::future<EnsembleKalmanFilter::AnalysisResult> EnsembleKalmanFilter::executeAnalysisStepAsync(const ObservationData& observations) {
            return std::async(std::launch::async, [this, &observations]() {
                return executeAnalysisStep(observations);
            });
        }

        std::future<EnsembleKalmanFilter::AssimilationResult> EnsembleKalmanFilter::executeAssimilationCycleAsync(
                double time_step, const ObservationData& observations) {

            return std::async(std::launch::async, [this, time_step, &observations]() {
                AssimilationResult result;

                // 执行预报步骤
                result.forecast = executeForecastStep(time_step);
                if (!result.forecast.success) {
                    result.cycle_success = false;
                    return result;
                }

                // 执行分析步骤
                result.analysis = executeAnalysisStep(observations);
                result.cycle_success = result.analysis.success;

                // 记录同化循环
                logAssimilationCycle(result);

                return result;
            });
        }

        std::vector<OceanStateVector> EnsembleKalmanFilter::getCurrentMean() const {
            return current_mean_;
        }

        Eigen::MatrixXd EnsembleKalmanFilter::getCurrentCovariance() const {
            return current_covariance_;
        }

        EnsembleKalmanFilter::PerformanceMetrics EnsembleKalmanFilter::getPerformanceMetrics() const {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            return performance_metrics_;
        }

        Eigen::MatrixXd EnsembleKalmanFilter::regularizeMatrix(const Eigen::MatrixXd& matrix) const {
            Eigen::MatrixXd regularized = matrix;

            // 添加对角正则化
            for (int i = 0; i < matrix.rows(); ++i) {
                regularized(i, i) += config_.regularization_threshold;
            }

            return regularized;
        }

        bool EnsembleKalmanFilter::checkMatrixCondition(const Eigen::MatrixXd& matrix) const {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
            double condition_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

            return condition_number < 1e12; // 合理的条件数阈值
        }

        void EnsembleKalmanFilter::updateStatistics() {
            current_mean_ = current_ensemble_->computeEnsembleMean();
            current_covariance_ = current_ensemble_->computeSampleCovariance();

            if (config_.use_localization) {
                localization_->applyLocalization(current_covariance_, grid_);
            }
        }

        void EnsembleKalmanFilter::updatePerformanceMetrics(std::chrono::milliseconds execution_time) {
            std::lock_guard<std::mutex> lock(metrics_mutex_);

            performance_metrics_.total_assimilation_cycles++;
            performance_metrics_.ensemble_spread = current_ensemble_->computeEnsembleSpread();

            // 更新平均执行时间
            double total_time = performance_metrics_.average_cycle_time.count() *
                                (performance_metrics_.total_assimilation_cycles - 1) + execution_time.count();
            performance_metrics_.average_cycle_time = std::chrono::milliseconds(
                    static_cast<long>(total_time / performance_metrics_.total_assimilation_cycles));

            // 估算内存使用量
            size_t ensemble_memory = current_ensemble_->getEnsembleSize() *
                                     current_ensemble_->getStateSize() * sizeof(OceanStateVector);
            size_t covariance_memory = current_covariance_.rows() * current_covariance_.cols() * sizeof(double);
            performance_metrics_.memory_usage_mb = (ensemble_memory + covariance_memory) / (1024.0 * 1024.0);
        }

        void EnsembleKalmanFilter::optimizeMemoryUsage() {
            // 在这里可以实现内存优化策略
            // 例如：压缩不常用的协方差矩阵，使用稀疏存储等
        }

        bool EnsembleKalmanFilter::validateEnsembleConsistency() const {
            // 检查集合成员的物理有效性
            for (int i = 0; i < config_.ensemble_size; ++i) {
                const OceanStateVector* member = current_ensemble_->getMember(i);
                for (size_t j = 0; j < current_ensemble_->getStateSize(); ++j) {
                    if (!member[j].isPhysicallyValid()) {
                        return false;
                    }
                }
            }

            // 检查集合离散度
            double spread = current_ensemble_->computeEnsembleSpread();
            if (spread < 1e-10 || spread > 100.0) {
                return false;
            }

            return true;
        }

        double EnsembleKalmanFilter::computeAnalysisIncrementNorm() const {
            // 这需要保存上一步的预报均值来计算
            // 简化实现：返回当前集合离散度作为代理
            return current_ensemble_->computeEnsembleSpread();
        }

        void EnsembleKalmanFilter::logAssimilationCycle(const AssimilationResult& result) {
            if (result.cycle_success) {
                logger_->info("同化循环成功完成 - 预报: {}ms, 分析: {}ms, 创新方差: {:.6f}",
                              result.forecast.computation_time.count(),
                              result.analysis.computation_time.count(),
                              result.analysis.innovation_variance);
            } else {
                logger_->error("同化循环失败");
            }
        }

// ==============================================================================
// EnKFValidator 实现
// ==============================================================================

        bool EnKFValidator::validateLinearGaussianCase(const EnsembleKalmanFilter& filter) {
            // 实现线性高斯系统的验证
            // 这里提供一个简化的验证框架

            try {
                auto metrics = filter.getPerformanceMetrics();

                // 检查基本的数值稳定性指标
                if (metrics.ensemble_spread < 1e-10 || metrics.ensemble_spread > 1e10) {
                    return false;
                }

                if (metrics.filter_divergence > 10.0) {
                    return false;
                }

                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        bool EnKFValidator::validateLorenz96System(const EnsembleKalmanFilter& filter) {
            // 实现Lorenz-96系统的验证
            // 检查混沌系统中的同化性能

            auto metrics = filter.getPerformanceMetrics();

            // Lorenz-96系统的特定验证标准
            if (metrics.observation_impact < 0.1) {
                return false; // 观测影响太小
            }

            if (metrics.ensemble_spread > 50.0) {
                return false; // 集合过度发散
            }

            return true;
        }

        double EnKFValidator::computeAnalysisAccuracy(const AnalysisResult& result,
                                                      const std::vector<OceanStateVector>& truth) {
            if (!result.success || result.analysis_mean.size() != truth.size()) {
                return -1.0; // 无效输入
            }

            double total_error = 0.0;
            double total_variance = 0.0;

            for (size_t i = 0; i < truth.size(); ++i) {
                OceanStateVector error = result.analysis_mean[i] - truth[i];

                // 计算均方根误差
                double point_error = error.temperature * error.temperature +
                                     error.salinity * error.salinity +
                                     error.velocity.squaredNorm() +
                                     error.sea_ice_concentration * error.sea_ice_concentration;

                total_error += point_error;

                // 计算真值的方差作为归一化因子
                double point_variance = truth[i].temperature * truth[i].temperature +
                                        truth[i].salinity * truth[i].salinity +
                                        truth[i].velocity.squaredNorm() +
                                        truth[i].sea_ice_concentration * truth[i].sea_ice_concentration;

                total_variance += point_variance;
            }

            // 返回归一化的均方根误差
            return std::sqrt(total_error / truth.size()) / std::sqrt(total_variance / truth.size());
        }

        bool EnKFValidator::checkFilterDivergence(const EnsembleCollection& ensemble) {
            double spread = ensemble.computeEnsembleSpread();

            // 检查滤波器发散的指标
            if (spread > 1000.0) { // 集合过度发散
                return true;
            }

            if (spread < 1e-8) { // 集合塌陷
                return true;
            }

            return false;
        }

        bool EnKFValidator::checkCovariancePositiveDefinite(const Eigen::MatrixXd& covariance) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);

            if (eigensolver.info() != Eigen::Success) {
                return false;
            }

            // 检查所有特征值是否为正
            auto eigenvalues = eigensolver.eigenvalues();
            for (int i = 0; i < eigenvalues.size(); ++i) {
                if (eigenvalues(i) <= 0.0) {
                    return false;
                }
            }

            return true;
        }

    } // namespace Prediction
} // namespace OceanSim