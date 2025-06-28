// include/core/AdvectionDiffusionSolver.h
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "data/GridDataStructure.h"
#include "algorithms/FiniteDifferenceSolver.h"

namespace OceanSim {
    namespace Core {

/**
 * @brief 平流扩散方程求解器
 * 用于求解污染物扩散、温度传输等标量输运问题
 */
        class AdvectionDiffusionSolver {
        public:
            // 数值格式
            enum class NumericalScheme {
                UPWIND,                 // 一阶迎风格式
                LAX_WENDROFF,          // Lax-Wendroff格式
                TVD_SUPERBEE,          // TVD Superbee格式
                WENO5,                 // 五阶WENO格式
                QUICK,                 // QUICK格式
                MUSCL                  // MUSCL格式
            };

            // 时间积分方法
            enum class TimeIntegration {
                EXPLICIT_EULER,         // 显式欧拉
                IMPLICIT_EULER,         // 隐式欧拉
                CRANK_NICOLSON,        // Crank-Nicolson
                RUNGE_KUTTA_4,         // 四阶RK
                ADAMS_BASHFORTH        // Adams-Bashforth
            };

            // 边界条件类型
            enum class BoundaryType {
                DIRICHLET,             // 第一类边界条件
                NEUMANN,               // 第二类边界条件
                ROBIN,                 // 第三类边界条件
                PERIODIC,              // 周期边界条件
                OUTFLOW                // 流出边界条件
            };

            // 构造函数
            AdvectionDiffusionSolver(
                    std::shared_ptr<Data::GridDataStructure> grid,
                    NumericalScheme scheme = NumericalScheme::TVD_SUPERBEE,
                    TimeIntegration time_method = TimeIntegration::RUNGE_KUTTA_4);

            ~AdvectionDiffusionSolver() = default;

            // 初始化
            void setInitialCondition(const std::string& field_name,
                                     const Eigen::MatrixXd& initial_field);
            void setVelocityField(const std::string& u_field,
                                  const std::string& v_field,
                                  const std::string& w_field = "");

            // 物理参数设置
            void setDiffusionCoefficient(double diffusion_coeff);
            void setDiffusionTensor(const Eigen::Matrix3d& diffusion_tensor);
            void setReactionTerm(const std::function<double(double, double, double, double)>& reaction);
            void setSourceTerm(const Eigen::MatrixXd& source_field);

            // 边界条件
            void setBoundaryCondition(BoundaryType type, int boundary_id, double value = 0.0);
            void setInletConcentration(int inlet_id, double concentration);
            void setWallFlux(int wall_id, double flux);

            // 求解
            void solve(double dt);
            void solveToSteadyState(double tolerance = 1e-6, int max_iterations = 10000);
            void solveTimeSequence(double t_start, double t_end, double dt,
                                   const std::string& output_prefix = "");

            // 自适应时间步长
            void enableAdaptiveTimeStep(bool enable, double cfl_target = 0.5);
            double computeOptimalTimeStep() const;

            // 结果获取
            const Eigen::MatrixXd& getSolution() const { return current_solution_; }
            Eigen::MatrixXd& getSolution() { return current_solution_; }
            double getMaxConcentration() const;
            double getTotalMass() const;

            // 质量守恒
            bool checkMassConservation(double tolerance = 1e-8) const;
            double computeMassBalance() const;

            // 数值分析
            double computePecletNumber() const;
            double computeCourantNumber(double dt) const;
            void analyzeNumericalStability() const;

            // 性能监控
            void enableProfiling(bool enable) { profiling_enabled_ = enable; }
            double getComputationTime() const { return computation_time_; }
            int getIterationCount() const { return iteration_count_; }

        private:
            std::shared_ptr<Data::GridDataStructure> grid_;
            std::shared_ptr<Algorithms::FiniteDifferenceSolver> fd_solver_;

            // 数值方法参数
            NumericalScheme numerical_scheme_;
            TimeIntegration time_integration_;

            // 解变量
            Eigen::MatrixXd current_solution_;
            Eigen::MatrixXd previous_solution_;
            std::vector<Eigen::MatrixXd> solution_history_;

            // 速度场
            std::string u_field_name_, v_field_name_, w_field_name_;
            bool has_velocity_field_ = false;

            // 物理参数
            double diffusion_coefficient_ = 1e-6;
            Eigen::Matrix3d diffusion_tensor_;
            bool has_anisotropic_diffusion_ = false;

            // 源项和反应项
            Eigen::MatrixXd source_term_;
            std::function<double(double, double, double, double)> reaction_function_;
            bool has_source_term_ = false;
            bool has_reaction_term_ = false;

            // 边界条件
            std::map<int, std::pair<BoundaryType, double>> boundary_conditions_;

            // 自适应时间步长
            bool adaptive_timestep_ = false;
            double cfl_target_ = 0.5;
            double previous_dt_ = 0.0;

            // 性能监控
            bool profiling_enabled_ = false;
            double computation_time_ = 0.0;
            int iteration_count_ = 0;

            // 求解方法实现
            void solveExplicit(double dt);
            void solveImplicit(double dt);
            void solveCrankNicolson(double dt);
            void solveRungeKutta4(double dt);

            // 平流项计算
            Eigen::MatrixXd computeAdvection(const Eigen::MatrixXd& field, double dt);
            Eigen::MatrixXd computeAdvectionUpwind(const Eigen::MatrixXd& field, double dt);
            Eigen::MatrixXd computeAdvectionTVD(const Eigen::MatrixXd& field, double dt);
            Eigen::MatrixXd computeAdvectionWENO5(const Eigen::MatrixXd& field, double dt);

            // 扩散项计算
            Eigen::MatrixXd computeDiffusion(const Eigen::MatrixXd& field, double dt);
            Eigen::MatrixXd computeAnisotropicDiffusion(const Eigen::MatrixXd& field, double dt);

            // 限制器函数
            double superbeeFluxLimiter(double r) const;
            double vanLeerFluxLimiter(double r) const;
            double minmodFluxLimiter(double r) const;

            // WENO重构
            double wenoReconstruction(const std::vector<double>& stencil, bool left_biased) const;
            std::vector<double> wenoWeights(const std::vector<double>& beta_values) const;

            // 边界条件应用
            void applyBoundaryConditions(Eigen::MatrixXd& field);
            void applyDirichletBC(Eigen::MatrixXd& field, int boundary_id, double value);
            void applyNeumannBC(Eigen::MatrixXd& field, int boundary_id, double flux);
            void applyPeriodicBC(Eigen::MatrixXd& field);

            // 稀疏矩阵求解（隐式方法）
            void setupImplicitMatrix(Eigen::SparseMatrix<double>& A,
                                     Eigen::VectorXd& b, double dt);
            void solveLinearSystem(const Eigen::SparseMatrix<double>& A,
                                   const Eigen::VectorXd& b,
                                   Eigen::VectorXd& x);

            // 辅助方法
            bool isInternalPoint(int i, int j) const;
            double computeLocalCourantNumber(int i, int j, double dt) const;
            void updateSolutionHistory();

            // 数值稳定性检查
            void checkNumericalStability(double dt) const;
            void detectOscillations() const;
        };

/**
 * @brief 多组分扩散求解器
 * 处理多种化学物质的同时扩散和反应
 */
        class MultiComponentSolver : public AdvectionDiffusionSolver {
        public:
            MultiComponentSolver(std::shared_ptr<Data::GridDataStructure> grid,
                                 int num_components);

            // 组分管理
            void setComponentName(int component_id, const std::string& name);
            void setComponentDiffusivity(int component_id, double diffusivity);
            void setComponentInitialCondition(int component_id, const Eigen::MatrixXd& initial);

            // 反应网络
            void addReaction(int reactant1, int reactant2, int product, double rate_constant);
            void setDecayRate(int component_id, double decay_rate);

            // 求解
            void solveMultiComponent(double dt);
            const std::vector<Eigen::MatrixXd>& getComponentSolutions() const;

        private:
            int num_components_;
            std::vector<std::string> component_names_;
            std::vector<double> diffusivities_;
            std::vector<Eigen::MatrixXd> component_solutions_;
            std::vector<double> decay_rates_;

            // 反应网络
            struct Reaction {
                int reactant1, reactant2, product;
                double rate_constant;
            };
            std::vector<Reaction> reactions_;

            void computeReactionTerms(std::vector<Eigen::MatrixXd>& reaction_rates);
            void solveReactionSystem(double dt);
        };

    } // namespace Core
} // namespace OceanSim