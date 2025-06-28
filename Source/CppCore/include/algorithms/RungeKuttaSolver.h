// include/algorithms/RungeKuttaSolver.h
#pragma once

#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace OceanSim {
    namespace Algorithms {

/**
 * @brief 高阶龙格-库塔求解器
 * 支持自适应时间步长和多种RK方法
 */
        class RungeKuttaSolver {
        public:
            // 函数类型定义
            using VectorFunction = std::function<Eigen::Vector3d(const Eigen::Vector3d&, double)>;
            using ScalarFunction = std::function<double(double, double)>;

            // RK方法类型
            enum class Method {
                RK4,           // 经典四阶RK
                RK45,          // Runge-Kutta-Fehlberg
                DOPRI5,        // Dormand-Prince 5(4)
                RK8,           // 八阶RK
                GAUSS_LEGENDRE // 高斯-勒让德隐式方法
            };

            // Butcher表结构
            struct ButcherTableau {
                Eigen::MatrixXd A;  // 系数矩阵
                Eigen::VectorXd b;  // 权重向量
                Eigen::VectorXd c;  // 节点向量
                Eigen::VectorXd b_err; // 误差估计权重（可选）
                int order;          // 方法阶数
                bool is_implicit;   // 是否为隐式方法
            };

            // 构造函数
            explicit RungeKuttaSolver(Method method = Method::RK4);
            RungeKuttaSolver(const ButcherTableau& tableau);

            ~RungeKuttaSolver() = default;

            // 单步求解
            Eigen::Vector3d solve(const Eigen::Vector3d& y0,
                                  const Eigen::Vector3d& dy0,
                                  double dt,
                                  const VectorFunction& f) const;

            // 自适应求解
            Eigen::Vector3d solveAdaptive(const Eigen::Vector3d& y0,
                                          const Eigen::Vector3d& dy0,
                                          double& dt,
                                          const VectorFunction& f,
                                          double tolerance = 1e-6,
                                          double dt_min = 1e-10,
                                          double dt_max = 1.0) const;

            // 标量方程求解
            double solve(double y0, double t0, double dt, const ScalarFunction& f) const;

            // 时间序列求解
            std::vector<Eigen::Vector3d> solveSequence(
                    const Eigen::Vector3d& y0,
                    double t0, double t_end, double dt,
                    const VectorFunction& f) const;

            // 设置参数
            void setMethod(Method method);
            void setCustomTableau(const ButcherTableau& tableau);
            void setTolerance(double tol) { tolerance_ = tol; }
            void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }

            // 获取信息
            int getOrder() const { return tableau_.order; }
            int getStages() const { return tableau_.A.rows(); }
            bool isImplicit() const { return tableau_.is_implicit; }

            // 预定义方法
            static ButcherTableau createRK4();
            static ButcherTableau createRK45();
            static ButcherTableau createDOPRI5();
            static ButcherTableau createRK8();
            static ButcherTableau createGaussLegendre(int stages);

        private:
            ButcherTableau tableau_;
            double tolerance_ = 1e-6;
            int max_iterations_ = 100;

            // 内部方法
            Eigen::Vector3d solveExplicit(const Eigen::Vector3d& y0,
                                          const Eigen::Vector3d& dy0,
                                          double dt,
                                          const VectorFunction& f) const;

            Eigen::Vector3d solveImplicit(const Eigen::Vector3d& y0,
                                          const Eigen::Vector3d& dy0,
                                          double dt,
                                          const VectorFunction& f) const;

            double estimateError(const Eigen::Vector3d& y_low,
                                 const Eigen::Vector3d& y_high) const;

            // 牛顿迭代求解隐式方法
            std::vector<Eigen::Vector3d> solveImplicitStages(
                    const Eigen::Vector3d& y0,
                    double dt,
                    const VectorFunction& f) const;

            // 雅可比矩阵数值计算
            Eigen::Matrix3d computeJacobian(const Eigen::Vector3d& y,
                                            double t,
                                            const VectorFunction& f,
                                            double eps = 1e-8) const;
        };

/**
 * @brief 向量化龙格-库塔求解器
 * 用于批量处理多个初值问题
 */
        class VectorizedRKSolver {
        public:
            using BatchFunction = std::function<std::vector<Eigen::Vector3d>(
                    const std::vector<Eigen::Vector3d>&, double)>;

            VectorizedRKSolver(RungeKuttaSolver::Method method = RungeKuttaSolver::Method::RK4);

            // 批量求解
            std::vector<Eigen::Vector3d> solveBatch(
                    const std::vector<Eigen::Vector3d>& y0_batch,
                    double dt,
                    const BatchFunction& f) const;

            // SIMD优化版本
            void solveBatchSIMD(
                    std::vector<Eigen::Vector3d>& y_batch,
                    double dt,
                    const BatchFunction& f) const;

        private:
            RungeKuttaSolver solver_;
        };

    } // namespace Algorithms
} // namespace OceanSim