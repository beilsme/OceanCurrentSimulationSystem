// src/algorithms/RungeKuttaSolver.cpp
#include "algorithms/RungeKuttaSolver.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace OceanSim {
    namespace Algorithms {

        RungeKuttaSolver::RungeKuttaSolver(Method method) {
            setMethod(method);
        }

        RungeKuttaSolver::RungeKuttaSolver(const ButcherTableau& tableau)
                : tableau_(tableau) {
        }

        void RungeKuttaSolver::setMethod(Method method) {
            switch (method) {
                case Method::RK4:
                    tableau_ = createRK4();
                    break;
                case Method::RK45:
                    tableau_ = createRK45();
                    break;
                case Method::DOPRI5:
                    tableau_ = createDOPRI5();
                    break;
                case Method::RK8:
                    tableau_ = createRK8();
                    break;
                case Method::GAUSS_LEGENDRE:
                    tableau_ = createGaussLegendre(2); // 默认2阶段
                    break;
                default:
                    tableau_ = createRK4();
            }
        }

        void RungeKuttaSolver::setCustomTableau(const ButcherTableau& tableau) {
            tableau_ = tableau;
        }

        Eigen::Vector3d RungeKuttaSolver::solve(const Eigen::Vector3d& y0,
                                                const Eigen::Vector3d& dy0,
                                                double dt,
                                                const VectorFunction& f) const {
            if (tableau_.is_implicit) {
                return solveImplicit(y0, dy0, dt, f);
            } else {
                return solveExplicit(y0, dy0, dt, f);
            }
        }

        Eigen::Vector3d RungeKuttaSolver::solveExplicit(const Eigen::Vector3d& y0,
                                                        const Eigen::Vector3d& dy0,
                                                        double dt,
                                                        const VectorFunction& f) const {
            int stages = tableau_.A.rows();
            std::vector<Eigen::Vector3d> k(stages);

            // 计算每个阶段的斜率
            for (int i = 0; i < stages; ++i) {
                Eigen::Vector3d y_temp = y0;

                // 计算中间值
                for (int j = 0; j < i; ++j) {
                    y_temp += dt * tableau_.A(i, j) * k[j];
                }

                // 计算当前阶段的斜率
                k[i] = f(y_temp, tableau_.c(i) * dt);
            }

            // 计算最终结果
            Eigen::Vector3d result = y0;
            for (int i = 0; i < stages; ++i) {
                result += dt * tableau_.b(i) * k[i];
            }

            return result;
        }

        Eigen::Vector3d RungeKuttaSolver::solveImplicit(const Eigen::Vector3d& y0,
                                                        const Eigen::Vector3d& dy0,
                                                        double dt,
                                                        const VectorFunction& f) const {
            // 使用牛顿迭代求解隐式阶段
            auto stages_solution = solveImplicitStages(y0, dt, f);

            // 计算最终结果
            Eigen::Vector3d result = y0;
            for (size_t i = 0; i < stages_solution.size(); ++i) {
                result += dt * tableau_.b(i) * f(stages_solution[i], tableau_.c(i) * dt);
            }

            return result;
        }

        std::vector<Eigen::Vector3d> RungeKuttaSolver::solveImplicitStages(
                const Eigen::Vector3d& y0,
                double dt,
                const VectorFunction& f) const {

            int stages = tableau_.A.rows();
            std::vector<Eigen::Vector3d> Y(stages, y0); // 初始猜测

            // 牛顿迭代求解非线性系统
            for (int iter = 0; iter < max_iterations_; ++iter) {
                std::vector<Eigen::Vector3d> F(stages);
                std::vector<Eigen::Matrix3d> J(stages);

                // 计算残差和雅可比矩阵
                for (int i = 0; i < stages; ++i) {
                    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                    for (int j = 0; j < stages; ++j) {
                        sum += tableau_.A(i, j) * f(Y[j], tableau_.c(j) * dt);
                    }

                    F[i] = Y[i] - y0 - dt * sum;
                    J[i] = computeJacobian(Y[i], tableau_.c(i) * dt, f);
                }

                // 求解线性系统 (使用简化的块对角近似)
                std::vector<Eigen::Vector3d> deltaY(stages);
                for (int i = 0; i < stages; ++i) {
                    Eigen::Matrix3d A_approx = Eigen::Matrix3d::Identity() - dt * tableau_.A(i, i) * J[i];
                    deltaY[i] = A_approx.lu().solve(-F[i]);
                }

                // 更新解
                double norm = 0.0;
                for (int i = 0; i < stages; ++i) {
                    Y[i] += deltaY[i];
                    norm += deltaY[i].norm();
                }

                if (norm < tolerance_) {
                    break;
                }
            }

            return Y;
        }

        Eigen::Matrix3d RungeKuttaSolver::computeJacobian(const Eigen::Vector3d& y,
                                                          double t,
                                                          const VectorFunction& f,
                                                          double eps) const {
            Eigen::Matrix3d jacobian;
            Eigen::Vector3d f0 = f(y, t);

            for (int j = 0; j < 3; ++j) {
                Eigen::Vector3d y_pert = y;
                y_pert(j) += eps;
                Eigen::Vector3d f_pert = f(y_pert, t);
                jacobian.col(j) = (f_pert - f0) / eps;
            }

            return jacobian;
        }

        Eigen::Vector3d RungeKuttaSolver::solveAdaptive(const Eigen::Vector3d& y0,
                                                        const Eigen::Vector3d& dy0,
                                                        double& dt,
                                                        const VectorFunction& f,
                                                        double tolerance,
                                                        double dt_min,
                                                        double dt_max) const {
            if (tableau_.b_err.size() == 0) {
                // 如果没有误差估计，使用固定步长
                return solve(y0, dy0, dt, f);
            }

            const double safety_factor = 0.9;
            const double max_factor = 2.0;
            const double min_factor = 0.5;

            while (true) {
                // 计算高阶和低阶解
                Eigen::Vector3d y_high = solve(y0, dy0, dt, f);

                // 计算低阶解（使用误差估计权重）
                int stages = tableau_.A.rows();
                std::vector<Eigen::Vector3d> k(stages);

                for (int i = 0; i < stages; ++i) {
                    Eigen::Vector3d y_temp = y0;
                    for (int j = 0; j < i; ++j) {
                        y_temp += dt * tableau_.A(i, j) * k[j];
                    }
                    k[i] = f(y_temp, tableau_.c(i) * dt);
                }

                Eigen::Vector3d y_low = y0;
                for (int i = 0; i < stages; ++i) {
                    y_low += dt * tableau_.b_err(i) * k[i];
                }

                // 计算误差估计
                double error = estimateError(y_low, y_high);

                if (error <= tolerance) {
                    // 接受步长，计算下一步的步长
                    double factor = safety_factor * std::pow(tolerance / error, 1.0 / (tableau_.order + 1));
                    factor = std::min(max_factor, std::max(min_factor, factor));
                    dt = std::min(dt_max, std::max(dt_min, factor * dt));
                    return y_high;
                } else {
                    // 拒绝步长，减小步长重试
                    double factor = safety_factor * std::pow(tolerance / error, 1.0 / (tableau_.order + 1));
                    factor = std::max(min_factor, factor);
                    dt = std::max(dt_min, factor * dt);

                    if (dt <= dt_min) {
                        throw std::runtime_error("时间步长小于最小值，无法满足误差容限");
                    }
                }
            }
        }

        double RungeKuttaSolver::estimateError(const Eigen::Vector3d& y_low,
                                               const Eigen::Vector3d& y_high) const {
            // 计算相对误差的范数
            Eigen::Vector3d error = y_high - y_low;
            Eigen::Vector3d scale = y_high.cwiseAbs().cwiseMax(y_low.cwiseAbs()).cwiseMax(1e-10);

            return (error.cwiseQuotient(scale)).norm() / std::sqrt(3.0);
        }

        double RungeKuttaSolver::solve(double y0, double t0, double dt, const ScalarFunction& f) const {
            // 标量版本的RK求解器
            int stages = tableau_.A.rows();
            std::vector<double> k(stages);

            for (int i = 0; i < stages; ++i) {
                double y_temp = y0;
                for (int j = 0; j < i; ++j) {
                    y_temp += dt * tableau_.A(i, j) * k[j];
                }
                k[i] = f(t0 + tableau_.c(i) * dt, y_temp);
            }

            double result = y0;
            for (int i = 0; i < stages; ++i) {
                result += dt * tableau_.b(i) * k[i];
            }

            return result;
        }

        std::vector<Eigen::Vector3d> RungeKuttaSolver::solveSequence(
                const Eigen::Vector3d& y0,
                double t0, double t_end, double dt,
                const VectorFunction& f) const {

            std::vector<Eigen::Vector3d> solution;
            solution.push_back(y0);

            Eigen::Vector3d y_current = y0;
            double t_current = t0;

            while (t_current < t_end) {
                double dt_actual = std::min(dt, t_end - t_current);
                y_current = solve(y_current, Eigen::Vector3d::Zero(), dt_actual, f);
                t_current += dt_actual;
                solution.push_back(y_current);
            }

            return solution;
        }

        // ========================= 预定义Butcher表实现 =========================

        RungeKuttaSolver::ButcherTableau RungeKuttaSolver::createRK4() {
            ButcherTableau tableau;
            tableau.order = 4;
            tableau.is_implicit = false;

            tableau.A = Eigen::MatrixXd::Zero(4, 4);
            tableau.A(1, 0) = 0.5;
            tableau.A(2, 1) = 0.5;
            tableau.A(3, 2) = 1.0;

            tableau.b = Eigen::VectorXd(4);
            tableau.b << 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0;

            tableau.c = Eigen::VectorXd(4);
            tableau.c << 0.0, 0.5, 0.5, 1.0;

            return tableau;
        }

        RungeKuttaSolver::ButcherTableau RungeKuttaSolver::createRK45() {
            ButcherTableau tableau;
            tableau.order = 4;
            tableau.is_implicit = false;

            // Runge-Kutta-Fehlberg 4(5)方法
            tableau.A = Eigen::MatrixXd::Zero(6, 6);
            tableau.A(1, 0) = 1.0/4.0;
            tableau.A(2, 0) = 3.0/32.0;
            tableau.A(2, 1) = 9.0/32.0;
            tableau.A(3, 0) = 1932.0/2197.0;
            tableau.A(3, 1) = -7200.0/2197.0;
            tableau.A(3, 2) = 7296.0/2197.0;
            tableau.A(4, 0) = 439.0/216.0;
            tableau.A(4, 1) = -8.0;
            tableau.A(4, 2) = 3680.0/513.0;
            tableau.A(4, 3) = -845.0/4104.0;
            tableau.A(5, 0) = -8.0/27.0;
            tableau.A(5, 1) = 2.0;
            tableau.A(5, 2) = -3544.0/2565.0;
            tableau.A(5, 3) = 1859.0/4104.0;
            tableau.A(5, 4) = -11.0/40.0;

            tableau.b = Eigen::VectorXd(6);
            tableau.b << 16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0;

            tableau.b_err = Eigen::VectorXd(6);
            tableau.b_err << 25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0;

            tableau.c = Eigen::VectorXd(6);
            tableau.c << 0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0;

            return tableau;
        }

        RungeKuttaSolver::ButcherTableau RungeKuttaSolver::createDOPRI5() {
            ButcherTableau tableau;
            tableau.order = 5;
            tableau.is_implicit = false;

            // Dormand-Prince 5(4)方法
            tableau.A = Eigen::MatrixXd::Zero(7, 7);
            tableau.A(1, 0) = 1.0/5.0;
            tableau.A(2, 0) = 3.0/40.0;
            tableau.A(2, 1) = 9.0/40.0;
            tableau.A(3, 0) = 44.0/45.0;
            tableau.A(3, 1) = -56.0/15.0;
            tableau.A(3, 2) = 32.0/9.0;
            tableau.A(4, 0) = 19372.0/6561.0;
            tableau.A(4, 1) = -25360.0/2187.0;
            tableau.A(4, 2) = 64448.0/6561.0;
            tableau.A(4, 3) = -212.0/729.0;
            tableau.A(5, 0) = 9017.0/3168.0;
            tableau.A(5, 1) = -355.0/33.0;
            tableau.A(5, 2) = 46732.0/5247.0;
            tableau.A(5, 3) = 49.0/176.0;
            tableau.A(5, 4) = -5103.0/18656.0;
            tableau.A(6, 0) = 35.0/384.0;
            tableau.A(6, 1) = 0.0;
            tableau.A(6, 2) = 500.0/1113.0;
            tableau.A(6, 3) = 125.0/192.0;
            tableau.A(6, 4) = -2187.0/6784.0;
            tableau.A(6, 5) = 11.0/84.0;

            tableau.b = Eigen::VectorXd(7);
            tableau.b << 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0;

            tableau.b_err = Eigen::VectorXd(7);
            tableau.b_err << 5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0;

            tableau.c = Eigen::VectorXd(7);
            tableau.c << 0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0;

            return tableau;
        }

        RungeKuttaSolver::ButcherTableau RungeKuttaSolver::createRK8() {
            ButcherTableau tableau;
            tableau.order = 8;
            tableau.is_implicit = false;

            // 简化的8阶方法（实际实现会更复杂）
            int stages = 13;
            tableau.A = Eigen::MatrixXd::Zero(stages, stages);
            tableau.b = Eigen::VectorXd::Zero(stages);
            tableau.c = Eigen::VectorXd::Zero(stages);

            // 这里需要实际的8阶RK系数，暂时使用简化实现
            // 实际使用时应该使用已验证的8阶RK系数

            return tableau;
        }

        RungeKuttaSolver::ButcherTableau RungeKuttaSolver::createGaussLegendre(int stages) {
            ButcherTableau tableau;
            tableau.order = 2 * stages;
            tableau.is_implicit = true;

            if (stages == 1) {
                // 1阶段Gauss-Legendre (中点法则)
                tableau.A = Eigen::MatrixXd(1, 1);
                tableau.A(0, 0) = 0.5;
                tableau.b = Eigen::VectorXd(1);
                tableau.b(0) = 1.0;
                tableau.c = Eigen::VectorXd(1);
                tableau.c(0) = 0.5;
            } else if (stages == 2) {
                // 2阶段Gauss-Legendre
                tableau.A = Eigen::MatrixXd(2, 2);
                tableau.A(0, 0) = 1.0/4.0;
                tableau.A(0, 1) = 1.0/4.0 - std::sqrt(3.0)/6.0;
                tableau.A(1, 0) = 1.0/4.0 + std::sqrt(3.0)/6.0;
                tableau.A(1, 1) = 1.0/4.0;

                tableau.b = Eigen::VectorXd(2);
                tableau.b(0) = 0.5;
                tableau.b(1) = 0.5;

                tableau.c = Eigen::VectorXd(2);
                tableau.c(0) = 0.5 - std::sqrt(3.0)/6.0;
                tableau.c(1) = 0.5 + std::sqrt(3.0)/6.0;
            } else {
                throw std::invalid_argument("仅支持1和2阶段的Gauss-Legendre方法");
            }

            return tableau;
        }

        // ========================= VectorizedRKSolver 实现 =========================

        VectorizedRKSolver::VectorizedRKSolver(RungeKuttaSolver::Method method) : solver_(method) {
        }

        std::vector<Eigen::Vector3d> VectorizedRKSolver::solveBatch(
                const std::vector<Eigen::Vector3d>& y0_batch,
                double dt,
                const BatchFunction& f) const {

            std::vector<Eigen::Vector3d> result;
            result.reserve(y0_batch.size());

            // 计算批量函数值
            std::vector<Eigen::Vector3d> f_values = f(y0_batch, 0.0);

            // 为每个初值求解
            for (size_t i = 0; i < y0_batch.size(); ++i) {
                auto single_f = [&f, i](const Eigen::Vector3d& y, double t) -> Eigen::Vector3d {
                    std::vector<Eigen::Vector3d> single_y = {y};
                    auto single_result = f(single_y, t);
                    return single_result[0];
                };

                Eigen::Vector3d solution = solver_.solve(y0_batch[i], Eigen::Vector3d::Zero(), dt, single_f);
                result.push_back(solution);
            }

            return result;
        }

        void VectorizedRKSolver::solveBatchSIMD(
                std::vector<Eigen::Vector3d>& y_batch,
                double dt,
                const BatchFunction& f) const {

            // SIMD优化版本的批量求解
            // 这里可以使用向量化指令来加速计算
            // 暂时使用标准实现
            auto result = solveBatch(y_batch, dt, f);
            y_batch = std::move(result);
        }

    } // namespace Algorithms
} // namespace OceanSim