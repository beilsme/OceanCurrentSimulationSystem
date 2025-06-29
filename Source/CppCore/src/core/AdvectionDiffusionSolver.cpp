// src/core/AdvectionDiffusionSolver.cpp
#include "core/AdvectionDiffusionSolver.h"
#include "core/PerformanceProfiler.h"
#include "algorithms/FiniteDifferenceSolver.h"
#include "data/GridDataStructure.h"
#include "utils/MathUtils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>


namespace OceanSim {
    namespace Core {

        AdvectionDiffusionSolver::AdvectionDiffusionSolver(
                std::shared_ptr<Data::GridDataStructure> grid,
                NumericalScheme scheme,
                TimeIntegration time_method)
                : grid_(grid), numerical_scheme_(scheme), time_integration_(time_method),
                  diffusion_tensor_(Eigen::Matrix3d::Identity()), has_anisotropic_diffusion_(false),
                  has_source_term_(false), has_reaction_term_(false), adaptive_timestep_(false),
                  profiling_enabled_(false), computation_time_(0.0), iteration_count_(0) {

            if (!grid_) {
                throw std::invalid_argument("网格数据结构指针不能为空");
            }

            auto dims = grid_->getDimensions();
            current_solution_ = Eigen::MatrixXd::Zero(dims[1], dims[0]);
            previous_solution_ = current_solution_;

            // 创建有限差分求解器
            int grid_size = dims[0];
            fd_solver_ = std::make_shared<Algorithms::FiniteDifferenceSolver>(grid_size, 1.0);
        }

        void AdvectionDiffusionSolver::setInitialCondition(const std::string& field_name,
                                                           const Eigen::MatrixXd& initial_field) {
            auto dims = grid_->getDimensions();
            if (initial_field.rows() != dims[1] || initial_field.cols() != dims[0]) {
                throw std::invalid_argument("初始条件维度与网格不匹配");
            }

            current_solution_ = initial_field;
            previous_solution_ = initial_field;

            // 将初始条件添加到网格数据结构中
            grid_->addField(field_name, initial_field);
        }

        void AdvectionDiffusionSolver::setVelocityField(const std::string& u_field,
                                                        const std::string& v_field,
                                                        const std::string& w_field) {
            u_field_name_ = u_field;
            v_field_name_ = v_field;
            w_field_name_ = w_field;
            has_velocity_field_ = true;
        }

        void AdvectionDiffusionSolver::setDiffusionCoefficient(double diffusion_coeff) {
            if (diffusion_coeff < 0.0) {
                throw std::invalid_argument("扩散系数必须非负");
            }
            diffusion_coefficient_ = diffusion_coeff;
            has_anisotropic_diffusion_ = false;
        }

        void AdvectionDiffusionSolver::setDiffusionTensor(const Eigen::Matrix3d& diffusion_tensor) {
            diffusion_tensor_ = diffusion_tensor;
            has_anisotropic_diffusion_ = true;
        }

        void AdvectionDiffusionSolver::setReactionTerm(const std::function<double(double, double, double, double)>& reaction) {
            reaction_function_ = reaction;
            has_reaction_term_ = true;
        }

        void AdvectionDiffusionSolver::setSourceTerm(const Eigen::MatrixXd& source_field) {
            auto dims = grid_->getDimensions();
            if (source_field.rows() != dims[1] || source_field.cols() != dims[0]) {
                throw std::invalid_argument("源项维度与网格不匹配");
            }

            source_term_ = source_field;
            has_source_term_ = true;
        }

        void AdvectionDiffusionSolver::setBoundaryCondition(BoundaryType type, int boundary_id, double value) {
            boundary_conditions_[boundary_id] = std::make_pair(type, value);
        }

        void AdvectionDiffusionSolver::setInletConcentration(int inlet_id, double concentration) {
            setBoundaryCondition(BoundaryType::DIRICHLET, inlet_id, concentration);
        }

        void AdvectionDiffusionSolver::setWallFlux(int wall_id, double flux) {
            setBoundaryCondition(BoundaryType::NEUMANN, wall_id, flux);
        }

        void AdvectionDiffusionSolver::solve(double dt) {
            if (profiling_enabled_) {
                auto timer = PerformanceProfiler::getInstance().createTimer("AdvectionDiffusion_Solve");
                auto start_time = std::chrono::high_resolution_clock::now();

                solveSingleTimeStep(dt);

                auto end_time = std::chrono::high_resolution_clock::now();
                computation_time_ += std::chrono::duration<double>(end_time - start_time).count();
                iteration_count_++;
            } else {
                solveSingleTimeStep(dt);
            }
        }

        void AdvectionDiffusionSolver::solveSingleTimeStep(double dt) {
            // 检查数值稳定性
            if (adaptive_timestep_) {
                dt = computeOptimalTimeStep();
            }

            checkNumericalStability(dt);

            // 根据时间积分方法选择求解策略
            switch (time_integration_) {
                case TimeIntegration::EXPLICIT_EULER:
                    solveExplicit(dt);
                    break;
                case TimeIntegration::IMPLICIT_EULER:
                    solveImplicit(dt);
                    break;
                case TimeIntegration::CRANK_NICOLSON:
                    solveCrankNicolson(dt);
                    break;
                case TimeIntegration::RUNGE_KUTTA_4:
                    solveRungeKutta4(dt);
                    break;
                case TimeIntegration::ADAMS_BASHFORTH:
                    solveAdamsBashforth(dt);
                    break;
                default:
                    solveExplicit(dt);
            }

            // 应用边界条件
            applyBoundaryConditions(current_solution_);

            // 更新历史记录
            updateSolutionHistory();
        }

        void AdvectionDiffusionSolver::solveExplicit(double dt) {
            previous_solution_ = current_solution_;

            // 计算平流项
            Eigen::MatrixXd advection_term = computeAdvection(current_solution_, dt);

            // 计算扩散项
            Eigen::MatrixXd diffusion_term = computeDiffusion(current_solution_, dt);

            // 更新解
            current_solution_ = previous_solution_ + dt * (diffusion_term - advection_term);

            // 添加源项
            if (has_source_term_) {
                current_solution_ += dt * source_term_;
            }

            // 添加反应项
            if (has_reaction_term_) {
                addReactionTerm(dt);
            }
        }

        void AdvectionDiffusionSolver::solveImplicit(double dt) {
            // 构建隐式系统矩阵
            auto dims = grid_->getDimensions();
            int n = dims[0] * dims[1];

            Eigen::SparseMatrix<double> A(n, n);
            Eigen::VectorXd b(n);

            setupImplicitMatrix(A, b, dt);

            // 求解线性系统
            Eigen::VectorXd x(n);
            solveLinearSystem(A, b, x);

            // 将解向量转换回矩阵形式
            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    int idx = j * dims[0] + i;
                    current_solution_(j, i) = x(idx);
                }
            }
        }

        void AdvectionDiffusionSolver::solveCrankNicolson(double dt) {
            previous_solution_ = current_solution_;

            // Crank-Nicolson方法：(I - dt/2 * L) * u^{n+1} = (I + dt/2 * L) * u^n
            auto dims = grid_->getDimensions();
            int n = dims[0] * dims[1];

            // 构建左边系数矩阵和右边向量
            Eigen::SparseMatrix<double> A_left(n, n);
            Eigen::VectorXd b_right(n);

            setupCrankNicolsonMatrix(A_left, b_right, dt);

            // 求解线性系统
            Eigen::VectorXd x(n);
            solveLinearSystem(A_left, b_right, x);

            // 将解向量转换回矩阵形式
            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    int idx = j * dims[0] + i;
                    current_solution_(j, i) = x(idx);
                }
            }
        }

        void AdvectionDiffusionSolver::solveRungeKutta4(double dt) {
            previous_solution_ = current_solution_;

            // RK4方法的四个阶段
            Eigen::MatrixXd k1, k2, k3, k4;
            Eigen::MatrixXd temp_solution;

            // k1 = f(t, y)
            k1 = computeRightHandSide(current_solution_);

            // k2 = f(t + dt/2, y + dt/2 * k1)
            temp_solution = current_solution_ + (dt/2) * k1;
            applyBoundaryConditions(temp_solution);
            k2 = computeRightHandSide(temp_solution);

            // k3 = f(t + dt/2, y + dt/2 * k2)
            temp_solution = current_solution_ + (dt/2) * k2;
            applyBoundaryConditions(temp_solution);
            k3 = computeRightHandSide(temp_solution);

            // k4 = f(t + dt, y + dt * k3)
            temp_solution = current_solution_ + dt * k3;
            applyBoundaryConditions(temp_solution);
            k4 = computeRightHandSide(temp_solution);

            // 更新解
            current_solution_ += (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

            // 添加源项和反应项
            if (has_source_term_) {
                current_solution_ += dt * source_term_;
            }

            if (has_reaction_term_) {
                addReactionTerm(dt);
            }
        }

        void AdvectionDiffusionSolver::solveAdamsBashforth(double dt) {
            // Adams-Bashforth方法需要历史信息
            if (solution_history_.size() < 2) {
                // 如果历史不足，使用RK4启动
                solveRungeKutta4(dt);
                return;
            }

            previous_solution_ = current_solution_;

            // 二阶Adams-Bashforth
            Eigen::MatrixXd f_n = computeRightHandSide(current_solution_);
            Eigen::MatrixXd f_n_minus_1 = computeRightHandSide(solution_history_.back());

            current_solution_ += dt * (1.5 * f_n - 0.5 * f_n_minus_1);

            // 添加源项和反应项
            if (has_source_term_) {
                current_solution_ += dt * source_term_;
            }

            if (has_reaction_term_) {
                addReactionTerm(dt);
            }
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeRightHandSide(const Eigen::MatrixXd& field) {
            Eigen::MatrixXd advection_term = computeAdvection(field, 1.0);
            Eigen::MatrixXd diffusion_term = computeDiffusion(field, 1.0);
            return diffusion_term - advection_term;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvection(const Eigen::MatrixXd& field, double dt) {
            if (!has_velocity_field_) {
                return Eigen::MatrixXd::Zero(field.rows(), field.cols());
            }

            switch (numerical_scheme_) {
                case NumericalScheme::UPWIND:
                    return computeAdvectionUpwind(field, dt);
                case NumericalScheme::LAX_WENDROFF:
                    return computeAdvectionLaxWendroff(field, dt);
                case NumericalScheme::TVD_SUPERBEE:
                    return computeAdvectionTVD(field, dt);
                case NumericalScheme::WENO5:
                    return computeAdvectionWENO5(field, dt);
                case NumericalScheme::QUICK:
                    return computeAdvectionQUICK(field, dt);
                case NumericalScheme::MUSCL:
                    return computeAdvectionMUSCL(field, dt);
                default:
                    return computeAdvectionUpwind(field, dt);
            }
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionUpwind(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            // 获取速度场
            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            for (int i = 1; i < dims[0] - 1; ++i) {
                for (int j = 1; j < dims[1] - 1; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // 一阶迎风格式
                    double dudx, dvdy;

                    if (u > 0) {
                        dudx = (field(j, i) - field(j, i-1)) / dx;
                    } else {
                        dudx = (field(j, i+1) - field(j, i)) / dx;
                    }

                    if (v > 0) {
                        dvdy = (field(j, i) - field(j-1, i)) / dy;
                    } else {
                        dvdy = (field(j+1, i) - field(j, i)) / dy;
                    }

                    result(j, i) = u * dudx + v * dvdy;
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionLaxWendroff(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            for (int i = 1; i < dims[0] - 1; ++i) {
                for (int j = 1; j < dims[1] - 1; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // Lax-Wendroff格式
                    double u_plus = (u + std::abs(u)) / 2.0;
                    double u_minus = (u - std::abs(u)) / 2.0;
                    double v_plus = (v + std::abs(v)) / 2.0;
                    double v_minus = (v - std::abs(v)) / 2.0;

                    double flux_x = u_plus * field(j, i-1) + u_minus * field(j, i+1);
                    double flux_y = v_plus * field(j-1, i) + v_minus * field(j+1, i);

                    // 添加扩散项修正
                    double nu_x = 0.5 * u * u * dt / dx;
                    double nu_y = 0.5 * v * v * dt / dy;

                    flux_x += nu_x * (field(j, i+1) - 2*field(j, i) + field(j, i-1)) / dx;
                    flux_y += nu_y * (field(j+1, i) - 2*field(j, i) + field(j-1, i)) / dy;

                    result(j, i) = (flux_x / dx) + (flux_y / dy);
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionTVD(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            for (int i = 2; i < dims[0] - 2; ++i) {
                for (int j = 2; j < dims[1] - 2; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // TVD格式使用限制器
                    double dudx, dvdy;

                    if (u > 0) {
                        double r = safeDivision(field(j, i) - field(j, i-1),
                                                field(j, i-1) - field(j, i-2));
                        double limiter = superbeeFluxLimiter(r);
                        dudx = (field(j, i) - field(j, i-1)) / dx;
                        dudx += 0.5 * limiter * (field(j, i-1) - field(j, i-2)) / dx;
                    } else {
                        double r = safeDivision(field(j, i+1) - field(j, i),
                                                field(j, i+2) - field(j, i+1));
                        double limiter = superbeeFluxLimiter(r);
                        dudx = (field(j, i+1) - field(j, i)) / dx;
                        dudx += 0.5 * limiter * (field(j, i+2) - field(j, i+1)) / dx;
                    }

                    // Y方向类似处理
                    if (v > 0) {
                        double r = safeDivision(field(j, i) - field(j-1, i),
                                                field(j-1, i) - field(j-2, i));
                        double limiter = superbeeFluxLimiter(r);
                        dvdy = (field(j, i) - field(j-1, i)) / dy;
                        dvdy += 0.5 * limiter * (field(j-1, i) - field(j-2, i)) / dy;
                    } else {
                        double r = safeDivision(field(j+1, i) - field(j, i),
                                                field(j+2, i) - field(j+1, i));
                        double limiter = superbeeFluxLimiter(r);
                        dvdy = (field(j+1, i) - field(j, i)) / dy;
                        dvdy += 0.5 * limiter * (field(j+2, i) - field(j+1, i)) / dy;
                    }

                    result(j, i) = u * dudx + v * dvdy;
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionWENO5(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            // WENO5需要更大的模板
            for (int i = 3; i < dims[0] - 3; ++i) {
                for (int j = 3; j < dims[1] - 3; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // X方向WENO重构
                    std::vector<double> stencil_x(5);
                    if (u > 0) {
                        for (int k = 0; k < 5; ++k) {
                            stencil_x[k] = field(j, i-2+k);
                        }
                    } else {
                        for (int k = 0; k < 5; ++k) {
                            stencil_x[k] = field(j, i+2-k);
                        }
                    }
                    double dudx = wenoReconstruction(stencil_x, u > 0) / dx;

                    // Y方向WENO重构
                    std::vector<double> stencil_y(5);
                    if (v > 0) {
                        for (int k = 0; k < 5; ++k) {
                            stencil_y[k] = field(j-2+k, i);
                        }
                    } else {
                        for (int k = 0; k < 5; ++k) {
                            stencil_y[k] = field(j+2-k, i);
                        }
                    }
                    double dvdy = wenoReconstruction(stencil_y, v > 0) / dy;

                    result(j, i) = u * dudx + v * dvdy;
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionQUICK(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            for (int i = 2; i < dims[0] - 2; ++i) {
                for (int j = 2; j < dims[1] - 2; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // QUICK格式
                    double dudx, dvdy;

                    if (u > 0) {
                        // 上游权重差分
                        dudx = (3*field(j, i) + 3*field(j, i-1) - field(j, i-2) - field(j, i+1)) / (8*dx);
                    } else {
                        dudx = (field(j, i-1) + field(j, i+2) - 3*field(j, i) - 3*field(j, i+1)) / (8*dx);
                    }

                    if (v > 0) {
                        dvdy = (3*field(j, i) + 3*field(j-1, i) - field(j-2, i) - field(j+1, i)) / (8*dy);
                    } else {
                        dvdy = (field(j-1, i) + field(j+2, i) - 3*field(j, i) - 3*field(j+1, i)) / (8*dy);
                    }

                    result(j, i) = u * dudx + v * dvdy;
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAdvectionMUSCL(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            for (int i = 2; i < dims[0] - 2; ++i) {
                for (int j = 2; j < dims[1] - 2; ++j) {
                    double u = u_field(j, i);
                    double v = v_field(j, i);

                    // MUSCL格式使用斜率限制器
                    double dudx, dvdy;

                    if (u > 0) {
                        double delta_minus = field(j, i) - field(j, i-1);
                        double delta_plus = field(j, i-1) - field(j, i-2);
                        double slope = minmodFluxLimiter(delta_minus / delta_plus) * delta_minus;
                        dudx = (field(j, i) - field(j, i-1) + 0.5 * slope) / dx;
                    } else {
                        double delta_minus = field(j, i+1) - field(j, i);
                        double delta_plus = field(j, i+2) - field(j, i+1);
                        double slope = minmodFluxLimiter(delta_minus / delta_plus) * delta_minus;
                        dudx = (field(j, i+1) - field(j, i) + 0.5 * slope) / dx;
                    }

                    // Y方向类似处理
                    if (v > 0) {
                        double delta_minus = field(j, i) - field(j-1, i);
                        double delta_plus = field(j-1, i) - field(j-2, i);
                        double slope = minmodFluxLimiter(delta_minus / delta_plus) * delta_minus;
                        dvdy = (field(j, i) - field(j-1, i) + 0.5 * slope) / dy;
                    } else {
                        double delta_minus = field(j+1, i) - field(j, i);
                        double delta_plus = field(j+2, i) - field(j+1, i);
                        double slope = minmodFluxLimiter(delta_minus / delta_plus) * delta_minus;
                        dvdy = (field(j+1, i) - field(j, i) + 0.5 * slope) / dy;
                    }

                    result(j, i) = u * dudx + v * dvdy;
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeDiffusion(const Eigen::MatrixXd& field, double dt) {
            if (has_anisotropic_diffusion_) {
                return computeAnisotropicDiffusion(field, dt);
            }

            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];
            double dx2 = dx * dx;
            double dy2 = dy * dy;

            // 标准的二阶中心差分格式
            for (int i = 1; i < dims[0] - 1; ++i) {
                for (int j = 1; j < dims[1] - 1; ++j) {
                    double d2udx2 = (field(j, i+1) - 2*field(j, i) + field(j, i-1)) / dx2;
                    double d2udy2 = (field(j+1, i) - 2*field(j, i) + field(j-1, i)) / dy2;

                    result(j, i) = diffusion_coefficient_ * (d2udx2 + d2udy2);
                }
            }

            return result;
        }

        Eigen::MatrixXd AdvectionDiffusionSolver::computeAnisotropicDiffusion(const Eigen::MatrixXd& field, double dt) {
            auto dims = grid_->getDimensions();
            Eigen::MatrixXd result = Eigen::MatrixXd::Zero(dims[1], dims[0]);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            // 各向异性扩散：∇·(D∇u)
            for (int i = 1; i < dims[0] - 1; ++i) {
                for (int j = 1; j < dims[1] - 1; ++j) {
                    // 计算梯度
                    double dudx = (field(j, i+1) - field(j, i-1)) / (2*dx);
                    double dudy = (field(j+1, i) - field(j-1, i)) / (2*dy);

                    // 扩散张量作用
                    double flux_x = diffusion_tensor_(0,0) * dudx + diffusion_tensor_(0,1) * dudy;
                    double flux_y = diffusion_tensor_(1,0) * dudx + diffusion_tensor_(1,1) * dudy;

                    // 计算散度 ∇·(D∇u)
                    if (i > 1 && i < dims[0] - 2) {
                        double dflux_x_dx = (flux_x - (diffusion_tensor_(0,0) * (field(j, i) - field(j, i-2))/(2*dx) +
                                                       diffusion_tensor_(0,1) * (field(j+1, i-1) - field(j-1, i-1))/(2*dy))) / dx;
                        result(j, i) += dflux_x_dx;
                    }

                    if (j > 1 && j < dims[1] - 2) {
                        double dflux_y_dy = (flux_y - (diffusion_tensor_(1,0) * (field(j-1, i+1) - field(j-1, i-1))/(2*dx) +
                                                       diffusion_tensor_(1,1) * (field(j, i) - field(j-2, i))/(2*dy))) / dy;
                        result(j, i) += dflux_y_dy;
                    }
                }
            }

            return result;
        }

        void AdvectionDiffusionSolver::addReactionTerm(double dt) {
            auto dims = grid_->getDimensions();
            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    double x = i * grid_->getSpacing()[0];
                    double y = j * grid_->getSpacing()[1];
                    double concentration = current_solution_(j, i);
                    double reaction = reaction_function_(x, y, 0.0, concentration);
                    current_solution_(j, i) += dt * reaction;
                }
            }
        }

        // ========================= 限制器函数实现 =========================

        double AdvectionDiffusionSolver::superbeeFluxLimiter(double r) const {
            if (r <= 0) return 0.0;
            if (r <= 0.5) return 2.0 * r;
            if (r <= 1.0) return 1.0;
            return std::min(r, std::min(2.0, 2.0 * r));
        }

        double AdvectionDiffusionSolver::vanLeerFluxLimiter(double r) const {
            if (r <= 0) return 0.0;
            return (2.0 * r) / (1.0 + r);
        }

        double AdvectionDiffusionSolver::minmodFluxLimiter(double r) const {
            if (r <= 0) return 0.0;
            return std::min(1.0, r);
        }

        // ========================= WENO重构实现 =========================

        double AdvectionDiffusionSolver::wenoReconstruction(const std::vector<double>& stencil, bool left_biased) const {
            if (stencil.size() != 5) return 0.0;

            const double eps = 1e-6;

            // 计算光滑性指标
            std::vector<double> beta(3);
            beta[0] = 13.0/12.0 * std::pow(stencil[0] - 2*stencil[1] + stencil[2], 2) +
                      0.25 * std::pow(stencil[0] - 4*stencil[1] + 3*stencil[2], 2);
            beta[1] = 13.0/12.0 * std::pow(stencil[1] - 2*stencil[2] + stencil[3], 2) +
                      0.25 * std::pow(stencil[1] - stencil[3], 2);
            beta[2] = 13.0/12.0 * std::pow(stencil[2] - 2*stencil[3] + stencil[4], 2) +
                      0.25 * std::pow(3*stencil[2] - 4*stencil[3] + stencil[4], 2);

            // 理想权重
            std::vector<double> gamma(3);
            if (left_biased) {
                gamma[0] = 0.1; gamma[1] = 0.6; gamma[2] = 0.3;
            } else {
                gamma[0] = 0.3; gamma[1] = 0.6; gamma[2] = 0.1;
            }

            // 计算非线性权重
            std::vector<double> alpha(3);
            for (int k = 0; k < 3; ++k) {
                alpha[k] = gamma[k] / std::pow(eps + beta[k], 2);
            }

            double sum_alpha = alpha[0] + alpha[1] + alpha[2];
            std::vector<double> omega(3);
            for (int k = 0; k < 3; ++k) {
                omega[k] = alpha[k] / sum_alpha;
            }

            // 计算重构值
            double q0 = (2*stencil[0] - 7*stencil[1] + 11*stencil[2]) / 6.0;
            double q1 = (-stencil[1] + 5*stencil[2] + 2*stencil[3]) / 6.0;
            double q2 = (2*stencil[2] + 5*stencil[3] - stencil[4]) / 6.0;

            return omega[0] * q0 + omega[1] * q1 + omega[2] * q2;
        }

        std::vector<double> AdvectionDiffusionSolver::wenoWeights(const std::vector<double>& beta_values) const {
            const double eps = 1e-6;
            std::vector<double> gamma = {0.1, 0.6, 0.3}; // 理想权重
            std::vector<double> alpha(3);

            for (int k = 0; k < 3; ++k) {
                alpha[k] = gamma[k] / std::pow(eps + beta_values[k], 2);
            }

            double sum_alpha = alpha[0] + alpha[1] + alpha[2];
            std::vector<double> omega(3);
            for (int k = 0; k < 3; ++k) {
                omega[k] = alpha[k] / sum_alpha;
            }

            return omega;
        }

        // ========================= 边界条件应用 =========================

        void AdvectionDiffusionSolver::applyBoundaryConditions(Eigen::MatrixXd& field) {
            auto dims = grid_->getDimensions();

            for (const auto& bc : boundary_conditions_) {
                int boundary_id = bc.first;
                BoundaryType type = bc.second.first;
                double value = bc.second.second;

                switch (type) {
                    case BoundaryType::DIRICHLET:
                        applyDirichletBC(field, boundary_id, value);
                        break;
                    case BoundaryType::NEUMANN:
                        applyNeumannBC(field, boundary_id, value);
                        break;
                    case BoundaryType::ROBIN:
                        applyRobinBC(field, boundary_id, value);
                        break;
                    case BoundaryType::PERIODIC:
                        applyPeriodicBC(field);
                        break;
                    case BoundaryType::OUTFLOW:
                        applyOutflowBC(field, boundary_id);
                        break;
                    default:
                        break;
                }
            }
        }

        void AdvectionDiffusionSolver::applyDirichletBC(Eigen::MatrixXd& field, int boundary_id, double value) {
            auto dims = grid_->getDimensions();

            // 简化处理：假设边界ID对应边界位置
            switch (boundary_id) {
                case 0: // 左边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, 0) = value;
                    }
                    break;
                case 1: // 右边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, dims[0]-1) = value;
                    }
                    break;
                case 2: // 下边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(0, i) = value;
                    }
                    break;
                case 3: // 上边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(dims[1]-1, i) = value;
                    }
                    break;
            }
        }

        void AdvectionDiffusionSolver::applyNeumannBC(Eigen::MatrixXd& field, int boundary_id, double flux) {
            auto dims = grid_->getDimensions();
            auto spacing = grid_->getSpacing();

            switch (boundary_id) {
                case 0: // 左边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, 0) = field(j, 1) - flux * spacing[0];
                    }
                    break;
                case 1: // 右边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, dims[0]-1) = field(j, dims[0]-2) + flux * spacing[0];
                    }
                    break;
                case 2: // 下边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(0, i) = field(1, i) - flux * spacing[1];
                    }
                    break;
                case 3: // 上边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(dims[1]-1, i) = field(dims[1]-2, i) + flux * spacing[1];
                    }
                    break;
            }
        }

        void AdvectionDiffusionSolver::applyRobinBC(Eigen::MatrixXd& field, int boundary_id, double value) {
            // Robin边界条件: αu + β∂u/∂n = γ
            // 这里简化为混合边界条件的实现
            auto dims = grid_->getDimensions();
            auto spacing = grid_->getSpacing();

            double alpha = 1.0; // 可以设为参数
            double beta = 1.0;
            double gamma = value;

            switch (boundary_id) {
                case 0: // 左边界
                    for (int j = 0; j < dims[1]; ++j) {
                        double du_dn = (field(j, 1) - field(j, 0)) / spacing[0];
                        field(j, 0) = (gamma - beta * du_dn) / alpha;
                    }
                    break;
                case 1: // 右边界
                    for (int j = 0; j < dims[1]; ++j) {
                        double du_dn = (field(j, dims[0]-1) - field(j, dims[0]-2)) / spacing[0];
                        field(j, dims[0]-1) = (gamma - beta * du_dn) / alpha;
                    }
                    break;
                    // 其他边界类似处理
            }
        }

        void AdvectionDiffusionSolver::applyPeriodicBC(Eigen::MatrixXd& field) {
            auto dims = grid_->getDimensions();

            // X方向周期边界
            for (int j = 0; j < dims[1]; ++j) {
                field(j, 0) = field(j, dims[0]-2);
                field(j, dims[0]-1) = field(j, 1);
            }

            // Y方向周期边界
            for (int i = 0; i < dims[0]; ++i) {
                field(0, i) = field(dims[1]-2, i);
                field(dims[1]-1, i) = field(1, i);
            }
        }

        void AdvectionDiffusionSolver::applyOutflowBC(Eigen::MatrixXd& field, int boundary_id) {
            // 流出边界条件：零梯度
            auto dims = grid_->getDimensions();

            switch (boundary_id) {
                case 0: // 左边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, 0) = field(j, 1);
                    }
                    break;
                case 1: // 右边界
                    for (int j = 0; j < dims[1]; ++j) {
                        field(j, dims[0]-1) = field(j, dims[0]-2);
                    }
                    break;
                case 2: // 下边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(0, i) = field(1, i);
                    }
                    break;
                case 3: // 上边界
                    for (int i = 0; i < dims[0]; ++i) {
                        field(dims[1]-1, i) = field(dims[1]-2, i);
                    }
                    break;
            }
        }

        // ========================= 矩阵设置和线性系统求解 =========================

        void AdvectionDiffusionSolver::setupImplicitMatrix(Eigen::SparseMatrix<double>& A,
                                                           Eigen::VectorXd& b, double dt) {
            auto dims = grid_->getDimensions();
            int n = dims[0] * dims[1];

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(5 * n); // 每个点最多5个非零元素

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];
            double dx2 = dx * dx;
            double dy2 = dy * dy;

            double alpha_x = diffusion_coefficient_ * dt / dx2;
            double alpha_y = diffusion_coefficient_ * dt / dy2;

            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    int idx = j * dims[0] + i;

                    if (isInternalPoint(i, j)) {
                        // 内部点：隐式扩散
                        double center_coeff = 1.0 + 2.0 * (alpha_x + alpha_y);

                        triplets.emplace_back(idx, idx, center_coeff);
                        triplets.emplace_back(idx, idx - 1, -alpha_x);        // 左
                        triplets.emplace_back(idx, idx + 1, -alpha_x);        // 右
                        triplets.emplace_back(idx, idx - dims[0], -alpha_y);  // 下
                        triplets.emplace_back(idx, idx + dims[0], -alpha_y);  // 上

                        // 右端项包含当前解和显式平流项
                        Eigen::MatrixXd advection_explicit = computeAdvection(current_solution_, dt);
                        b(idx) = current_solution_(j, i) - dt * advection_explicit(j, i);

                        if (has_source_term_) {
                            b(idx) += dt * source_term_(j, i);
                        }
                    } else {
                        // 边界点
                        triplets.emplace_back(idx, idx, 1.0);
                        b(idx) = current_solution_(j, i);
                    }
                }
            }

            A.setFromTriplets(triplets.begin(), triplets.end());
        }

        void AdvectionDiffusionSolver::setupCrankNicolsonMatrix(Eigen::SparseMatrix<double>& A,
                                                                Eigen::VectorXd& b, double dt) {
            auto dims = grid_->getDimensions();
            int n = dims[0] * dims[1];

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(5 * n);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];
            double dx2 = dx * dx;
            double dy2 = dy * dy;

            double alpha_x = diffusion_coefficient_ * dt / (2.0 * dx2);
            double alpha_y = diffusion_coefficient_ * dt / (2.0 * dy2);

            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    int idx = j * dims[0] + i;

                    if (isInternalPoint(i, j)) {
                        // 左端矩阵：(I - dt/2 * L)
                        double center_coeff = 1.0 + 2.0 * (alpha_x + alpha_y);

                        triplets.emplace_back(idx, idx, center_coeff);
                        triplets.emplace_back(idx, idx - 1, -alpha_x);
                        triplets.emplace_back(idx, idx + 1, -alpha_x);
                        triplets.emplace_back(idx, idx - dims[0], -alpha_y);
                        triplets.emplace_back(idx, idx + dims[0], -alpha_y);

                        // 右端向量：(I + dt/2 * L) * u^n - dt * 平流项
                        double rhs = current_solution_(j, i);

                        // 添加显式扩散项
                        if (i > 0 && i < dims[0] - 1 && j > 0 && j < dims[1] - 1) {
                            double d2udx2 = (current_solution_(j, i+1) - 2*current_solution_(j, i) + current_solution_(j, i-1)) / dx2;
                            double d2udy2 = (current_solution_(j+1, i) - 2*current_solution_(j, i) + current_solution_(j-1, i)) / dy2;
                            rhs += (dt/2.0) * diffusion_coefficient_ * (d2udx2 + d2udy2);
                        }

                        // 减去平流项
                        Eigen::MatrixXd advection_explicit = computeAdvection(current_solution_, dt);
                        rhs -= dt * advection_explicit(j, i);

                        if (has_source_term_) {
                            rhs += dt * source_term_(j, i);
                        }

                        b(idx) = rhs;
                    } else {
                        // 边界点
                        triplets.emplace_back(idx, idx, 1.0);
                        b(idx) = current_solution_(j, i);
                    }
                }
            }

            A.setFromTriplets(triplets.begin(), triplets.end());
        }

        void AdvectionDiffusionSolver::solveLinearSystem(const Eigen::SparseMatrix<double>& A,
                                                         const Eigen::VectorXd& b,
                                                         Eigen::VectorXd& x) {
            // 使用共轭梯度法求解稀疏线性系统
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
            cg.setMaxIterations(1000);
            cg.setTolerance(1e-6);
            cg.compute(A);

            if (cg.info() != Eigen::Success) {
                std::cerr << "警告：矩阵分解失败" << std::endl;
                return;
            }

            x = cg.solve(b);

            if (cg.info() != Eigen::Success) {
                std::cerr << "警告：线性系统求解失败，迭代次数：" << cg.iterations()
                          << "，估计误差：" << cg.error() << std::endl;
            }
        }

        // ========================= 自适应时间步长和稳定性检查 =========================

        double AdvectionDiffusionSolver::computeOptimalTimeStep() const {
            if (!has_velocity_field_) return previous_dt_;

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            double max_velocity = 0.0;
            auto dims = grid_->getDimensions();

            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    double vel_mag = std::sqrt(u_field(j, i) * u_field(j, i) +
                                               v_field(j, i) * v_field(j, i));
                    max_velocity = std::max(max_velocity, vel_mag);
                }
            }

            // CFL条件
            double dt_cfl = cfl_target_ * std::min(dx, dy) / (max_velocity + 1e-15);

            // 扩散稳定性条件
            double dt_diff = 0.25 * std::min(dx*dx, dy*dy) / (diffusion_coefficient_ + 1e-15);

            double optimal_dt = std::min(dt_cfl, dt_diff);

            // 限制时间步长变化率
            if (previous_dt_ > 0.0) {
                optimal_dt = std::min(optimal_dt, 1.5 * previous_dt_);
                optimal_dt = std::max(optimal_dt, 0.5 * previous_dt_);
            }

            previous_dt_ = optimal_dt;
            return optimal_dt;
        }

        void AdvectionDiffusionSolver::enableAdaptiveTimeStep(bool enable, double cfl_target) {
            adaptive_timestep_ = enable;
            cfl_target_ = cfl_target;
        }

        double AdvectionDiffusionSolver::computePecletNumber() const {
            if (!has_velocity_field_) return 0.0;

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double L = std::min(spacing[0], spacing[1]);

            double max_velocity = 0.0;
            auto dims = grid_->getDimensions();

            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    double vel_mag = std::sqrt(u_field(j, i) * u_field(j, i) +
                                               v_field(j, i) * v_field(j, i));
                    max_velocity = std::max(max_velocity, vel_mag);
                }
            }

            return max_velocity * L / (diffusion_coefficient_ + 1e-15);
        }

        double AdvectionDiffusionSolver::computeCourantNumber(double dt) const {
            if (!has_velocity_field_) return 0.0;

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            double max_cfl = 0.0;
            auto dims = grid_->getDimensions();

            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    double cfl_x = std::abs(u_field(j, i)) * dt / dx;
                    double cfl_y = std::abs(v_field(j, i)) * dt / dy;
                    max_cfl = std::max(max_cfl, std::max(cfl_x, cfl_y));
                }
            }

            return max_cfl;
        }

        void AdvectionDiffusionSolver::analyzeNumericalStability() const {
            std::cout << "\n=== 数值稳定性分析 ===" << std::endl;

            double dt_optimal = computeOptimalTimeStep();
            std::cout << "建议的最优时间步长: " << dt_optimal << " s" << std::endl;

            double peclet = computePecletNumber();
            std::cout << "Peclet数: " << peclet << std::endl;
            if (peclet > 2.0) {
                std::cout << "警告: Peclet数较大，建议使用迎风格式或TVD格式" << std::endl;
            }

            if (has_velocity_field_) {
                double cfl = computeCourantNumber(dt_optimal);
                std::cout << "Courant数: " << cfl << std::endl;
                if (cfl > 1.0) {
                    std::cout << "警告: CFL条件不满足，可能导致数值不稳定" << std::endl;
                }
            }

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];
            double diff_num = diffusion_coefficient_ * dt_optimal / std::min(dx*dx, dy*dy);
            std::cout << "扩散数: " << diff_num << std::endl;
            if (diff_num > 0.5) {
                std::cout << "警告: 扩散数过大，显式格式可能不稳定" << std::endl;
            }
        }

        void AdvectionDiffusionSolver::checkNumericalStability(double dt) const {
            double cfl = computeCourantNumber(dt);
            auto spacing = grid_->getSpacing();
            double diff_num = diffusion_coefficient_ * dt /
                              std::min(spacing[0] * spacing[0], spacing[1] * spacing[1]);

            if (cfl > 1.0) {
                std::cerr << "警告：CFL数 " << cfl << " 超过稳定性限制" << std::endl;
            }

            if (diff_num > 0.5 && time_integration_ == TimeIntegration::EXPLICIT_EULER) {
                std::cerr << "警告：扩散数 " << diff_num << " 超过显式格式稳定性限制" << std::endl;
            }
        }

        void AdvectionDiffusionSolver::detectOscillations() const {
            auto dims = grid_->getDimensions();
            int oscillation_count = 0;

            for (int i = 1; i < dims[0] - 1; ++i) {
                for (int j = 1; j < dims[1] - 1; ++j) {
                    double center = current_solution_(j, i);
                    double left = current_solution_(j, i-1);
                    double right = current_solution_(j, i+1);
                    double down = current_solution_(j-1, i);
                    double up = current_solution_(j+1, i);

                    // 检测局部极值
                    if ((center > left && center > right && center > down && center > up) ||
                        (center < left && center < right && center < down && center < up)) {
                        oscillation_count++;
                    }
                }
            }

            double oscillation_ratio = static_cast<double>(oscillation_count) / ((dims[0]-2) * (dims[1]-2));
            if (oscillation_ratio > 0.1) {
                std::cerr << "警告：检测到数值振荡，振荡点比例：" << oscillation_ratio * 100 << "%" << std::endl;
            }
        }

        // ========================= 质量守恒和结果分析 =========================

        bool AdvectionDiffusionSolver::checkMassConservation(double tolerance) const {
            static double initial_mass = -1.0;

            double current_mass = getTotalMass();

            if (initial_mass < 0.0) {
                initial_mass = current_mass;
                return true;
            }

            double mass_change = std::abs(current_mass - initial_mass) / (initial_mass + 1e-15);
            return mass_change <= tolerance;
        }

        double AdvectionDiffusionSolver::computeMassBalance() const {
            double current_mass = getTotalMass();

            // 计算通过边界的质量流量
            double boundary_flux = 0.0;
            auto dims = grid_->getDimensions();
            auto spacing = grid_->getSpacing();

            // 简化计算边界通量
            for (int j = 0; j < dims[1]; ++j) {
                // 左右边界
                if (has_velocity_field_) {
                    const auto& u_field = grid_->getField2D(u_field_name_);
                    boundary_flux += u_field(j, 0) * current_solution_(j, 0) * spacing[1];           // 左边界流入
                    boundary_flux -= u_field(j, dims[0]-1) * current_solution_(j, dims[0]-1) * spacing[1]; // 右边界流出
                }
            }

            for (int i = 0; i < dims[0]; ++i) {
                // 上下边界
                if (has_velocity_field_) {
                    const auto& v_field = grid_->getField2D(v_field_name_);
                    boundary_flux += v_field(0, i) * current_solution_(0, i) * spacing[0];           // 下边界流入
                    boundary_flux -= v_field(dims[1]-1, i) * current_solution_(dims[1]-1, i) * spacing[0]; // 上边界流出
                }
            }

            return boundary_flux;
        }

        double AdvectionDiffusionSolver::getTotalMass() const {
            auto spacing = grid_->getSpacing();
            double cell_area = spacing[0] * spacing[1];
            return current_solution_.sum() * cell_area;
        }

        double AdvectionDiffusionSolver::getMaxConcentration() const {
            return current_solution_.maxCoeff();
        }

        // ========================= 高级求解功能 =========================

        void AdvectionDiffusionSolver::solveToSteadyState(double tolerance, int max_iterations) {
            double dt = computeOptimalTimeStep();

            std::cout << "开始稳态求解，容限：" << tolerance << "，最大迭代次数：" << max_iterations << std::endl;

            for (int iter = 0; iter < max_iterations; ++iter) {
                Eigen::MatrixXd old_solution = current_solution_;
                solve(dt);

                // 检查收敛性
                double residual = (current_solution_ - old_solution).norm() / (old_solution.norm() + 1e-15);

                if (iter % 100 == 0) {
                    std::cout << "迭代 " << iter << "，残差：" << residual << std::endl;
                }

                if (residual < tolerance) {
                    std::cout << "在第 " << iter << " 次迭代后达到稳态，残差：" << residual << std::endl;
                    break;
                }

                if (iter == max_iterations - 1) {
                    std::cout << "警告：未在最大迭代次数内收敛，最终残差：" << residual << std::endl;
                }
            }
        }

        void AdvectionDiffusionSolver::solveTimeSequence(double t_start, double t_end, double dt,
                                                         const std::string& output_prefix) {
            double current_time = t_start;
            int step_count = 0;

            std::cout << "开始时间序列求解，从 t=" << t_start << " 到 t=" << t_end
                      << "，时间步长：" << dt << std::endl;

            while (current_time < t_end) {
                double actual_dt = std::min(dt, t_end - current_time);

                if (adaptive_timestep_) {
                    actual_dt = computeOptimalTimeStep();
                    actual_dt = std::min(actual_dt, t_end - current_time);
                }

                solve(actual_dt);
                current_time += actual_dt;
                step_count++;

                // 输出中间结果
                if (!output_prefix.empty() && step_count % 10 == 0) {
                    std::string filename = output_prefix + "_t" + std::to_string(current_time) + ".dat";
                    saveResult(filename);
                }

                // 质量守恒检查
                if (!checkMassConservation(1e-6)) {
                    std::cerr << "警告：时间 t=" << current_time << " 时质量不守恒" << std::endl;
                }

                if (step_count % 100 == 0) {
                    std::cout << "完成步数：" << step_count << "，当前时间：" << current_time
                              << "，最大浓度：" << getMaxConcentration() << std::endl;
                }
            }

            std::cout << "时间序列求解完成，总步数：" << step_count << std::endl;
        }

        // ========================= 辅助方法 =========================

        bool AdvectionDiffusionSolver::isInternalPoint(int i, int j) const {
            auto dims = grid_->getDimensions();
            return i > 0 && i < dims[0] - 1 && j > 0 && j < dims[1] - 1;
        }

        double AdvectionDiffusionSolver::computeLocalCourantNumber(int i, int j, double dt) const {
            if (!has_velocity_field_) return 0.0;

            const auto& u_field = grid_->getField2D(u_field_name_);
            const auto& v_field = grid_->getField2D(v_field_name_);

            auto spacing = grid_->getSpacing();
            double dx = spacing[0];
            double dy = spacing[1];

            double cfl_x = std::abs(u_field(j, i)) * dt / dx;
            double cfl_y = std::abs(v_field(j, i)) * dt / dy;

            return std::max(cfl_x, cfl_y);
        }

        void AdvectionDiffusionSolver::updateSolutionHistory() {
            // 限制历史记录的大小
            const size_t max_history_size = 10;

            solution_history_.push_back(current_solution_);
            if (solution_history_.size() > max_history_size) {
                solution_history_.erase(solution_history_.begin());
            }
        }

        double AdvectionDiffusionSolver::safeDivision(double numerator, double denominator) const {
            if (std::abs(denominator) < 1e-15) return 0.0;
            return numerator / denominator;
        }

        void AdvectionDiffusionSolver::saveResult(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "无法打开文件：" << filename << std::endl;
                return;
            }

            auto dims = grid_->getDimensions();
            auto spacing = grid_->getSpacing();

            file << "# X Y Concentration\n";
            for (int i = 0; i < dims[0]; ++i) {
                for (int j = 0; j < dims[1]; ++j) {
                    double x = i * spacing[0];
                    double y = j * spacing[1];
                    file << x << " " << y << " " << current_solution_(j, i) << "\n";
                }
            }

            file.close();
            std::cout << "结果已保存到：" << filename << std::endl;
        }

        // ========================= MultiComponentSolver 实现 =========================

        MultiComponentSolver::MultiComponentSolver(std::shared_ptr<Data::GridDataStructure> grid,
                                                   int num_components)
                : AdvectionDiffusionSolver(grid), num_components_(num_components) {

            component_names_.resize(num_components_);
            diffusivities_.resize(num_components_, 1e-6);
            decay_rates_.resize(num_components_, 0.0);

            auto dims = grid_->getDimensions();
            component_solutions_.resize(num_components_);
            for (int i = 0; i < num_components_; ++i) {
                component_solutions_[i] = Eigen::MatrixXd::Zero(dims[1], dims[0]);
                component_names_[i] = "Component_" + std::to_string(i);
            }
        }

        void MultiComponentSolver::setComponentName(int component_id, const std::string& name) {
            if (component_id >= 0 && component_id < num_components_) {
                component_names_[component_id] = name;
            }
        }

        void MultiComponentSolver::setComponentDiffusivity(int component_id, double diffusivity) {
            if (component_id >= 0 && component_id < num_components_ && diffusivity >= 0.0) {
                diffusivities_[component_id] = diffusivity;
            }
        }

        void MultiComponentSolver::setComponentInitialCondition(int component_id, const Eigen::MatrixXd& initial) {
            if (component_id >= 0 && component_id < num_components_) {
                auto dims = grid_->getDimensions();
                if (initial.rows() == dims[1] && initial.cols() == dims[0]) {
                    component_solutions_[component_id] = initial;
                }
            }
        }

        void MultiComponentSolver::addReaction(int reactant1, int reactant2, int product, double rate_constant) {
            if (reactant1 >= 0 && reactant1 < num_components_ &&
                reactant2 >= 0 && reactant2 < num_components_ &&
                product >= 0 && product < num_components_ &&
                rate_constant >= 0.0) {
                reactions_.push_back({reactant1, reactant2, product, rate_constant});
            }
        }

        void MultiComponentSolver::setDecayRate(int component_id, double decay_rate) {
            if (component_id >= 0 && component_id < num_components_ && decay_rate >= 0.0) {
                decay_rates_[component_id] = decay_rate;
            }
        }

        void MultiComponentSolver::solveMultiComponent(double dt) {
            // 为每个组分求解传输方程
            for (int comp = 0; comp < num_components_; ++comp) {
                // 设置当前组分的扩散系数
                setDiffusionCoefficient(diffusivities_[comp]);

                // 设置当前解为组分浓度
                current_solution_ = component_solutions_[comp];

                // 求解单个组分的传输
                solveSingleTimeStep(dt);

                // 保存求解结果
                component_solutions_[comp] = current_solution_;
            }

            // 处理化学反应
            if (!reactions_.empty()) {
                solveReactionSystem(dt);
            }

            // 处理衰变
            for (int comp = 0; comp < num_components_; ++comp) {
                if (decay_rates_[comp] > 0.0) {
                    component_solutions_[comp] *= std::exp(-decay_rates_[comp] * dt);
                }
            }

            // 确保所有浓度非负
            for (int comp = 0; comp < num_components_; ++comp) {
                component_solutions_[comp] = component_solutions_[comp].cwiseMax(0.0);
            }
        }

        void MultiComponentSolver::computeReactionTerms(std::vector<Eigen::MatrixXd>& reaction_rates) {
            auto dims = grid_->getDimensions();

            // 初始化反应速率
            for (int comp = 0; comp < num_components_; ++comp) {
                reaction_rates[comp] = Eigen::MatrixXd::Zero(dims[1], dims[0]);
            }

            // 计算每个反应的贡献
            for (const auto& reaction : reactions_) {
                for (int i = 0; i < dims[0]; ++i) {
                    for (int j = 0; j < dims[1]; ++j) {
                        double c1 = component_solutions_[reaction.reactant1](j, i);
                        double c2 = (reaction.reactant1 == reaction.reactant2) ?
                                    c1 : component_solutions_[reaction.reactant2](j, i);

                        double rate = reaction.rate_constant * c1 * c2;

                        // 反应物浓度减少
                        reaction_rates[reaction.reactant1](j, i) -= rate;
                        if (reaction.reactant1 != reaction.reactant2) {
                            reaction_rates[reaction.reactant2](j, i) -= rate;
                        }

                        // 产物浓度增加
                        reaction_rates[reaction.product](j, i) += rate;
                    }
                }
            }
        }

        void MultiComponentSolver::solveReactionSystem(double dt) {
            std::vector<Eigen::MatrixXd> reaction_rates(num_components_);
            computeReactionTerms(reaction_rates);

            // 使用显式欧拉方法更新组分浓度
            for (int comp = 0; comp < num_components_; ++comp) {
                component_solutions_[comp] += dt * reaction_rates[comp];
            }
        }

        const std::vector<Eigen::MatrixXd>& MultiComponentSolver::getComponentSolutions() const {
            return component_solutions_;
        }

        double MultiComponentSolver::getComponentMass(int component_id) const {
            if (component_id < 0 || component_id >= num_components_) return 0.0;

            auto spacing = grid_->getSpacing();
            double cell_area = spacing[0] * spacing[1];
            return component_solutions_[component_id].sum() * cell_area;
        }

        double MultiComponentSolver::getTotalSystemMass() const {
            double total_mass = 0.0;
            for (int comp = 0; comp < num_components_; ++comp) {
                total_mass += getComponentMass(comp);
            }
            return total_mass;
        }

        void MultiComponentSolver::printComponentSummary() const {
            std::cout << "\n=== 多组分系统摘要 ===" << std::endl;
            std::cout << "组分数量: " << num_components_ << std::endl;

            for (int comp = 0; comp < num_components_; ++comp) {
                double mass = getComponentMass(comp);
                double max_conc = component_solutions_[comp].maxCoeff();
                double min_conc = component_solutions_[comp].minCoeff();

                std::cout << "组分 " << comp << " (" << component_names_[comp] << "):" << std::endl;
                std::cout << "  总质量: " << mass << std::endl;
                std::cout << "  最大浓度: " << max_conc << std::endl;
                std::cout << "  最小浓度: " << min_conc << std::endl;
                std::cout << "  扩散系数: " << diffusivities_[comp] << std::endl;
                std::cout << "  衰变速率: " << decay_rates_[comp] << std::endl;
            }

            std::cout << "总系统质量: " << getTotalSystemMass() << std::endl;
            std::cout << "反应数量: " << reactions_.size() << std::endl;
        }

        void MultiComponentSolver::saveComponentResults(const std::string& filename_prefix) const {
            for (int comp = 0; comp < num_components_; ++comp) {
                std::string filename = filename_prefix + "_" + component_names_[comp] + ".dat";
                std::ofstream file(filename);

                if (!file.is_open()) {
                    std::cerr << "无法打开文件：" << filename << std::endl;
                    continue;
                }

                auto dims = grid_->getDimensions();
                auto spacing = grid_->getSpacing();

                file << "# X Y " << component_names_[comp] << "_Concentration\n";
                for (int i = 0; i < dims[0]; ++i) {
                    for (int j = 0; j < dims[1]; ++j) {
                        double x = i * spacing[0];
                        double y = j * spacing[1];
                        file << x << " " << y << " " << component_solutions_[comp](j, i) << "\n";
                    }
                }

                file.close();
            }

            std::cout << "所有组分结果已保存，前缀：" << filename_prefix << std::endl;
        }

    } // namespace Core
} // namespace OceanSim