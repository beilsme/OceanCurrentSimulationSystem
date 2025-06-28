// src/core/CurrentFieldSolver.cpp
#include "core/CurrentFieldSolver.h"
#include "utils/MathUtils.h"
#include "utils/Logger.h"
#include <cmath>
#include <algorithm>

namespace OceanSim {
    namespace Core {

// OceanState 构造函数
        CurrentFieldSolver::OceanState::OceanState(int nx, int ny, int nz) {
            resize(nx, ny, nz);
        }

        void CurrentFieldSolver::OceanState::resize(int nx, int ny, int nz) {
            u.resize(nx, ny);
            v.resize(nx, ny);
            w.resize(nx, ny);
            temperature.resize(nx, ny);
            salinity.resize(nx, ny);
            density.resize(nx, ny);
            pressure.resize(nx, ny);
            ssh.resize(nx, ny);

            // 初始化为零
            u.setZero();
            v.setZero();
            w.setZero();
            temperature.setZero();
            salinity.setZero();
            density.setConstant(1025.0); // 默认海水密度
            pressure.setZero();
            ssh.setZero();
        }

        CurrentFieldSolver::CurrentFieldSolver(
                std::shared_ptr<Data::GridDataStructure> grid,
                const PhysicalParameters& params)
                : grid_(grid), params_(params) {

            // 获取网格尺寸
            auto dimensions = grid_->getDimensions();
            nx_ = dimensions[0];
            ny_ = dimensions[1];
            nz_ = dimensions[2];

            auto spacing = grid_->getSpacing();
            dx_ = spacing[0];
            dy_ = spacing[1];

            // 设置垂直层厚度（σ坐标系统）
            dz_.resize(nz_);
            for (int k = 0; k < nz_; ++k) {
                dz_[k] = 1.0 / nz_; // 等厚度层，实际应用中可变
            }

            // 初始化状态
            current_state_.resize(nx_, ny_, nz_);
            previous_state_.resize(nx_, ny_, nz_);

            // 创建有限差分求解器
            fd_solver_ = std::make_shared<Algorithms::FiniteDifferenceSolver>(
                    nx_, ny_, nz_, dx_, dy_, dz_);

            Utils::Logger::info("CurrentFieldSolver initialized: {}x{}x{}", nx_, ny_, nz_);
        }

        void CurrentFieldSolver::initialize(const OceanState& initial_state) {
            current_state_ = initial_state;
            previous_state_ = initial_state;

            // 计算初始密度场
            computeDensity();

            // 计算初始压力场
            computeHydrostaticPressure();

            Utils::Logger::info("Ocean state initialized");
        }

        void CurrentFieldSolver::setBottomTopography(const Eigen::MatrixXd& bottom_depth) {
            bottom_depth_ = bottom_depth;
            has_topography_ = true;
            Utils::Logger::info("Bottom topography set");
        }

        void CurrentFieldSolver::setWindStress(const Eigen::MatrixXd& tau_x,
                                               const Eigen::MatrixXd& tau_y) {
            wind_stress_x_ = tau_x;
            wind_stress_y_ = tau_y;
            has_wind_forcing_ = true;
            Utils::Logger::info("Wind stress forcing set");
        }

        void CurrentFieldSolver::stepForward(double dt) {
            // 保存上一时刻状态
            previous_state_ = current_state_;

            // 检查CFL条件
            double max_dt = computeCFLTimeStep();
            if (dt > max_dt) {
                Utils::Logger::warning("Time step {} exceeds CFL limit {}", dt, max_dt);
            }

            // 时间分裂算法：先正压后斜压
            stepForwardBarotropic(dt);
            stepForwardBaroclinic(dt);

            // 应用边界条件
            applyBoundaryConditions();

            // 质量守恒检查
            if (!checkMassConservation()) {
                Utils::Logger::warning("Mass conservation violated");
            }
        }

        void CurrentFieldSolver::stepForwardBarotropic(double dt) {
            // 正压模式：快速重力波
            auto& u = current_state_.u;
            auto& v = current_state_.v;
            auto& ssh = current_state_.ssh;

            // 临时存储
            Eigen::MatrixXd u_new = u;
            Eigen::MatrixXd v_new = v;
            Eigen::MatrixXd ssh_new = ssh;

            // 连续方程：∂η/∂t + ∇·(H*u) = 0
            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    double H = has_topography_ ? bottom_depth_(i,j) : 1000.0; // 默认深度

                    double div_flux = ((H * u(i+1,j)) - (H * u(i-1,j))) / (2.0 * dx_) +
                                      ((H * v(i,j+1)) - (H * v(i,j-1))) / (2.0 * dy_);

                    ssh_new(i,j) = ssh(i,j) - dt * div_flux;
                }
            }

            // 动量方程
            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    // 压力梯度项
                    double deta_dx = (ssh(i+1,j) - ssh(i-1,j)) / (2.0 * dx_);
                    double deta_dy = (ssh(i,j+1) - ssh(i,j-1)) / (2.0 * dy_);

                    // 科里奥利力
                    double f = params_.coriolis_f; // 简化为常数

                    // 风应力强迫
                    double tau_x = has_wind_forcing_ ? wind_stress_x_(i,j) / params_.reference_density : 0.0;
                    double tau_y = has_wind_forcing_ ? wind_stress_y_(i,j) / params_.reference_density : 0.0;

                    // u分量
                    u_new(i,j) = u(i,j) + dt * (
                            f * v(i,j) - params_.gravity * deta_dx + tau_x
                    );

                    // v分量
                    v_new(i,j) = v(i,j) + dt * (
                            -f * u(i,j) - params_.gravity * deta_dy + tau_y
                    );
                }
            }

            current_state_.u = u_new;
            current_state_.v = v_new;
            current_state_.ssh = ssh_new;
        }

        void CurrentFieldSolver::stepForwardBaroclinic(double dt) {
            // 斜压模式：温度、盐度、密度演化

            // 求解温度方程
            solveTemperatureEquation(dt);

            // 求解盐度方程
            solveSalinityEquation(dt);

            // 更新密度
            computeDensity();

            // 更新压力场
            computeHydrostaticPressure();
        }

        void CurrentFieldSolver::solveTemperatureEquation(double dt) {
            auto& T = current_state_.temperature;
            auto& u = current_state_.u;
            auto& v = current_state_.v;
            auto& w = current_state_.w;

            Eigen::MatrixXd T_new = T;

            // 温度平流扩散方程: ∂T/∂t + u·∇T = ∇·(K∇T)
            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    // 平流项
                    double advection =
                            u(i,j) * (T(i+1,j) - T(i-1,j)) / (2.0 * dx_) +
                            v(i,j) * (T(i,j+1) - T(i,j-1)) / (2.0 * dy_);

                    // 水平扩散项
                    double diffusion_h = params_.diffusivity_h * (
                            (T(i+1,j) - 2*T(i,j) + T(i-1,j)) / (dx_ * dx_) +
                            (T(i,j+1) - 2*T(i,j) + T(i,j-1)) / (dy_ * dy_)
                    );

                    T_new(i,j) = T(i,j) + dt * (-advection + diffusion_h);
                }
            }

            current_state_.temperature = T_new;
        }

        void CurrentFieldSolver::solveSalinityEquation(double dt) {
            auto& S = current_state_.salinity;
            auto& u = current_state_.u;
            auto& v = current_state_.v;

            Eigen::MatrixXd S_new = S;

            // 盐度平流扩散方程: ∂S/∂t + u·∇S = ∇·(K∇S)
            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    // 平流项
                    double advection =
                            u(i,j) * (S(i+1,j) - S(i-1,j)) / (2.0 * dx_) +
                            v(i,j) * (S(i,j+1) - S(i,j-1)) / (2.0 * dy_);

                    // 水平扩散项
                    double diffusion_h = params_.diffusivity_h * (
                            (S(i+1,j) - 2*S(i,j) + S(i-1,j)) / (dx_ * dx_) +
                            (S(i,j+1) - 2*S(i,j) + S(i,j-1)) / (dy_ * dy_)
                    );

                    S_new(i,j) = S(i,j) + dt * (-advection + diffusion_h);
                }
            }

            current_state_.salinity = S_new;
        }

        void CurrentFieldSolver::computeDensity() {
            auto& T = current_state_.temperature;
            auto& S = current_state_.salinity;
            auto& rho = current_state_.density;
            auto& p = current_state_.pressure;

            // 使用简化的海水状态方程
            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    rho(i,j) = computeSeawaterDensity(T(i,j), S(i,j), p(i,j));
                }
            }
        }

        double CurrentFieldSolver::computeSeawaterDensity(double temperature,
                                                          double salinity,
                                                          double pressure) const {
            // UNESCO海水状态方程的简化版本
            // 实际应用中应使用更精确的TEOS-10状态方程

            double T = temperature;
            double S = salinity;
            double P = pressure * 1e-5; // 转换为bar

            // 纯水密度
            double rho_w = 999.842594 + 6.793952e-2*T - 9.095290e-3*T*T
                           + 1.001685e-4*T*T*T - 1.120083e-6*T*T*T*T
                           + 6.536332e-9*T*T*T*T*T;

            // 盐度修正
            double A = 8.24493e-1 - 4.0899e-3*T + 7.6438e-5*T*T
                       - 8.2467e-7*T*T*T + 5.3875e-9*T*T*T*T;
            double B = -5.72466e-3 + 1.0227e-4*T - 1.6546e-6*T*T;
            double C = 4.8314e-4;

            double rho = rho_w + A*S + B*S*std::sqrt(S) + C*S*S;

            // 压力修正（简化）
            double K = 19652.21 + 148.4206*T - 2.327105*T*T + 1.360477e-2*T*T*T;
            K += S * (54.6746 - 0.603459*T + 1.09987e-2*T*T);

            rho *= (1.0 + P / K);

            return rho;
        }

        void CurrentFieldSolver::computeHydrostaticPressure() {
            auto& rho = current_state_.density;
            auto& p = current_state_.pressure;
            auto& ssh = current_state_.ssh;

            // 静水压力：p = ρgh + p_atm
            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    double depth = has_topography_ ? bottom_depth_(i,j) : 1000.0;
                    p(i,j) = rho(i,j) * params_.gravity * (depth + ssh(i,j));
                }
            }
        }

        void CurrentFieldSolver::applyBoundaryConditions() {
            // 简化的边界条件：无滑移条件
            auto& u = current_state_.u;
            auto& v = current_state_.v;
            auto& T = current_state_.temperature;
            auto& S = current_state_.salinity;
            auto& ssh = current_state_.ssh;

            // 东西边界
            for (int j = 0; j < ny_; ++j) {
                // 西边界
                u(0,j) = 0.0;
                v(0,j) = 0.0;
                T(0,j) = T(1,j);
                S(0,j) = S(1,j);
                ssh(0,j) = ssh(1,j);

                // 东边界
                u(nx_-1,j) = 0.0;
                v(nx_-1,j) = 0.0;
                T(nx_-1,j) = T(nx_-2,j);
                S(nx_-1,j) = S(nx_-2,j);
                ssh(nx_-1,j) = ssh(nx_-2,j);
            }

            // 南北边界
            for (int i = 0; i < nx_; ++i) {
                // 南边界
                u(i,0) = 0.0;
                v(i,0) = 0.0;
                T(i,0) = T(i,1);
                S(i,0) = S(i,1);
                ssh(i,0) = ssh(i,1);

                // 北边界
                u(i,ny_-1) = 0.0;
                v(i,ny_-1) = 0.0;
                T(i,ny_-1) = T(i,ny_-2);
                S(i,ny_-1) = S(i,ny_-2);
                ssh(i,ny_-1) = ssh(i,ny_-2);
            }
        }

        double CurrentFieldSolver::computeCFLTimeStep() const {
            const auto& u = current_state_.u;
            const auto& v = current_state_.v;

            double max_u = u.cwiseAbs().maxCoeff();
            double max_v = v.cwiseAbs().maxCoeff();

            double cfl_x = dx_ / std::max(max_u, 1e-10);
            double cfl_y = dy_ / std::max(max_v, 1e-10);
            double cfl_gravity = std::min(dx_, dy_) / std::sqrt(params_.gravity * 1000.0);

            return 0.5 * std::min({cfl_x, cfl_y, cfl_gravity});
        }

        Eigen::MatrixXd CurrentFieldSolver::computeVorticity() const {
            const auto& u = current_state_.u;
            const auto& v = current_state_.v;

            Eigen::MatrixXd vorticity(nx_, ny_);

            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    double dv_dx = (v(i+1,j) - v(i-1,j)) / (2.0 * dx_);
                    double du_dy = (u(i,j+1) - u(i,j-1)) / (2.0 * dy_);
                    vorticity(i,j) = dv_dx - du_dy;
                }
            }

            return vorticity;
        }

        Eigen::MatrixXd CurrentFieldSolver::computeDivergence() const {
            const auto& u = current_state_.u;
            const auto& v = current_state_.v;

            Eigen::MatrixXd divergence(nx_, ny_);

            for (int i = 1; i < nx_-1; ++i) {
                for (int j = 1; j < ny_-1; ++j) {
                    double du_dx = (u(i+1,j) - u(i-1,j)) / (2.0 * dx_);
                    double dv_dy = (v(i,j+1) - v(i,j-1)) / (2.0 * dy_);
                    divergence(i,j) = du_dx + dv_dy;
                }
            }

            return divergence;
        }

        Eigen::MatrixXd CurrentFieldSolver::computeKineticEnergy() const {
            const auto& u = current_state_.u;
            const auto& v = current_state_.v;

            Eigen::MatrixXd ke(nx_, ny_);

            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    ke(i,j) = 0.5 * (u(i,j)*u(i,j) + v(i,j)*v(i,j));
                }
            }

            return ke;
        }

        double CurrentFieldSolver::computeTotalEnergy() const {
            auto ke = computeKineticEnergy();
            const auto& ssh = current_state_.ssh;

            double total_ke = ke.sum() * dx_ * dy_;
            double total_pe = 0.5 * params_.gravity * ssh.array().square().sum() * dx_ * dy_;

            return total_ke + total_pe;
        }

        bool CurrentFieldSolver::checkMassConservation(double tolerance) const {
            return std::abs(computeMassImbalance()) < tolerance;
        }

        double CurrentFieldSolver::computeMassImbalance() const {
            const auto& u = current_state_.u;
            const auto& v = current_state_.v;

            double mass_flux_x = 0.0;
            double mass_flux_y = 0.0;

            // 计算通过边界的质量通量
            for (int j = 0; j < ny_; ++j) {
                mass_flux_x += u(0,j) - u(nx_-1,j);
            }

            for (int i = 0; i < nx_; ++i) {
                mass_flux_y += v(i,0) - v(i,ny_-1);
            }

            return (mass_flux_x * dy_ + mass_flux_y * dx_);
        }

    } // namespace Core
} // namespace OceanSim