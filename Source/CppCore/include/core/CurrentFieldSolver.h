// include/core/CurrentFieldSolver.h
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <functional>
#include "data/GridDataStructure.h"
#include "algorithms/FiniteDifferenceSolver.h"

namespace OceanSim {
    namespace Core {

/**
 * @brief 洋流场求解器 - 求解海洋流体动力学方程
 * 基于HYCOM混合坐标系统和有限差分方法
 */
        class CurrentFieldSolver {
        public:
            // 海洋状态变量结构
            struct OceanState {
                Eigen::MatrixXd u;           // 东向速度分量
                Eigen::MatrixXd v;           // 北向速度分量
                Eigen::MatrixXd w;           // 垂向速度分量
                Eigen::MatrixXd temperature; // 温度场
                Eigen::MatrixXd salinity;    // 盐度场
                Eigen::MatrixXd density;     // 密度场
                Eigen::MatrixXd pressure;    // 压力场
                Eigen::MatrixXd ssh;         // 海表面高度

                OceanState() = default;
                OceanState(int nx, int ny, int nz);
                void resize(int nx, int ny, int nz);
            };

            // 物理参数
            struct PhysicalParameters {
                double gravity = 9.81;              // 重力加速度
                double coriolis_f = 1e-4;          // 科里奥利参数
                double beta = 2e-11;               // β平面近似参数
                double viscosity_h = 1e3;          // 水平粘性系数
                double viscosity_v = 1e-3;         // 垂直粘性系数
                double diffusivity_h = 1e2;        // 水平扩散系数
                double diffusivity_v = 1e-4;       // 垂直扩散系数
                double reference_density = 1025.0;  // 参考密度
            };

            // 构造函数
            CurrentFieldSolver(std::shared_ptr<Data::GridDataStructure> grid,
                               const PhysicalParameters& params = PhysicalParameters{});

            ~CurrentFieldSolver() = default;

            // 初始化
            void initialize(const OceanState& initial_state);
            void setBottomTopography(const Eigen::MatrixXd& bottom_depth);
            void setWindStress(const Eigen::MatrixXd& tau_x, const Eigen::MatrixXd& tau_y);

            // 时间积分
            void stepForward(double dt);
            void stepForwardBarotropic(double dt);    // 正压模式
            void stepForwardBaroclinic(double dt);    // 斜压模式

            // 模式分解
            void splitBarotropicBaroclinic();
            void coupleBarotropicBaroclinic();

            // 数值求解方法
            void solveNavierStokes(double dt);
            void solveContinuityEquation(double dt);
            void solveTemperatureEquation(double dt);
            void solveSalinityEquation(double dt);
            void computeDensity();

            // 边界条件
            void applyBoundaryConditions();
            void applyOpenBoundaryConditions();
            void applyNoSlipBoundaryConditions();

            // 获取结果
            const OceanState& getCurrentState() const { return current_state_; }
            OceanState& getCurrentState() { return current_state_; }

            // 诊断量计算
            Eigen::MatrixXd computeVorticity() const;
            Eigen::MatrixXd computeDivergence() const;
            Eigen::MatrixXd computeKineticEnergy() const;
            double computeTotalEnergy() const;

            // 质量守恒检查
            bool checkMassConservation(double tolerance = 1e-10) const;
            double computeMassImbalance() const;

        private:
            std::shared_ptr<Data::GridDataStructure> grid_;
            std::shared_ptr<Algorithms::FiniteDifferenceSolver> fd_solver_;

            OceanState current_state_;
            OceanState previous_state_;
            PhysicalParameters params_;

            // 网格参数
            int nx_, ny_, nz_;
            double dx_, dy_;
            std::vector<double> dz_;  // 垂直层厚度

            // 地形和强迫
            Eigen::MatrixXd bottom_depth_;
            Eigen::MatrixXd wind_stress_x_, wind_stress_y_;
            bool has_topography_ = false;
            bool has_wind_forcing_ = false;

            // 数值方法
            void computeHorizontalDiffusion(Eigen::MatrixXd& field, double coeff, double dt);
            void computeVerticalDiffusion(Eigen::MatrixXd& field, double coeff, double dt);
            void computeAdvection(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                  const Eigen::MatrixXd& w, Eigen::MatrixXd& field, double dt);

            // 压力梯度计算
            void computePressureGradient(Eigen::MatrixXd& dpdt_x, Eigen::MatrixXd& dpdt_y);
            void computeHydrostaticPressure();

            // 科里奥利效应
            void applyCoriolis(double dt);

            // 状态方程
            double computeSeawaterDensity(double temperature, double salinity, double pressure) const;

            // 数值稳定性
            double computeCFLTimeStep() const;
            void applyArtificialViscosity(double dt);

            // 湍流模式（简化版）
            void computeTurbulentViscosity();

            // 辅助方法
            void exchangeGhostCells();
            bool isValidGridPoint(int i, int j, int k) const;
            double interpolateBilinear(const Eigen::MatrixXd& field, double x, double y) const;
        };

    } // namespace Core
} // namespace OceanSim