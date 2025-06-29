#include "algorithms/FiniteDifferenceSolver.h"
#include "utils/Logger.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace OceanSim {
    namespace Algorithms {

        FiniteDifferenceSolver::FiniteDifferenceSolver(int gridSize, double timeStep)
                : gridSize_(gridSize), timeStep_(timeStep), gridSpacing_(1.0 / gridSize), boundaryType_(0) {
            if (gridSize <= 0) {
                throw std::invalid_argument("Grid size must be positive");
            }
            if (timeStep <= 0) {
                throw std::invalid_argument("Time step must be positive");
            }

            LOG_INFO("FiniteDifferenceSolver initialized with grid size: " + std::to_string(gridSize));
            LOG_INFO("Grid spacing: " + std::to_string(gridSpacing_) + ", Time step: " + std::to_string(timeStep_));
        }

        FiniteDifferenceSolver::~FiniteDifferenceSolver() {
            LOG_DEBUG("FiniteDifferenceSolver destroyed");
        }

        bool FiniteDifferenceSolver::solveAdvectionDiffusion(
                const std::vector<double>& u,
                std::vector<double>& concentration,
                double diffusivity) {

            PERF_TIMER("AdvectionDiffusion");

            if (u.size() != concentration.size() || u.size() != static_cast<size_t>(gridSize_)) {
                LOG_ERROR("Input arrays size mismatch");
                return false;
            }

            std::vector<double> newConcentration(concentration.size());

            // 实现显式有限差分格式求解平流扩散方程
            // ∂C/∂t + u∂C/∂x = D∂²C/∂x²
            for (int i = 1; i < gridSize_ - 1; ++i) {
                // 计算平流项 (一阶迎风格式)
                double advectionTerm;
                if (u[i] >= 0) {
                    advectionTerm = u[i] * (concentration[i] - concentration[i-1]) / gridSpacing_;
                } else {
                    advectionTerm = u[i] * (concentration[i+1] - concentration[i]) / gridSpacing_;
                }

                // 计算扩散项 (中心差分格式)
                double diffusionTerm = diffusivity *
                                       (concentration[i+1] - 2.0 * concentration[i] + concentration[i-1]) /
                                       (gridSpacing_ * gridSpacing_);

                // 时间步进
                newConcentration[i] = concentration[i] + timeStep_ * (-advectionTerm + diffusionTerm);
            }

            // 应用边界条件
            newConcentration[0] = concentration[0];
            newConcentration[gridSize_-1] = concentration[gridSize_-1];

            applyBoundaryConditions(newConcentration);

            concentration = std::move(newConcentration);

            // 验证CFL条件
            double maxVelocity = *std::max_element(u.begin(), u.end());
            double cfl = computeCFL(maxVelocity);
            if (cfl > 1.0) {
                LOG_WARNING("CFL condition violated: " + std::to_string(cfl));
            }

            return true;
        }

        void FiniteDifferenceSolver::computeSpatialDerivative(
                const std::vector<double>& field,
                std::vector<double>& derivative,
                int order) {

            if (field.size() != derivative.size()) {
                derivative.resize(field.size());
            }

            if (order == 1) {
                // 一阶导数 - 中心差分格式
                for (size_t i = 1; i < field.size() - 1; ++i) {
                    derivative[i] = (field[i+1] - field[i-1]) / (2.0 * gridSpacing_);
                }

                // 边界点使用向前/向后差分
                derivative[0] = (field[1] - field[0]) / gridSpacing_;
                derivative[field.size()-1] = (field[field.size()-1] - field[field.size()-2]) / gridSpacing_;

            } else if (order == 2) {
                // 二阶导数 - 中心差分格式
                for (size_t i = 1; i < field.size() - 1; ++i) {
                    derivative[i] = (field[i+1] - 2.0 * field[i] + field[i-1]) /
                                    (gridSpacing_ * gridSpacing_);
                }

                // 边界点设为零
                derivative[0] = 0.0;
                derivative[field.size()-1] = 0.0;
            }
        }

        void FiniteDifferenceSolver::setBoundaryConditions(
                int boundaryType,
                const std::vector<double>& boundaryValues) {

            boundaryType_ = boundaryType;
            boundaryValues_ = boundaryValues;

            LOG_INFO("Boundary conditions set: type = " + std::to_string(boundaryType));
        }

        void FiniteDifferenceSolver::applyBoundaryConditions(std::vector<double>& field) {
            switch (boundaryType_) {
                case 0: // Dirichlet边界条件
                    if (boundaryValues_.size() >= 2) {
                        field[0] = boundaryValues_[0];
                        field[field.size()-1] = boundaryValues_[1];
                    }
                    break;

                case 1: // Neumann边界条件
                    if (boundaryValues_.size() >= 2) {
                        field[0] = field[1] - boundaryValues_[0] * gridSpacing_;
                        field[field.size()-1] = field[field.size()-2] + boundaryValues_[1] * gridSpacing_;
                    }
                    break;

                case 2: // 周期性边界条件
                    field[0] = field[field.size()-2];
                    field[field.size()-1] = field[1];
                    break;

                default:
                    LOG_WARNING("Unknown boundary condition type: " + std::to_string(boundaryType_));
                    break;
            }
        }

        double FiniteDifferenceSolver::computeCFL(double velocity) const {
            return std::abs(velocity) * timeStep_ / gridSpacing_;
        }

    } // namespace Algorithms
} // namespace OceanSim