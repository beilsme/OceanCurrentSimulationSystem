#ifndef FINITE_DIFFERENCE_SOLVER_H
#define FINITE_DIFFERENCE_SOLVER_H

#include <vector>
#include <memory>

namespace OceanSim {
    namespace Algorithms {

/**
 * @brief 有限差分求解器类
 * 
 * 实现海洋洋流模拟中的有限差分数值方法
 * 支持一阶、二阶和高阶有限差分格式
 */
        class FiniteDifferenceSolver {
        public:
            /**
             * @brief 构造函数
             * @param gridSize 网格大小
             * @param timeStep 时间步长
             */
            FiniteDifferenceSolver(int gridSize, double timeStep);

            /**
             * @brief 析构函数
             */
            virtual ~FiniteDifferenceSolver();

            /**
             * @brief 求解平流扩散方程
             * @param u 速度场数组
             * @param concentration 浓度场数组
             * @param diffusivity 扩散系数
             * @return 求解结果状态
             */
            bool solveAdvectionDiffusion(
                    const std::vector<double>& u,
                    std::vector<double>& concentration,
                    double diffusivity
            );

            /**
             * @brief 计算空间导数
             * @param field 输入场
             * @param derivative 输出导数
             * @param order 导数阶数
             */
            void computeSpatialDerivative(
                    const std::vector<double>& field,
                    std::vector<double>& derivative,
                    int order = 1
            );

            /**
             * @brief 设置边界条件
             * @param boundaryType 边界类型 (0: Dirichlet, 1: Neumann, 2: Periodic)
             * @param boundaryValues 边界值
             */
            void setBoundaryConditions(int boundaryType, const std::vector<double>& boundaryValues);

            /**
             * @brief 获取网格间距
             * @return 网格间距值
             */
            double getGridSpacing() const { return gridSpacing_; }

            /**
             * @brief 获取时间步长
             * @return 时间步长值
             */
            double getTimeStep() const { return timeStep_; }

        private:
            int gridSize_;              // 网格大小
            double timeStep_;           // 时间步长
            double gridSpacing_;        // 网格间距
            int boundaryType_;          // 边界条件类型
            std::vector<double> boundaryValues_;  // 边界值

            /**
             * @brief 应用边界条件
             * @param field 场数据
             */
            void applyBoundaryConditions(std::vector<double>& field);

            /**
             * @brief 计算CFL条件
             * @param velocity 流速
             * @return CFL数
             */
            double computeCFL(double velocity) const;
        };

    } // namespace Algorithms
} // namespace OceanSim

#endif // FINITE_DIFFERENCE_SOLVER_H