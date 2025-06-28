// include/core/ParticleSimulator.h
#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <omp.h>
#include "data/GridDataStructure.h"
#include "algorithms/RungeKuttaSolver.h"

namespace OceanSim {
    namespace Core {

/**
 * @brief 海洋粒子模拟器 - 实现拉格朗日粒子追踪
 * 支持大规模并行计算和高精度数值积分
 */
        class ParticleSimulator {
        public:
            // 粒子状态结构
            struct Particle {
                Eigen::Vector3d position;    // 位置 (x, y, z)
                Eigen::Vector3d velocity;    // 速度
                double age = 0.0;            // 粒子年龄
                int id = -1;                 // 粒子ID
                bool active = true;          // 是否活跃

                Particle() = default;
                Particle(const Eigen::Vector3d& pos) : position(pos) {}
            };

            // 构造函数
            ParticleSimulator(std::shared_ptr<Data::GridDataStructure> grid,
                              std::shared_ptr<Algorithms::RungeKuttaSolver> solver);

            ~ParticleSimulator() = default;

            // 初始化粒子
            void initializeParticles(const std::vector<Eigen::Vector3d>& positions);
            void initializeRandomParticles(int count, const Eigen::Vector3d& bounds_min,
                                           const Eigen::Vector3d& bounds_max);

            // 时间积分
            void stepForward(double dt);
            void stepForwardAdaptive(double dt, double tolerance = 1e-6);

            // 并行计算
            void setNumThreads(int threads);
            void enableVectorization(bool enable);

            // 获取结果
            const std::vector<Particle>& getParticles() const { return particles_; }
            std::vector<std::vector<Eigen::Vector3d>> getTrajectories() const;

            // 性能监控
            double getComputationTime() const { return computation_time_; }
            size_t getActiveParticleCount() const;

            // 边界条件处理
            void setBoundaryConditions(const Eigen::Vector3d& bounds_min,
                                       const Eigen::Vector3d& bounds_max,
                                       bool periodic = false);

            // 污染物扩散模拟
            void enableDiffusion(double diffusion_coefficient);
            void setSourceTerm(const Eigen::Vector3d& source_pos, double source_rate);

        private:
            std::vector<Particle> particles_;
            std::vector<std::vector<Eigen::Vector3d>> trajectories_;

            std::shared_ptr<Data::GridDataStructure> grid_;
            std::shared_ptr<Algorithms::RungeKuttaSolver> solver_;

            // 计算参数
            int num_threads_ = omp_get_max_threads();
            bool vectorization_enabled_ = true;
            double computation_time_ = 0.0;

            // 边界条件
            Eigen::Vector3d bounds_min_, bounds_max_;
            bool periodic_boundaries_ = false;

            // 扩散参数
            bool diffusion_enabled_ = false;
            double diffusion_coeff_ = 0.0;

            // 源项
            bool has_source_ = false;
            Eigen::Vector3d source_position_;
            double source_rate_ = 0.0;

            // 内部方法
            Eigen::Vector3d interpolateVelocity(const Eigen::Vector3d& position) const;
            void applyBoundaryConditions(Particle& particle);
            void applyDiffusion(Particle& particle, double dt);
            Eigen::Vector3d addNoise(double variance) const;

            // 并行计算辅助
            void updateParticlesParallel(double dt);
            void updateParticlesSIMD(double dt);
        };

    } // namespace Core
} // namespace OceanSim