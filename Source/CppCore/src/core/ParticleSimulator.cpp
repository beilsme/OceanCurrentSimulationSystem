// src/core/ParticleSimulator.cpp
#include "core/ParticleSimulator.h"
#include "utils/MathUtils.h"
#include "utils/Logger.h"
#include <chrono>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#  include <immintrin.h>
#  define SIMD_SUPPORTED 1
#else
#  define SIMD_SUPPORTED 0
#endif


namespace OceanSim {
    namespace Core {

        ParticleSimulator::ParticleSimulator(
                std::shared_ptr<Data::GridDataStructure> grid,
                std::shared_ptr<Algorithms::RungeKuttaSolver> solver)
                : grid_(grid), solver_(solver) {

            LOG_INFO("ParticleSimulator initialized with " + std::to_string(num_threads_) + " threads");
        }

        void ParticleSimulator::initializeParticles(
                const std::vector<Eigen::Vector3d>& positions) {

            particles_.clear();
            trajectories_.clear();

            particles_.reserve(positions.size());
            trajectories_.resize(positions.size());

            for (size_t i = 0; i < positions.size(); ++i) {
                Particle p(positions[i]);
                p.id = static_cast<int>(i);
                particles_.push_back(p);
                trajectories_[i].push_back(positions[i]);
            }

            LOG_INFO("Initialized " + std::to_string(particles_.size()) + " particles");
        }

        void ParticleSimulator::initializeRandomParticles(
                int count, const Eigen::Vector3d& bounds_min,
                const Eigen::Vector3d& bounds_max) {

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist_x(bounds_min.x(), bounds_max.x());
            std::uniform_real_distribution<double> dist_y(bounds_min.y(), bounds_max.y());
            std::uniform_real_distribution<double> dist_z(bounds_min.z(), bounds_max.z());

            std::vector<Eigen::Vector3d> positions;
            positions.reserve(count);

            for (int i = 0; i < count; ++i) {
                positions.emplace_back(dist_x(gen), dist_y(gen), dist_z(gen));
            }

            initializeParticles(positions);
        }

        void ParticleSimulator::stepForward(double dt) {
            auto start = std::chrono::high_resolution_clock::now();

            if (vectorization_enabled_ && particles_.size() > 1000) {
                updateParticlesSIMD(dt);
            } else {
                updateParticlesParallel(dt);
            }

            auto end = std::chrono::high_resolution_clock::now();
            computation_time_ = std::chrono::duration<double>(end - start).count();
        }

        void ParticleSimulator::updateParticlesParallel(double dt) {
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
            for (size_t i = 0; i < particles_.size(); ++i) {
                if (!particles_[i].active) continue;

                auto& particle = particles_[i];

                // 获取当前位置的速度场
                Eigen::Vector3d current_velocity = interpolateVelocity(particle.position);

                // 使用Runge-Kutta方法更新位置
                auto rk_func = [this](const Eigen::Vector3d& pos, double) {
                    return interpolateVelocity(pos);
                };

                Eigen::Vector3d new_position = solver_->solve(
                        particle.position, current_velocity, dt, rk_func);

                // 应用扩散
                if (diffusion_enabled_) {
                    applyDiffusion(particle, dt);
                }

                particle.position = new_position;
                particle.age += dt;

                // 边界条件处理
                applyBoundaryConditions(particle);

                // 记录轨迹
                if (particle.id >= 0 && particle.id < static_cast<int>(trajectories_.size())) {
                    trajectories_[particle.id].push_back(particle.position);
                }
            }
        }

        void ParticleSimulator::updateParticlesSIMD(double dt) {
#if SIMD_SUPPORTED
            // 使用Intel TBB进行SIMD优化的并行计算
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, particles_.size(), 8),
                    [this, dt](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            if (!particles_[i].active) continue;

                            auto& particle = particles_[i];

                            // 4个粒子的SIMD处理
                            if (i + 3 < range.end()) {
                                __m256d pos_x = _mm256_set_pd(
                                        particles_[i].position.x(),
                                        particles_[i+1].position.x(),
                                        particles_[i+2].position.x(),
                                        particles_[i+3].position.x()
                                );

                                // SIMD速度插值和位置更新
                                // 简化版本，实际实现需要更复杂的向量化
                                for (int j = 0; j < 4 && i + j < range.end(); ++j) {
                                    updateParticlesSingle(particles_[i + j], dt);
                                }
                                i += 3; // 跳过已处理的粒子
                            } else {
                                updateParticlesSingle(particle, dt);
                            }
                        }
                    }
            );
#else
            updateParticlesParallel(dt);
#endif
        }

        void ParticleSimulator::updateParticlesSingle(Particle& particle, double dt) {
            Eigen::Vector3d current_velocity = interpolateVelocity(particle.position);

            auto rk_func = [this](const Eigen::Vector3d& pos, double) {
                return interpolateVelocity(pos);
            };

            particle.position = solver_->solve(
                    particle.position, current_velocity, dt, rk_func);

            if (diffusion_enabled_) {
                applyDiffusion(particle, dt);
            }

            particle.age += dt;
            applyBoundaryConditions(particle);

            if (particle.id >= 0 && particle.id < static_cast<int>(trajectories_.size())) {
                trajectories_[particle.id].push_back(particle.position);
            }
        }

        Eigen::Vector3d ParticleSimulator::interpolateVelocity(
                const Eigen::Vector3d& position) const {

            if (!grid_) {
                return Eigen::Vector3d::Zero();
            }

            // 三线性插值获取速度场
            return grid_->interpolateVector(position, "velocity");
        }

        void ParticleSimulator::applyBoundaryConditions(Particle& particle) {
            if (periodic_boundaries_) {
                // 周期性边界条件
                for (int i = 0; i < 3; ++i) {
                    double range = bounds_max_[i] - bounds_min_[i];
                    if (particle.position[i] < bounds_min_[i]) {
                        particle.position[i] += range;
                    } else if (particle.position[i] > bounds_max_[i]) {
                        particle.position[i] -= range;
                    }
                }
            } else {
                // 反射边界条件或粒子失活
                bool out_of_bounds = false;
                for (int i = 0; i < 3; ++i) {
                    if (particle.position[i] < bounds_min_[i] ||
                        particle.position[i] > bounds_max_[i]) {
                        out_of_bounds = true;
                        break;
                    }
                }

                if (out_of_bounds) {
                    particle.active = false;
                }
            }
        }

        void ParticleSimulator::applyDiffusion(Particle& particle, double dt) {
            if (!diffusion_enabled_) return;

            // 随机游走模拟扩散
            double variance = 2.0 * diffusion_coeff_ * dt;
            Eigen::Vector3d noise = addNoise(variance);
            particle.position += noise;
        }

        Eigen::Vector3d ParticleSimulator::addNoise(double variance) const {
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, std::sqrt(variance));

            return Eigen::Vector3d(dist(gen), dist(gen), dist(gen));
        }

        void ParticleSimulator::stepForwardAdaptive(double dt, double tolerance) {
            // 自适应时间步长算法
            double current_dt = dt;
            const double min_dt = dt * 0.01;
            const double max_dt = dt * 2.0;

            for (auto& particle : particles_) {
                if (!particle.active) continue;

                Eigen::Vector3d pos_backup = particle.position;

                // 两次半步长计算
                stepForwardSingle(particle, current_dt * 0.5);
                stepForwardSingle(particle, current_dt * 0.5);
                Eigen::Vector3d pos_half = particle.position;

                // 恢复并进行一次全步长计算
                particle.position = pos_backup;
                stepForwardSingle(particle, current_dt);
                Eigen::Vector3d pos_full = particle.position;

                // 估算误差
                double error = (pos_half - pos_full).norm();

                if (error > tolerance && current_dt > min_dt) {
                    current_dt *= 0.5;
                    particle.position = pos_backup; // 重新计算
                } else if (error < tolerance * 0.1 && current_dt < max_dt) {
                    current_dt *= 1.5;
                }
            }
        }

        void ParticleSimulator::stepForwardSingle(Particle& particle, double dt) {
            Eigen::Vector3d velocity = interpolateVelocity(particle.position);
            particle.position += velocity * dt;

            if (diffusion_enabled_) {
                applyDiffusion(particle, dt);
            }

            applyBoundaryConditions(particle);
        }

        std::vector<std::vector<Eigen::Vector3d>>
        ParticleSimulator::getTrajectories() const {
            return trajectories_;
        }

        size_t ParticleSimulator::getActiveParticleCount() const {
            return std::count_if(particles_.begin(), particles_.end(),
                                 [](const Particle& p) { return p.active; });
        }

        void ParticleSimulator::setBoundaryConditions(
                const Eigen::Vector3d& bounds_min,
                const Eigen::Vector3d& bounds_max,
                bool periodic) {

            bounds_min_ = bounds_min;
            bounds_max_ = bounds_max;
            periodic_boundaries_ = periodic;
        }

        void ParticleSimulator::enableDiffusion(double diffusion_coefficient) {
            diffusion_enabled_ = true;
            diffusion_coeff_ = diffusion_coefficient;
        }

        void ParticleSimulator::setSourceTerm(
                const Eigen::Vector3d& source_pos, double source_rate) {

            has_source_ = true;
            source_position_ = source_pos;
            source_rate_ = source_rate;
        }

        void ParticleSimulator::setNumThreads(int threads) {
            num_threads_ = std::max(1, std::min(threads, omp_get_max_threads()));
            omp_set_num_threads(num_threads_);
        }

        void ParticleSimulator::enableVectorization(bool enable) {
            vectorization_enabled_ = enable;
        }

    } // namespace Core
} // namespace OceanSim