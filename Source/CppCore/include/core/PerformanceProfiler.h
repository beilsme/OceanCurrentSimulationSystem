// include/core/PerformanceProfiler.h
#pragma once

#include <Eigen/Sparse>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>

namespace OceanSim {
    namespace Core {

/**
 * @brief 高性能计算性能分析器
 * 监控计算时间、内存使用、并行效率等关键指标
 */
        class PerformanceProfiler {
        public:
            // 性能指标结构
            struct ProfileData {
                std::string name;
                double total_time = 0.0;           // 总时间（秒）
                double min_time = std::numeric_limits<double>::max();
                double max_time = 0.0;
                double avg_time = 0.0;
                size_t call_count = 0;             // 调用次数
                size_t memory_usage = 0;           // 内存使用（字节）
                double cpu_utilization = 0.0;     // CPU利用率
                int thread_count = 0;              // 线程数
                double parallel_efficiency = 0.0;  // 并行效率

                ProfileData(const std::string& n) : name(n) {}

                void updateTime(double time) {
                    total_time += time;
                    min_time = std::min(min_time, time);
                    max_time = std::max(max_time, time);
                    call_count++;
                    avg_time = total_time / call_count;
                }
            };

            // 计时器类
            class Timer {
            public:
                Timer(const std::string& name, PerformanceProfiler* profiler);
                ~Timer();

                void stop();
                double getElapsedTime() const;

            private:
                std::string name_;
                PerformanceProfiler* profiler_;
                std::chrono::high_resolution_clock::time_point start_time_;
                bool stopped_ = false;
            };

            // 内存监控器
            class MemoryMonitor {
            public:
                MemoryMonitor();

                size_t getCurrentMemoryUsage();
                size_t getPeakMemoryUsage();
                void recordMemoryUsage(const std::string& tag);
                std::vector<std::pair<std::string, size_t>> getMemoryHistory() const;

            private:
                size_t peak_memory_ = 0;
                std::vector<std::pair<std::string, size_t>> memory_history_;
                mutable std::mutex memory_mutex_;
            };

            // 单例模式
            static PerformanceProfiler& getInstance();

            // 禁用拷贝构造和赋值
            PerformanceProfiler(const PerformanceProfiler&) = delete;
            PerformanceProfiler& operator=(const PerformanceProfiler&) = delete;

            // 性能监控控制
            void enable(bool enabled = true);
            void disable();
            bool isEnabled() const { return enabled_; }

            // 计时功能
            void startTimer(const std::string& name);
            void stopTimer(const std::string& name);
            std::unique_ptr<Timer> createTimer(const std::string& name);

            // 性能数据记录
            void recordExecution(const std::string& name, double time);
            void recordMemoryUsage(const std::string& name, size_t memory_bytes);
            void recordParallelMetrics(const std::string& name, int thread_count, double efficiency);

            // 数据获取
            const ProfileData* getProfileData(const std::string& name) const;
            std::vector<ProfileData> getAllProfileData() const;
            double getTotalTime() const;
            size_t getTotalMemoryUsage() const;

            // 统计分析
            void printSummary(std::ostream& os = std::cout) const;
            void printDetailedReport(std::ostream& os = std::cout) const;
            void exportToJSON(const std::string& filename) const;
            void exportToCSV(const std::string& filename) const;

            // 重置和清理
            void reset();
            void clear();

            // 实时监控
            void startRealTimeMonitoring(double interval_seconds = 1.0);
            void stopRealTimeMonitoring();

            // 性能瓶颈检测
            std::vector<std::string> detectBottlenecks(double threshold_percentage = 10.0) const;
            std::string getMostExpensiveOperation() const;

            // 内存泄漏检测
            bool detectMemoryLeaks() const;
            std::vector<std::pair<std::string, size_t>> getMemoryLeaks() const;

            // 并行性能分析
            double getOverallParallelEfficiency() const;
            std::vector<std::pair<std::string, double>> getParallelEfficiencyByOperation() const;

            // 基准测试
            void runMicrobenchmark(const std::string& name,
                                   const std::function<void()>& operation,
                                   int iterations = 1000);
            void compareBenchmarks(const std::string& baseline, const std::string& comparison) const;

        private:
            PerformanceProfiler();
            ~PerformanceProfiler();

            bool enabled_ = false;
            mutable std::mutex data_mutex_;
            std::unordered_map<std::string, ProfileData> profile_data_;
            std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> active_timers_;

            // 内存监控
            std::unique_ptr<MemoryMonitor> memory_monitor_;

            // 实时监控
            std::atomic<bool> real_time_monitoring_ = false;
            std::unique_ptr<std::thread> monitoring_thread_;
            double monitoring_interval_ = 1.0;

            // 系统资源监控
            void updateSystemMetrics();
            double getCurrentCPUUsage();
            size_t getCurrentMemoryUsage();
            int getCurrentThreadCount();

            // 数据处理
            void processProfileData();
            void calculateStatistics();

            // 输出格式化
            std::string formatTime(double seconds) const;
            std::string formatMemory(size_t bytes) const;
            std::string formatPercentage(double percentage) const;
        };

/**
 * @brief RAII风格的性能分析宏
 */
#define PROFILE_SCOPE(name) \
    auto timer_##__LINE__ = OceanSim::Core::PerformanceProfiler::getInstance().createTimer(name)

#define PROFILE_FUNCTION() \
    PROFILE_SCOPE(__FUNCTION__)

#define PROFILE_BLOCK(name) \
    for (auto timer = OceanSim::Core::PerformanceProfiler::getInstance().createTimer(name); \
         timer != nullptr; timer = nullptr)

/**
 * @brief 专用性能计数器
 */
        class PerformanceCounters {
        public:
            // 计算性能计数器
            static void incrementParticleCount(size_t count);
            static void incrementTimeSteps();
            static void recordIterationTime(double time);
            static void recordConvergenceTime(double time);

            // 并行计算计数器
            static void recordParallelTime(int thread_count, double time);
            static void recordLoadBalance(const std::vector<double>& thread_times);
            static void recordCommunicationOverhead(double comm_time, double comp_time);

            // 内存访问计数器
            static void recordCacheHit();
            static void recordCacheMiss();
            static void recordMemoryBandwidth(size_t bytes_transferred, double time);

            // 数值算法计数器
            static void recordLinearSolverIterations(int iterations);
            static void recordNonlinearSolverIterations(int iterations);
            static void recordMatrixVectorOperations(size_t count);

            // 获取统计数据
            static size_t getTotalParticleCount();
            static size_t getTotalTimeSteps();
            static double getAverageIterationTime();
            static double getCacheHitRatio();
            static double getAverageParallelEfficiency();

            // 重置计数器
            static void reset();

        private:
            static std::atomic<size_t> total_particles_;
            static std::atomic<size_t> total_timesteps_;
            static std::atomic<size_t> cache_hits_;
            static std::atomic<size_t> cache_misses_;
            static std::atomic<double> total_iteration_time_;
            static std::atomic<size_t> iteration_count_;
            static std::mutex parallel_data_mutex_;
            static std::vector<std::pair<int, double>> parallel_times_;
        };

/**
 * @brief GPU性能监控器（CUDA/OpenCL）
 */
        class GPUProfiler {
        public:
            struct GPUMetrics {
                double kernel_time = 0.0;
                double memory_transfer_time = 0.0;
                size_t memory_usage = 0;
                double gpu_utilization = 0.0;
                double memory_bandwidth = 0.0;
            };

            static void startKernelTimer(const std::string& kernel_name);
            static void stopKernelTimer(const std::string& kernel_name);

            static void recordMemoryTransfer(size_t bytes, bool host_to_device);
            static void recordGPUMemoryUsage(size_t bytes);

            static GPUMetrics getGPUMetrics(const std::string& kernel_name);
            static void printGPUSummary(std::ostream& os = std::cout);

        private:
            static std::unordered_map<std::string, GPUMetrics> gpu_metrics_;
            static std::mutex gpu_mutex_;
        };

/**
 * @brief 自动性能优化建议生成器
 */
        class PerformanceOptimizer {
        public:
            struct OptimizationSuggestion {
                std::string category;      // 优化类别
                std::string description;   // 建议描述
                double potential_speedup;  // 潜在加速比
                int priority;             // 优先级 (1-10)
            };

            static std::vector<OptimizationSuggestion> analyzePerformance(
                    const std::vector<PerformanceProfiler::ProfileData>& profile_data);

            static std::vector<OptimizationSuggestion> suggestMemoryOptimizations(
                    const PerformanceProfiler::MemoryMonitor& memory_monitor);

            static std::vector<OptimizationSuggestion> suggestParallelOptimizations(
                    const std::vector<std::pair<std::string, double>>& parallel_efficiency);

            static void printOptimizationReport(
                    const std::vector<OptimizationSuggestion>& suggestions,
                    std::ostream& os = std::cout);

        private:
            static bool detectInefficiency(const PerformanceProfiler::ProfileData& data);
            static double estimateSpeedup(const std::string& optimization_type);
        };

    } // namespace Core
} // namespace OceanSim