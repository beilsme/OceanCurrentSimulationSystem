#ifndef PARALLEL_COMPUTE_ENGINE_H
#define PARALLEL_COMPUTE_ENGINE_H

#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace OceanSimulation {
    namespace Core {

/**
 * @brief 海洋模拟并行计算引擎
 * 
 * 提供高性能并行处理能力，专门针对海洋洋流模拟、粒子追踪和数值计算进行优化。
 * 支持OpenMP、Intel TBB和自定义线程池实现。
 */
        class ParallelComputeEngine {
        public:
            /**
             * @brief 并行执行策略
             */
            enum class ExecutionPolicy {
                Sequential,      // 顺序执行
                Parallel,        // 并行执行
                Vectorized,      // 向量化执行
                HybridParallel   // 混合并行
            };

            /**
             * @brief 任务优先级
             */
            enum class Priority {
                Low = 0,
                Normal = 1,
                High = 2,
                Critical = 3
            };

            /**
             * @brief 任务函数签名
             */
            using TaskFunction = std::function<void(size_t, size_t)>;
            using VoidTaskFunction = std::function<void()>;

            /**
             * @brief 计算引擎配置参数
             */
            struct Config {
                size_t maxThreads = std::thread::hardware_concurrency();
                size_t workStealingQueueSize = 1024;
                bool enableAffinity = true;
                bool enableHyperthreading = false;
                ExecutionPolicy defaultPolicy = ExecutionPolicy::Parallel;
                size_t chunkSize = 1000;
                double loadBalanceThreshold = 0.8;
            };

            /**
             * @brief 性能统计信息
             */
            struct PerformanceStats {
                std::atomic<uint64_t> tasksExecuted{0};
                std::atomic<uint64_t> totalExecutionTime{0};
                std::atomic<uint32_t> threadsActive{0};
                std::atomic<double> cpuUtilization{0.0};
                std::chrono::system_clock::time_point startTime;
            };

        private:
            Config config_;
            std::atomic<bool> running_;
            std::vector<std::thread> workers_;
            std::mutex taskMutex_;
            std::condition_variable taskCondition_;
            PerformanceStats stats_;

            // 具有优先级支持的任务队列
            struct Task {
                VoidTaskFunction function;
                Priority priority;
                std::chrono::system_clock::time_point submitTime;

                bool operator<(const Task& other) const {
                    return priority < other.priority;
                }
            };

            std::priority_queue<Task> taskQueue_;
            std::atomic<size_t> activeThreads_;

            // 工作线程函数
            void workerFunction(size_t threadId);

            // 负载均衡
            void balanceLoad();

        public:
            /**
             * @brief 构造函数
             * @param config 引擎配置参数
             */
            explicit ParallelComputeEngine(const Config& config = Config{});

            /**
             * @brief 析构函数
             */
            ~ParallelComputeEngine();

            /**
             * @brief 启动计算引擎
             */
            void start();

            /**
             * @brief 停止计算引擎
             */
            void stop();

            /**
             * @brief 提交并行任务
             * @param func 要执行的函数
             * @param begin 起始索引
             * @param end 结束索引
             * @param priority 任务优先级
             * @return future对象用于获取结果
             */
            std::future<void> submitTask(const TaskFunction& func, size_t begin, size_t end,
                                         Priority priority = Priority::Normal);

            /**
             * @brief 提交简单任务
             * @param func 要执行的函数
             * @param priority 任务优先级
             * @return future对象用于获取结果
             */
            std::future<void> submitTask(const VoidTaskFunction& func,
                                         Priority priority = Priority::Normal);

            /**
             * @brief 并行for循环
             * @param begin 起始索引
             * @param end 结束索引
             * @param func 循环体函数
             * @param policy 执行策略
             */
            void parallelFor(size_t begin, size_t end, const TaskFunction& func,
                             ExecutionPolicy policy = ExecutionPolicy::Parallel);

            /**
             * @brief 并行reduce操作
             * @param begin 起始索引
             * @param end 结束索引
             * @param identity 初始值
             * @param mapFunc 映射函数
             * @param reduceFunc 归约函数
             * @return 归约结果
             */
            template<typename T>
            T parallelReduce(size_t begin, size_t end, T identity,
                             std::function<T(size_t)> mapFunc,
                             std::function<T(T, T)> reduceFunc);

            /**
             * @brief 等待所有任务完成
             */
            void waitForAll();

            /**
             * @brief 获取性能统计信息
             */
            PerformanceStats getPerformanceStats() const;

            /**
             * @brief 设置线程亲和性
             */
            void setThreadAffinity(size_t threadId, size_t cpuId);

            /**
             * @brief 获取可用线程数量
             */
            size_t getAvailableThreads() const;

            /**
             * @brief 更新配置
             */
            void updateConfig(const Config& newConfig);
        };

// 模板方法实现
        template<typename T>
        T ParallelComputeEngine::parallelReduce(size_t begin, size_t end, T identity,
                                                std::function<T(size_t)> mapFunc,
                                                std::function<T(T, T)> reduceFunc) {
            if (begin >= end) return identity;

            const size_t numThreads = std::min(config_.maxThreads, end - begin);
            const size_t chunkSize = (end - begin + numThreads - 1) / numThreads;

            std::vector<std::future<T>> futures;
            futures.reserve(numThreads);

            for (size_t i = 0; i < numThreads; ++i) {
                size_t chunkBegin = begin + i * chunkSize;
                size_t chunkEnd = std::min(chunkBegin + chunkSize, end);

                if (chunkBegin >= chunkEnd) break;

                auto future = std::async(std::launch::async, [=]() {
                    T localResult = identity;
                    for (size_t j = chunkBegin; j < chunkEnd; ++j) {
                        localResult = reduceFunc(localResult, mapFunc(j));
                    }
                    return localResult;
                });

                futures.push_back(std::move(future));
            }

            T result = identity;
            for (auto& future : futures) {
                result = reduceFunc(result, future.get());
            }

            return result;
        }

    } // namespace Core
} // namespace OceanSimulation

#endif // PARALLEL_COMPUTE_ENGINE_H