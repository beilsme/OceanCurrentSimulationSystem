#include <algorithm>
#include "algorithms/ParallelComputeEngine.h"
#include <iostream>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace OceanSimulation {
    namespace Core {

        ParallelComputeEngine::ParallelComputeEngine(const Config& config)
                : config_(config), running_(false), activeThreads_(0) {
            stats_.startTime = std::chrono::system_clock::now();
        }

        ParallelComputeEngine::~ParallelComputeEngine() {
            stop();
        }

        void ParallelComputeEngine::start() {
            if (running_.exchange(true)) {
                return; // 已经在运行
            }

            // 创建工作线程
            workers_.reserve(config_.maxThreads);
            for (size_t i = 0; i < config_.maxThreads; ++i) {
                workers_.emplace_back(&ParallelComputeEngine::workerFunction, this, i);

                // 设置线程亲和性
                if (config_.enableAffinity) {
                    setThreadAffinity(i, i % std::thread::hardware_concurrency());
                }
            }

            std::cout << "并行计算引擎已启动，使用 " << config_.maxThreads << " 个线程" << std::endl;
        }

        void ParallelComputeEngine::stop() {
            if (!running_.exchange(false)) {
                return; // 已经停止
            }

            // 通知所有工作线程停止
            taskCondition_.notify_all();

            // 等待所有线程完成
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            workers_.clear();
            std::cout << "并行计算引擎已停止" << std::endl;
        }

        void ParallelComputeEngine::workerFunction(size_t threadId) {
            while (running_) {
                Task task;
                bool hasTask = false;

                // 从任务队列获取任务
                {
                    std::unique_lock<std::mutex> lock(taskMutex_);
                    taskCondition_.wait(lock, [this] {
                        return !taskQueue_.empty() || !running_;
                    });

                    if (!running_) break;

                    if (!taskQueue_.empty()) {
                        task = taskQueue_.top();
                        taskQueue_.pop();
                        hasTask = true;
                    }
                }

                if (hasTask) {
                    ++activeThreads_;
                    auto startTime = std::chrono::high_resolution_clock::now();

                    try {
                        task.function();
                        ++stats_.tasksExecuted;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "线程 " << threadId << " 执行任务时发生异常: " << e.what() << std::endl;
                    }

                    auto endTime = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
                    stats_.totalExecutionTime += duration.count();

                    --activeThreads_;
                    stats_.threadsActive = activeThreads_.load();
                }
            }
        }

        std::future<void> ParallelComputeEngine::submitTask(const TaskFunction& func, size_t begin, size_t end, Priority priority) {
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();

            VoidTaskFunction wrappedFunc = [func, begin, end, promise]() {
                try {
                    func(begin, end);
                    promise->set_value();
                }
                catch (...) {
                    promise->set_exception(std::current_exception());
                }
            };

            {
                std::lock_guard<std::mutex> lock(taskMutex_);
                taskQueue_.push({wrappedFunc, priority, std::chrono::system_clock::now()});
            }

            taskCondition_.notify_one();
            return future;
        }

        std::future<void> ParallelComputeEngine::submitTask(const VoidTaskFunction& func, Priority priority) {
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();

            VoidTaskFunction wrappedFunc = [func, promise]() {
                try {
                    func();
                    promise->set_value();
                }
                catch (...) {
                    promise->set_exception(std::current_exception());
                }
            };

            {
                std::lock_guard<std::mutex> lock(taskMutex_);
                taskQueue_.push({wrappedFunc, priority, std::chrono::system_clock::now()});
            }

            taskCondition_.notify_one();
            return future;
        }

        void ParallelComputeEngine::parallelFor(size_t begin, size_t end, const TaskFunction& func, ExecutionPolicy policy) {
            if (begin >= end) return;

            switch (policy) {
                case ExecutionPolicy::Sequential: {
                    func(begin, end);
                    break;
                }

                case ExecutionPolicy::Parallel: {
#ifdef _OPENMP
                    // 使用OpenMP实现
            #pragma omp parallel for schedule(dynamic, config_.chunkSize)
            for (size_t i = begin; i < end; ++i) {
                func(i, i + 1);
            }
#else
                    // 使用自定义线程池实现
                    const size_t numThreads = std::min(config_.maxThreads, end - begin);
                    const size_t chunkSize = std::max(config_.chunkSize, (end - begin + numThreads - 1) / numThreads);

                    std::vector<std::future<void>> futures;
                    futures.reserve((end - begin + chunkSize - 1) / chunkSize);

                    for (size_t i = begin; i < end; i += chunkSize) {
                        size_t chunkEnd = std::min(i + chunkSize, end);
                        futures.push_back(submitTask(func, i, chunkEnd));
                    }

                    // 等待所有任务完成
                    for (auto& future : futures) {
                        future.wait();
                    }
#endif
                    break;
                }

                case ExecutionPolicy::Vectorized: {
                    // 向量化执行（需要编译器支持自动向量化）
#pragma omp simd
                    for (size_t i = begin; i < end; ++i) {
                        func(i, i + 1);
                    }
                    break;
                }

                case ExecutionPolicy::HybridParallel: {
                    // 混合并行和向量化
#ifdef _OPENMP
                    #pragma omp parallel for simd schedule(dynamic, config_.chunkSize)
            for (size_t i = begin; i < end; ++i) {
                func(i, i + 1);
            }
#else
                    parallelFor(begin, end, func, ExecutionPolicy::Parallel);
#endif
                    break;
                }
            }
        }

        void ParallelComputeEngine::waitForAll() {
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(taskMutex_);
                    if (taskQueue_.empty() && activeThreads_ == 0) {
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        ParallelComputeEngine::PerformanceStats ParallelComputeEngine::getPerformanceStats() const {
            return stats_;
        }

        void ParallelComputeEngine::setThreadAffinity(size_t threadId, size_t cpuId) {
            if (threadId >= workers_.size()) return;

#ifdef _WIN32
            HANDLE threadHandle = workers_[threadId].native_handle();
    DWORD_PTR affinityMask = 1ULL << cpuId;
    SetThreadAffinityMask(threadHandle, affinityMask);
#elif defined(__linux__)
            pthread_t thread = workers_[threadId].native_handle();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
        }

        size_t ParallelComputeEngine::getAvailableThreads() const {
            return config_.maxThreads - activeThreads_.load();
        }

        void ParallelComputeEngine::updateConfig(const Config& newConfig) {
            config_ = newConfig;

            // 如果正在运行，需要重启以应用新配置
            if (running_) {
                stop();
                start();
            }
        }

        void ParallelComputeEngine::balanceLoad() {
            // 简单的负载均衡实现
            double utilization = static_cast<double>(activeThreads_) / config_.maxThreads;
            if (utilization > config_.loadBalanceThreshold) {
                // 可以实现更复杂的负载均衡策略
                std::cout << "当前CPU利用率: " << utilization * 100 << "%" << std::endl;
            }
        }

    } // namespace Core
} // namespace OceanSimulation