#include "utils/ThreadPool.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>

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

// 线程本地存储
        thread_local uint32_t ThreadPool::threadLocalId_ = UINT32_MAX;

// ===========================================
// WorkStealingQueue 实现
// ===========================================

        void ThreadPool::WorkStealingQueue::push_back(Task&& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(task));
        }

        bool ThreadPool::WorkStealingQueue::pop_front(Task& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                return false;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
            return true;
        }

        bool ThreadPool::WorkStealingQueue::pop_back(Task& task) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                return false;
            }
            task = std::move(queue_.back());
            queue_.pop_back();
            return true;
        }

        size_t ThreadPool::WorkStealingQueue::size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

        bool ThreadPool::WorkStealingQueue::empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        void ThreadPool::WorkStealingQueue::clear() {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.clear();
        }

// ===========================================
// ThreadPool 主要实现
// ===========================================

        ThreadPool::ThreadPool(const Config& config)
                : config_(config), state_(PoolState::Stopped) {

            taskStats_.startTime = std::chrono::system_clock::now();
            lastResizeCheck_ = std::chrono::system_clock::now();

            // 验证配置
            if (config_.threadCount == 0) {
                config_.threadCount = std::thread::hardware_concurrency();
            }

            if (config_.maxThreads < config_.minThreads) {
                config_.maxThreads = config_.minThreads;
            }

            std::cout << "线程池初始化:" << std::endl;
            std::cout << "  线程数量: " << config_.threadCount << std::endl;
            std::cout << "  最大队列: " << config_.maxQueueSize << std::endl;
            std::cout << "  负载策略: ";
            switch (config_.strategy) {
                case LoadBalanceStrategy::RoundRobin: std::cout << "轮询"; break;
                case LoadBalanceStrategy::LeastLoaded: std::cout << "最少负载"; break;
                case LoadBalanceStrategy::WorkStealing: std::cout << "工作窃取"; break;
                case LoadBalanceStrategy::Affinity: std::cout << "亲和性优先"; break;
            }
            std::cout << std::endl;
        }

        ThreadPool::~ThreadPool() {
            stop(false); // 强制停止

            if (config_.enableStatistics) {
                std::cout << "\n=== 线程池销毁统计 ===" << std::endl;
                printStats();
            }
        }

        void ThreadPool::start() {
            PoolState expected = PoolState::Stopped;
            if (!state_.compare_exchange_strong(expected, PoolState::Starting)) {
                std::cout << "线程池已经在运行或正在启动" << std::endl;
                return;
            }

            try {
                // 创建线程统计对象
                threadStats_.clear();
                threadStats_.reserve(config_.threadCount);
                for (size_t i = 0; i < config_.threadCount; ++i) {
                    threadStats_.emplace_back(std::make_unique<ThreadStats>());
                    threadStats_[i]->threadId = static_cast<uint32_t>(i);
                }

                // 创建工作窃取队列
                if (config_.enableWorkStealing) {
                    workQueues_.clear();
                    workQueues_.reserve(config_.threadCount);
                    for (size_t i = 0; i < config_.threadCount; ++i) {
                        workQueues_.emplace_back(std::make_unique<WorkStealingQueue>());
                    }
                }

                // 创建工作线程
                workers_.clear();
                workers_.reserve(config_.threadCount);

                for (size_t i = 0; i < config_.threadCount; ++i) {
                    workers_.emplace_back(&ThreadPool::workerFunction, this, static_cast<uint32_t>(i));

                    if (config_.enableThreadAffinity) {
                        setThreadAffinity(static_cast<uint32_t>(i),
                                          static_cast<uint32_t>(i % std::thread::hardware_concurrency()));
                    }
                }

                state_ = PoolState::Running;
                std::cout << "线程池启动成功，" << config_.threadCount << " 个工作线程" << std::endl;
            }
            catch (const std::exception& e) {
                state_ = PoolState::Stopped;
                std::cerr << "线程池启动失败: " << e.what() << std::endl;
                throw;
            }
        }

        void ThreadPool::stop(bool graceful) {
            PoolState expected = PoolState::Running;
            if (!state_.compare_exchange_strong(expected, PoolState::Stopping)) {
                if (state_ == PoolState::Stopped) {
                    return; // 已经停止
                }
                // 如果状态不是Running，等待状态稳定
                while (state_ != PoolState::Stopped && state_ != PoolState::Running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if (state_ == PoolState::Stopped) return;
            }

            std::cout << "正在停止线程池..." << std::endl;

            if (graceful) {
                // 优雅停止：等待任务完成
                waitForAll(30000); // 最多等待30秒
            } else {
                // 强制停止：清空队列
                clearQueue();
            }

            // 通知所有线程停止
            condition_.notify_all();

            // 等待所有线程结束
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            workers_.clear();
            workQueues_.clear();

            state_ = PoolState::Stopped;
            std::cout << "线程池已停止" << std::endl;
        }

        void ThreadPool::pause() {
            PoolState expected = PoolState::Running;
            if (state_.compare_exchange_strong(expected, PoolState::Paused)) {
                std::cout << "线程池已暂停" << std::endl;
            }
        }

        void ThreadPool::resume() {
            PoolState expected = PoolState::Paused;
            if (state_.compare_exchange_strong(expected, PoolState::Running)) {
                condition_.notify_all();
                std::cout << "线程池已恢复" << std::endl;
            }
        }

        void ThreadPool::workerFunction(uint32_t threadId) {
            threadLocalId_ = threadId;
            ThreadStats& stats = *threadStats_[threadId];
            stats.isActive = true;
            stats.lastActivity = std::chrono::system_clock::now();

            std::cout << "工作线程 " << threadId << " 启动" << std::endl;

            while (state_ == PoolState::Running || state_ == PoolState::Paused) {
                if (state_ == PoolState::Paused) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }

                Task task;
                bool hasTask = getTask(task, threadId);

                if (hasTask) {
                    auto startTime = std::chrono::high_resolution_clock::now();

                    try {
                        task.function();
                        ++stats.tasksExecuted;
                        ++taskStats_.tasksCompleted;

                        auto endTime = std::chrono::high_resolution_clock::now();
                        auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

                        updateThreadStats(threadId, task, executionTime);
                    }
                    catch (const std::exception& e) {
                        std::cerr << "线程 " << threadId << " 执行任务异常: " << e.what() << std::endl;
                    }
                    catch (...) {
                        std::cerr << "线程 " << threadId << " 执行任务发生未知异常" << std::endl;
                    }

                    stats.lastActivity = std::chrono::system_clock::now();
                } else {
                    // 没有任务，进入空闲状态
                    auto idleStart = std::chrono::high_resolution_clock::now();

                    std::unique_lock<std::mutex> lock(globalQueueMutex_);
                    condition_.wait_for(lock, config_.idleTimeout, [this] {
                        return !globalQueue_.empty() || state_ != PoolState::Running;
                    });

                    auto idleEnd = std::chrono::high_resolution_clock::now();
                    auto idleTime = std::chrono::duration_cast<std::chrono::microseconds>(idleEnd - idleStart);
                    stats.idleTime += idleTime.count();
                }

                // 检查动态调整
                if (config_.enableDynamicSizing && threadId == 0) {
                    checkDynamicResize();
                }
            }

            stats.isActive = false;
            std::cout << "工作线程 " << threadId << " 退出" << std::endl;
        }

        bool ThreadPool::getTask(Task& task, uint32_t threadId) {
            // 首先尝试从本地队列获取任务
            if (config_.enableWorkStealing && workQueues_[threadId]) {
                if (workQueues_[threadId]->pop_front(task)) {
                    return true;
                }
            }

            // 然后尝试从全局队列获取任务
            {
                std::lock_guard<std::mutex> lock(globalQueueMutex_);
                if (!globalQueue_.empty()) {
                    task = globalQueue_.top();
                    globalQueue_.pop();
                    return true;
                }
            }

            // 最后尝试工作窃取
            if (config_.enableWorkStealing) {
                return stealWork(task, threadId);
            }

            return false;
        }

        bool ThreadPool::stealWork(Task& task, uint32_t thiefId) {
            // 随机选择受害者线程
            static thread_local std::random_device rd;
            static thread_local std::mt19937 gen(rd());

            std::uniform_int_distribution<> dis(0, static_cast<int>(config_.threadCount - 1));

            for (size_t attempts = 0; attempts < config_.threadCount; ++attempts) {
                uint32_t victimId = static_cast<uint32_t>(dis(gen));

                if (victimId != thiefId && workQueues_[victimId]) {
                    if (workQueues_[victimId]->pop_back(task)) {
                        ++threadStats_[thiefId]->workSteals;
                        return true;
                    }
                }
            }

            return false;
        }

        void ThreadPool::updateThreadStats(uint32_t threadId, const Task& task,
                                           std::chrono::microseconds executionTime) {
            ThreadStats& stats = *threadStats_[threadId];

            stats.totalWorkTime += executionTime.count();
            taskStats_.totalExecutionTime += executionTime.count();

            // 计算等待时间
            auto waitTime = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now() - task.submitTime);

            // 更新平均等待时间（使用指数移动平均）
            const double alpha = 0.1;
            taskStats_.averageWaitTime = static_cast<uint32_t>(
                    alpha * waitTime.count() + (1 - alpha) * taskStats_.averageWaitTime.load());
        }

        void ThreadPool::checkDynamicResize() {
            auto now = std::chrono::system_clock::now();
            auto timeSinceLastCheck = std::chrono::duration_cast<std::chrono::seconds>(now - lastResizeCheck_);

            if (timeSinceLastCheck.count() < 5) return; // 每5秒检查一次

            lastResizeCheck_ = now;

            size_t queueSize = getQueueSize();
            size_t activeThreads = getActiveThreadCount();
            double utilization = getUtilization();

            // 扩容条件：队列积压且利用率高
            if (queueSize > config_.threadCount * 2 && utilization > 0.8 &&
                config_.threadCount < config_.maxThreads) {

                size_t newSize = std::min(config_.threadCount + 1, config_.maxThreads);
                std::cout << "动态扩容线程池: " << config_.threadCount << " -> " << newSize << std::endl;
                resizePool(newSize);
            }
                // 缩容条件：长时间低利用率
            else if (utilization < 0.3 && config_.threadCount > config_.minThreads) {
                size_t newSize = std::max(config_.threadCount - 1, config_.minThreads);
                std::cout << "动态缩容线程池: " << config_.threadCount << " -> " << newSize << std::endl;
                resizePool(newSize);
            }
        }

        void ThreadPool::resizePool(size_t newSize) {
            if (newSize == config_.threadCount || state_ != PoolState::Running) {
                return;
            }

            if (newSize > config_.threadCount) {
                // 扩容：添加新线程
                size_t oldSize = config_.threadCount;
                config_.threadCount = newSize;

                // 扩展统计对象和队列
                for (size_t i = oldSize; i < newSize; ++i) {
                    threadStats_.emplace_back(std::make_unique<ThreadStats>());
                    threadStats_[i]->threadId = static_cast<uint32_t>(i);

                    if (config_.enableWorkStealing) {
                        workQueues_.emplace_back(std::make_unique<WorkStealingQueue>());
                    }

                    workers_.emplace_back(&ThreadPool::workerFunction, this, static_cast<uint32_t>(i));

                    if (config_.enableThreadAffinity) {
                        setThreadAffinity(static_cast<uint32_t>(i),
                                          static_cast<uint32_t>(i % std::thread::hardware_concurrency()));
                    }
                }
            } else {
                // 缩容：移除线程（通过设置标志，让线程自然退出）
                config_.threadCount = newSize;
                // 注意：实际的线程移除会在下次重启时生效，或者需要更复杂的实现
            }
        }

        std::vector<std::future<void>> ThreadPool::submitBatch(
                const std::vector<std::function<void()>>& tasks, Priority priority) {

            std::vector<std::future<void>> futures;
            futures.reserve(tasks.size());

            for (const auto& task : tasks) {
                futures.push_back(submit(priority, task));
            }

            return futures;
        }

        void ThreadPool::submitParallelFor(size_t begin, size_t end,
                                           std::function<void(size_t)> func,
                                           size_t chunkSize, Priority priority) {
            if (begin >= end) return;

            if (chunkSize == 0) {
                chunkSize = std::max(size_t(1), (end - begin) / (config_.threadCount * 4));
            }

            std::vector<std::future<void>> futures;

            for (size_t i = begin; i < end; i += chunkSize) {
                size_t chunkEnd = std::min(i + chunkSize, end);

                auto chunkTask = [func, i, chunkEnd]() {
                    for (size_t j = i; j < chunkEnd; ++j) {
                        func(j);
                    }
                };

                futures.push_back(submit(priority, chunkTask));
            }

            // 等待所有块完成
            for (auto& future : futures) {
                future.wait();
            }
        }

        bool ThreadPool::waitForAll(uint32_t timeout) {
            auto startTime = std::chrono::system_clock::now();

            while (true) {
                bool allDone = true;

                // 检查全局队列
                {
                    std::lock_guard<std::mutex> lock(globalQueueMutex_);
                    if (!globalQueue_.empty()) {
                        allDone = false;
                    }
                }

                // 检查工作窃取队列
                if (allDone && config_.enableWorkStealing) {
                    for (const auto& queue : workQueues_) {
                        if (queue && !queue->empty()) {
                            allDone = false;
                            break;
                        }
                    }
                }

                // 检查是否有线程在工作
                if (allDone) {
                    for (const auto& stats : threadStats_) {
                        if (stats->isActive.load()) {
                            allDone = false;
                            break;
                        }
                    }
                }

                if (allDone) return true;

                if (timeout > 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now() - startTime);
                    if (elapsed.count() >= timeout) {
                        return false;
                    }
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        size_t ThreadPool::clearQueue() {
            size_t cleared = 0;

            // 清空全局队列
            {
                std::lock_guard<std::mutex> lock(globalQueueMutex_);
                cleared += globalQueue_.size();
                while (!globalQueue_.empty()) {
                    globalQueue_.pop();
                }
            }

            // 清空工作窃取队列
            if (config_.enableWorkStealing) {
                for (auto& queue : workQueues_) {
                    if (queue) {
                        cleared += queue->size();
                        queue->clear();
                    }
                }
            }

            std::cout << "清空了 " << cleared << " 个待处理任务" << std::endl;
            return cleared;
        }

        void ThreadPool::setThreadAffinity(uint32_t threadId, uint32_t cpuId) {
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

// 统计和监控实现
        ThreadPool::TaskStats ThreadPool::getTaskStats() const {
            TaskStats stats{};
            stats.tasksSubmitted = taskStats_.tasksSubmitted.load();
            stats.tasksCompleted = taskStats_.tasksCompleted.load();
            stats.tasksRejected = taskStats_.tasksRejected.load();
            stats.totalExecutionTime = taskStats_.totalExecutionTime.load();
            stats.averageWaitTime = taskStats_.averageWaitTime.load();
            stats.peakQueueSize = taskStats_.peakQueueSize.load();
            stats.startTime = taskStats_.startTime;
            return stats;
        }

        ThreadPool::ThreadStats ThreadPool::getThreadStats(uint32_t threadId) const {
            if (threadId >= threadStats_.size()) {
                return ThreadStats{};
            }
            const auto& src = *threadStats_[threadId];
            ThreadStats stats{};
            stats.threadId = src.threadId;
            stats.tasksExecuted = src.tasksExecuted.load();
            stats.totalWorkTime = src.totalWorkTime.load();
            stats.idleTime = src.idleTime.load();
            stats.workSteals = src.workSteals.load();
            stats.isActive = src.isActive.load();
            stats.lastActivity = src.lastActivity;
            return stats;
        }

        void ThreadPool::printStats() const {
            std::cout << "=== 线程池统计信息 ===" << std::endl;
            std::cout << "任务提交: " << taskStats_.tasksSubmitted.load() << std::endl;
            std::cout << "任务完成: " << taskStats_.tasksCompleted.load() << std::endl;
            std::cout << "任务拒绝: " << taskStats_.tasksRejected.load() << std::endl;
            std::cout << "队列峰值: " << taskStats_.peakQueueSize.load() << std::endl;
            std::cout << "平均等待: " << taskStats_.averageWaitTime.load() << " μs" << std::endl;
            std::cout << "线程利用率: " << std::fixed << std::setprecision(2)
                      << getUtilization() * 100 << "%" << std::endl;

            std::cout << "\n=== 线程详细信息 ===" << std::endl;
            for (size_t i = 0; i < threadStats_.size(); ++i) {
                const auto& stats = *threadStats_[i];
                std::cout << "线程 " << i << ": "
                          << "执行=" << stats.tasksExecuted.load() << ", "
                          << "工作时间=" << stats.totalWorkTime.load() / 1000 << "ms, "
                          << "空闲时间=" << stats.idleTime.load() / 1000 << "ms, "
                          << "窃取=" << stats.workSteals.load() << std::endl;
            }
        }

        double ThreadPool::getUtilization() const {
            if (threadStats_.empty()) return 0.0;

            uint64_t totalWorkTime = 0;
            uint64_t totalIdleTime = 0;

            for (const auto& stats : threadStats_) {
                totalWorkTime += stats->totalWorkTime.load();
                totalIdleTime += stats->idleTime.load();
            }

            uint64_t totalTime = totalWorkTime + totalIdleTime;
            return totalTime > 0 ? static_cast<double>(totalWorkTime) / totalTime : 0.0;
        }

        size_t ThreadPool::getQueueSize() const {
            size_t size = 0;

            {
                std::lock_guard<std::mutex> lock(globalQueueMutex_);
                size += globalQueue_.size();
            }

            if (config_.enableWorkStealing) {
                for (const auto& queue : workQueues_) {
                    if (queue) {
                        size += queue->size();
                    }
                }
            }

            return size;
        }

        size_t ThreadPool::getActiveThreadCount() const {
            size_t count = 0;
            for (const auto& stats : threadStats_) {
                if (stats->isActive.load()) {
                    ++count;
                }
            }
            return count;
        }

        ThreadPool::PoolState ThreadPool::getState() const {
            return state_.load();
        }

    } // namespace Core
} // namespace OceanSimulation