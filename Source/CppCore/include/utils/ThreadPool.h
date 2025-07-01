#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>
#include <chrono>
#include <unordered_map>

namespace OceanSimulation {
    namespace Core {

/**
 * @brief 高性能线程池
 * 
 * 为海洋模拟系统提供高效的线程池实现，支持任务优先级、工作窃取、
 * 线程亲和性设置和动态负载均衡。
 */
        class ThreadPool {
        public:
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
             * @brief 线程池状态
             */
            enum class PoolState {
                Stopped,
                Starting,
                Running,
                Stopping,
                Paused
            };

            /**
             * @brief 负载均衡策略
             */
            enum class LoadBalanceStrategy {
                RoundRobin,     // 轮询
                LeastLoaded,    // 最少负载
                WorkStealing,   // 工作窃取
                Affinity        // 亲和性优先
            };

            /**
             * @brief 线程池配置
             */
            struct Config {
                size_t threadCount = std::thread::hardware_concurrency();
                size_t maxQueueSize = 10000;
                LoadBalanceStrategy strategy = LoadBalanceStrategy::WorkStealing;
                bool enableWorkStealing = true;
                bool enableThreadAffinity = false;
                bool enableDynamicSizing = true;
                size_t minThreads = 2;
                size_t maxThreads = std::thread::hardware_concurrency() * 2;
                std::chrono::milliseconds idleTimeout{5000}; // 空闲超时
                bool enablePriorityQueue = true;
                bool enableStatistics = true;
            };

            /**
             * @brief 任务统计信息
             */
            struct TaskStats {
                std::atomic<uint64_t> tasksSubmitted{0};
                std::atomic<uint64_t> tasksCompleted{0};
                std::atomic<uint64_t> tasksRejected{0};
                std::atomic<uint64_t> totalExecutionTime{0}; // 微秒
                std::atomic<uint32_t> averageWaitTime{0};    // 微秒
                std::atomic<uint32_t> peakQueueSize{0};
                std::chrono::system_clock::time_point startTime;
            };

            /**
             * @brief 线程统计信息
             */
            struct ThreadStats {
                uint32_t threadId;
                std::atomic<uint64_t> tasksExecuted{0};
                std::atomic<uint64_t> totalWorkTime{0};  // 微秒
                std::atomic<uint64_t> idleTime{0};       // 微秒
                std::atomic<uint32_t> workSteals{0};
                std::atomic<bool> isActive{false};
                std::chrono::system_clock::time_point lastActivity;
            };

        private:
            /**
             * @brief 任务包装器
             */
            struct Task {
                std::function<void()> function;
                Priority priority;
                std::chrono::system_clock::time_point submitTime;
                uint32_t targetThread; // 目标线程ID（用于亲和性）

                Task() : priority(Priority::Normal), targetThread(UINT32_MAX) {}

                Task(std::function<void()> f, Priority p = Priority::Normal, uint32_t target = UINT32_MAX)
                        : function(std::move(f)), priority(p), targetThread(target) {
                    submitTime = std::chrono::system_clock::now();
                }

                bool operator<(const Task& other) const {
                    return priority < other.priority;
                }
            };

            /**
             * @brief 工作窃取队列
             */
            class WorkStealingQueue {
            private:
                mutable std::mutex mutex_;
                std::deque<Task> queue_;

            public:
                void push_back(Task&& task);
                bool pop_front(Task& task);
                bool pop_back(Task& task); // 用于工作窃取
                size_t size() const;
                bool empty() const;
                void clear();
            };

            Config config_;
            std::atomic<PoolState> state_;

            // 线程管理
            std::vector<std::thread> workers_;
            std::vector<std::unique_ptr<ThreadStats>> threadStats_;
            std::vector<std::unique_ptr<WorkStealingQueue>> workQueues_;

            // 全局任务队列（优先级队列）
            std::priority_queue<Task> globalQueue_;
            mutable std::mutex globalQueueMutex_;
            std::condition_variable condition_;

            // 统计信息
            TaskStats taskStats_;
            mutable std::mutex statsMutex_;

            // 线程本地存储
            thread_local static uint32_t threadLocalId_;

            // 动态调整
            std::atomic<bool> needResize_{false};
            std::chrono::system_clock::time_point lastResizeCheck_;

            // 内部方法
            void workerFunction(uint32_t threadId);
            bool getTask(Task& task, uint32_t threadId);
            bool stealWork(Task& task, uint32_t thiefId);
            void updateThreadStats(uint32_t threadId, const Task& task,
                                   std::chrono::microseconds executionTime);
            void checkDynamicResize();
            void resizePool(size_t newSize);
           

        public:
            /**
             * @brief 构造函数
             * @param config 线程池配置
             */
            explicit ThreadPool(const Config& config = Config{});

            /**
             * @brief 析构函数
             */
            ~ThreadPool();

            /**
             * @brief 启动线程池
             */
            void start();

            /**
             * @brief 停止线程池
             * @param graceful 是否优雅停止（等待任务完成）
             */
            void stop(bool graceful = true);

            /**
             * @brief 暂停线程池
             */
            void pause();

            /**
             * @brief 恢复线程池
             */
            void resume();

            // ===========================================
            // 任务提交接口
            // ===========================================

            /**
             * @brief 提交任务
             * @tparam F 函数类型
             * @tparam Args 参数类型
             * @param priority 任务优先级
             * @param f 要执行的函数
             * @param args 函数参数
             * @return future对象用于获取结果
             */
            template<class F, class... Args>
            auto submit(Priority priority, F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type>;

            /**
             * @brief 提交任务（默认优先级）
             */
            template<class F, class... Args>
            auto submit(F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type>;

            /**
             * @brief 提交任务到指定线程
             * @param threadId 目标线程ID
             * @param priority 任务优先级
             * @param f 要执行的函数
             * @param args 函数参数
             */
            template<class F, class... Args>
            auto submitToThread(uint32_t threadId, Priority priority, F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type>;

            /**
             * @brief 批量提交任务
             * @param tasks 任务列表
             * @param priority 任务优先级
             * @return future列表
             */
            std::vector<std::future<void>> submitBatch(
                    const std::vector<std::function<void()>>& tasks,
                    Priority priority = Priority::Normal);

            /**
             * @brief 提交并行for循环任务
             * @param begin 起始索引
             * @param end 结束索引
             * @param func 循环体函数
             * @param chunkSize 分块大小
             * @param priority 任务优先级
             */
            void submitParallelFor(size_t begin, size_t end,
                                   std::function<void(size_t)> func,
                                   size_t chunkSize = 0,
                                   Priority priority = Priority::Normal);

            // ===========================================
            // 控制接口
            // ===========================================

            /**
             * @brief 等待所有任务完成
             * @param timeout 超时时间（毫秒），0表示无限等待
             * @return 是否在超时前完成
             */
            bool waitForAll(uint32_t timeout = 0);

            /**
             * @brief 清空任务队列
             * @return 清除的任务数量
             */
            size_t clearQueue();

            /**
             * @brief 设置线程数量
             * @param count 新的线程数量
             */
            void setThreadCount(size_t count);

            /**
             * @brief 获取线程数量
             */
            size_t getThreadCount() const;

            /**
             * @brief 获取队列大小
             */
            size_t getQueueSize() const;

            /**
             * @brief 获取活跃线程数
             */
            size_t getActiveThreadCount() const;

            /**
             * @brief 获取线程池状态
             */
            PoolState getState() const;

            // ===========================================
            // 统计和监控接口
            // ===========================================

            /**
             * @brief 获取任务统计信息
             */
            TaskStats getTaskStats() const;

            /**
             * @brief 获取线程统计信息
             * @param threadId 线程ID
             */
            ThreadStats getThreadStats(uint32_t threadId) const;

            /**
             * @brief 获取所有线程统计信息
             */
            std::vector<ThreadStats> getAllThreadStats() const;

            /**
             * @brief 重置统计信息
             */
            void resetStats();

            /**
             * @brief 打印统计报告
             */
            void printStats() const;

            /**
             * @brief 导出统计信息
             * @param filename 输出文件名
             */
            void exportStats(const std::string& filename) const;

            /**
             * @brief 获取线程利用率
             * @return 利用率百分比
             */
            double getUtilization() const;

            /**
             * @brief 获取平均任务等待时间
             * @return 等待时间（微秒）
             */
            uint32_t getAverageWaitTime() const;

            /**
             * @brief 获取平均任务执行时间
             * @return 执行时间（微秒）
             */
            uint32_t getAverageExecutionTime() const;

            // ===========================================
            // 配置管理接口
            // ===========================================

            /**
             * @brief 更新配置
             * @param newConfig 新配置
             */
            void updateConfig(const Config& newConfig);

            /**
             * @brief 获取当前配置
             */
            const Config& getConfig() const;

            /**
             * @brief 启用/禁用工作窃取
             * @param enabled 是否启用
             */
            void setWorkStealingEnabled(bool enabled);

            /**
             * @brief 设置负载均衡策略
             * @param strategy 策略类型
             */
            void setLoadBalanceStrategy(LoadBalanceStrategy strategy);

            /**
             * @brief 设置线程亲和性
             * @param threadId 线程ID
             * @param cpuId CPU核心ID
             */
            void setThreadAffinity(uint32_t threadId, uint32_t cpuId);

        private:
            // 禁用拷贝构造和赋值
            ThreadPool(const ThreadPool&) = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;
        };

// ===========================================
// 模板方法实现
// ===========================================

        template<class F, class... Args>
        auto ThreadPool::submit(Priority priority, F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {

            using return_type = typename std::result_of<F(Args...)>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> result = task->get_future();

            {
                std::unique_lock<std::mutex> lock(globalQueueMutex_);

                if (state_ != PoolState::Running) {
                    throw std::runtime_error("线程池未运行");
                }

                if (globalQueue_.size() >= config_.maxQueueSize) {
                    ++taskStats_.tasksRejected;
                    throw std::runtime_error("任务队列已满");
                }

                globalQueue_.emplace([task]() { (*task)(); }, priority);
                ++taskStats_.tasksSubmitted;

                if (globalQueue_.size() > taskStats_.peakQueueSize) {
                    taskStats_.peakQueueSize = globalQueue_.size();
                }
            }

            condition_.notify_one();
            return result;
        }

        template<class F, class... Args>
        auto ThreadPool::submit(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
            return submit(Priority::Normal, std::forward<F>(f), std::forward<Args>(args)...);
        }

        template<class F, class... Args>
        auto ThreadPool::submitToThread(uint32_t threadId, Priority priority, F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {

            using return_type = typename std::result_of<F(Args...)>::type;

            if (threadId >= config_.threadCount) {
                throw std::invalid_argument("无效的线程ID");
            }

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            std::future<return_type> result = task->get_future();

            if (config_.enableWorkStealing && workQueues_[threadId]) {
                workQueues_[threadId]->push_back(
                        Task([task]() { (*task)(); }, priority, threadId)
                );
                ++taskStats_.tasksSubmitted;
            } else {
                return submit(priority, std::forward<F>(f), std::forward<Args>(args)...);
            }

            return result;
        }

    } // namespace Core
} // namespace OceanSimulation

#endif // THREAD_POOL_H