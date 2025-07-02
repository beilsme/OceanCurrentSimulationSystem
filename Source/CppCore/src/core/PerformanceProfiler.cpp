// src/core/PerformanceProfiler.cpp
#include "core/PerformanceProfiler.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <thread>
#include <fstream>
#include <mutex>
#include <filesystem>   // C++17
namespace fs = std::filesystem;

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#endif

namespace OceanSim {
    namespace Core {

        // ========================= Timer 类实现 =========================

        PerformanceProfiler::Timer::Timer(const std::string& name, PerformanceProfiler* profiler)
                : name_(name), profiler_(profiler), start_time_(std::chrono::high_resolution_clock::now()) {
        }

        PerformanceProfiler::Timer::~Timer() {
            if (!stopped_) {
                stop();
            }
        }

        void PerformanceProfiler::Timer::stop() {
            if (!stopped_) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
                double elapsed_seconds = duration.count() / 1000000.0;

                if (profiler_ && profiler_->isEnabled()) {
                    profiler_->recordExecution(name_, elapsed_seconds);
                }
                stopped_ = true;
            }
        }

        double PerformanceProfiler::Timer::getElapsedTime() const {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time_);
            return duration.count() / 1000000.0;
        }

        // ========================= MemoryMonitor 类实现 =========================

        PerformanceProfiler::MemoryMonitor::MemoryMonitor() : peak_memory_(0) {
        }

        size_t PerformanceProfiler::MemoryMonitor::getCurrentMemoryUsage() {
            size_t memory_usage = 0;

#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
                memory_usage = pmc.WorkingSetSize;
            }
#elif defined(__linux__)
            std::ifstream status_file("/proc/self/status");
            std::string line;
            while (std::getline(status_file, line)) {
                if (line.substr(0, 6) == "VmRSS:") {
                    std::istringstream iss(line.substr(6));
                    size_t kb;
                    if (iss >> kb) {
                        memory_usage = kb * 1024; // 转换为字节
                    }
                    break;
                }
            }
#elif defined(__APPLE__)
            struct mach_task_basic_info info;
            mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                          (task_info_t)&info, &info_count) == KERN_SUCCESS) {
                memory_usage = info.resident_size;
            }
#endif

            return memory_usage;
        }

        size_t PerformanceProfiler::MemoryMonitor::getPeakMemoryUsage() {
            size_t current = getCurrentMemoryUsage();
            peak_memory_ = std::max(peak_memory_, current);
            return peak_memory_;
        }

        void PerformanceProfiler::MemoryMonitor::recordMemoryUsage(const std::string& tag) {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            size_t current_usage = getCurrentMemoryUsage();
            memory_history_.emplace_back(tag, current_usage);
            peak_memory_ = std::max(peak_memory_, current_usage);
        }

        std::vector<std::pair<std::string, size_t>> PerformanceProfiler::MemoryMonitor::getMemoryHistory() const {
            std::lock_guard<std::mutex> lock(memory_mutex_);
            return memory_history_;
        }

        // ========================= PerformanceProfiler 主类实现 =========================

        PerformanceProfiler::PerformanceProfiler()
                : enabled_(false), memory_monitor_(std::make_unique<MemoryMonitor>()) {
        }

        PerformanceProfiler::~PerformanceProfiler() {
            if (real_time_monitoring_) {
                stopRealTimeMonitoring();
            }
        }

        PerformanceProfiler& PerformanceProfiler::getInstance() {
            static PerformanceProfiler instance;
            return instance;
        }

        void PerformanceProfiler::enable(bool enabled) {
            std::lock_guard<std::mutex> lock(data_mutex_);
            enabled_ = enabled;
        }

        void PerformanceProfiler::disable() {
            enable(false);
        }

        void PerformanceProfiler::startTimer(const std::string& name) {
            if (!enabled_) return;

            std::lock_guard<std::mutex> lock(data_mutex_);
            active_timers_[name] = std::chrono::high_resolution_clock::now();
        }

        void PerformanceProfiler::stopTimer(const std::string& name) {
            if (!enabled_) return;

            auto end_time = std::chrono::high_resolution_clock::now();

            std::lock_guard<std::mutex> lock(data_mutex_);
            auto it = active_timers_.find(name);
            if (it != active_timers_.end()) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - it->second);
                double elapsed_seconds = duration.count() / 1000000.0;

                recordExecution(name, elapsed_seconds);
                active_timers_.erase(it);
            }
        }

        std::unique_ptr<PerformanceProfiler::Timer> PerformanceProfiler::createTimer(const std::string& name) {
            if (!enabled_) return nullptr;
            return std::make_unique<Timer>(name, this);
        }

        void PerformanceProfiler::recordExecution(const std::string& name, double time) {
            if (!enabled_) return;

            std::lock_guard<std::mutex> lock(data_mutex_);

            auto it = profile_data_.find(name);
            if (it == profile_data_.end()) {
                profile_data_.emplace(name, ProfileData(name));
                it = profile_data_.find(name);
            }

            it->second.updateTime(time);

            // 记录内存使用
            size_t current_memory = memory_monitor_->getCurrentMemoryUsage();
            it->second.memory_usage = std::max(it->second.memory_usage, current_memory);
        }

        void PerformanceProfiler::recordMemoryUsage(const std::string& name, size_t memory_bytes) {
            if (!enabled_) return;

            std::lock_guard<std::mutex> lock(data_mutex_);
            auto it = profile_data_.find(name);
            if (it != profile_data_.end()) {
                it->second.memory_usage = std::max(it->second.memory_usage, memory_bytes);
            }

            memory_monitor_->recordMemoryUsage(name);
        }

        void PerformanceProfiler::recordParallelMetrics(const std::string& name, int thread_count, double efficiency) {
            if (!enabled_) return;

            std::lock_guard<std::mutex> lock(data_mutex_);
            auto it = profile_data_.find(name);
            if (it != profile_data_.end()) {
                it->second.thread_count = thread_count;
                it->second.parallel_efficiency = efficiency;
            }
        }

        const PerformanceProfiler::ProfileData* PerformanceProfiler::getProfileData(const std::string& name) const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            auto it = profile_data_.find(name);
            return (it != profile_data_.end()) ? &it->second : nullptr;
        }

        std::vector<PerformanceProfiler::ProfileData> PerformanceProfiler::getAllProfileData() const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            std::vector<ProfileData> result;
            result.reserve(profile_data_.size());

            for (const auto& pair : profile_data_) {
                result.push_back(pair.second);
            }

            return result;
        }

        double PerformanceProfiler::getTotalTime() const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            double total = 0.0;
            for (const auto& pair : profile_data_) {
                total += pair.second.total_time;
            }
            return total;
        }

        size_t PerformanceProfiler::getTotalMemoryUsage() const {
            return memory_monitor_->getCurrentMemoryUsage();
        }

        /* === getElapsedTime 实现 === */
        double PerformanceProfiler::getElapsedTime(const std::string& name) const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            auto it = profile_data_.find(name);
            return (it != profile_data_.end()) ? it->second.total_time : 0.0;
        }

        /* === generateReport 实现 === */
        void PerformanceProfiler::generateReport(const std::string& filename) const {
            if (filename.empty()) return;

            fs::path path(filename);
            std::string ext = path.extension().string();

            try {
                if (ext == ".json" || ext == ".JSON") {
                    exportToJSON(filename);
                } else if (ext == ".csv" || ext == ".CSV") {
                    exportToCSV(filename);
                } else {                 // 其它一律写纯文本
                    std::ofstream ofs(filename);
                    if (!ofs) throw std::runtime_error("无法打开文件: " + filename);
                    printDetailedReport(ofs);
                }
            } catch (const std::exception& e) {
                std::cerr << "[PerformanceProfiler] 生成报告失败: "
                          << e.what() << std::endl;
            }
        }
        
        void PerformanceProfiler::printSummary(std::ostream& os) const {
            std::lock_guard<std::mutex> lock(data_mutex_);

            os << "\n===== 性能分析摘要 =====\n";
            os << std::fixed << std::setprecision(6);

            if (profile_data_.empty()) {
                os << "没有性能数据可显示\n";
                return;
            }

            os << std::setw(25) << "操作名称"
               << std::setw(12) << "总时间(s)"
               << std::setw(12) << "平均时间(s)"
               << std::setw(10) << "调用次数"
               << std::setw(15) << "内存使用(MB)" << "\n";
            os << std::string(80, '-') << "\n";

            for (const auto& pair : profile_data_) {
                const auto& data = pair.second;
                os << std::setw(25) << data.name.substr(0, 24)
                   << std::setw(12) << data.total_time
                   << std::setw(12) << data.avg_time
                   << std::setw(10) << data.call_count
                   << std::setw(15) << (data.memory_usage / (1024.0 * 1024.0)) << "\n";
            }

            os << std::string(80, '-') << "\n";
            os << "总执行时间: " << formatTime(getTotalTime()) << "\n";
            os << "当前内存使用: " << formatMemory(getTotalMemoryUsage()) << "\n";
            os << "峰值内存使用: " << formatMemory(memory_monitor_->getPeakMemoryUsage()) << "\n";
        }

        void PerformanceProfiler::printDetailedReport(std::ostream& os) const {
            std::lock_guard<std::mutex> lock(data_mutex_);

            os << "\n===== 详细性能报告 =====\n";

            for (const auto& pair : profile_data_) {
                const auto& data = pair.second;
                os << "\n操作: " << data.name << "\n";
                os << "  总时间: " << formatTime(data.total_time) << "\n";
                os << "  平均时间: " << formatTime(data.avg_time) << "\n";
                os << "  最小时间: " << formatTime(data.min_time) << "\n";
                os << "  最大时间: " << formatTime(data.max_time) << "\n";
                os << "  调用次数: " << data.call_count << "\n";
                os << "  内存使用: " << formatMemory(data.memory_usage) << "\n";

                if (data.thread_count > 0) {
                    os << "  线程数: " << data.thread_count << "\n";
                    os << "  并行效率: " << formatPercentage(data.parallel_efficiency) << "\n";
                }
            }

            // 显示内存历史
            auto memory_history = memory_monitor_->getMemoryHistory();
            if (!memory_history.empty()) {
                os << "\n内存使用历史:\n";
                for (const auto& entry : memory_history) {
                    os << "  " << entry.first << ": " << formatMemory(entry.second) << "\n";
                }
            }
        }

        void PerformanceProfiler::exportToJSON(const std::string& filename) const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            std::ofstream file(filename);

            if (!file.is_open()) {
                throw std::runtime_error("无法打开文件进行JSON导出: " + filename);
            }

            file << "{\n";
            file << "  \"performance_data\": [\n";

            bool first = true;
            for (const auto& pair : profile_data_) {
                if (!first) file << ",\n";
                first = false;

                const auto& data = pair.second;
                file << "    {\n";
                file << "      \"name\": \"" << data.name << "\",\n";
                file << "      \"total_time\": " << data.total_time << ",\n";
                file << "      \"avg_time\": " << data.avg_time << ",\n";
                file << "      \"min_time\": " << data.min_time << ",\n";
                file << "      \"max_time\": " << data.max_time << ",\n";
                file << "      \"call_count\": " << data.call_count << ",\n";
                file << "      \"memory_usage\": " << data.memory_usage << ",\n";
                file << "      \"thread_count\": " << data.thread_count << ",\n";
                file << "      \"parallel_efficiency\": " << data.parallel_efficiency << "\n";
                file << "    }";
            }

            file << "\n  ],\n";
            file << "  \"summary\": {\n";
            file << "    \"total_time\": " << getTotalTime() << ",\n";
            file << "    \"current_memory\": " << getTotalMemoryUsage() << ",\n";
            file << "    \"peak_memory\": " << memory_monitor_->getPeakMemoryUsage() << "\n";
            file << "  }\n";
            file << "}\n";
        }

        void PerformanceProfiler::exportToCSV(const std::string& filename) const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            std::ofstream file(filename);

            if (!file.is_open()) {
                throw std::runtime_error("无法打开文件进行CSV导出: " + filename);
            }

            // CSV 头部
            file << "Name,TotalTime,AvgTime,MinTime,MaxTime,CallCount,MemoryUsage,ThreadCount,ParallelEfficiency\n";

            for (const auto& pair : profile_data_) {
                const auto& data = pair.second;
                file << data.name << ","
                     << data.total_time << ","
                     << data.avg_time << ","
                     << data.min_time << ","
                     << data.max_time << ","
                     << data.call_count << ","
                     << data.memory_usage << ","
                     << data.thread_count << ","
                     << data.parallel_efficiency << "\n";
            }
        }

        void PerformanceProfiler::reset() {
            std::lock_guard<std::mutex> lock(data_mutex_);
            profile_data_.clear();
            active_timers_.clear();
        }

        void PerformanceProfiler::clear() {
            reset();
        }

        void PerformanceProfiler::startRealTimeMonitoring(double interval_seconds) {
            if (real_time_monitoring_) return;

            monitoring_interval_ = interval_seconds;
            real_time_monitoring_ = true;

            monitoring_thread_ = std::make_unique<std::thread>([this]() {
                while (real_time_monitoring_) {
                    updateSystemMetrics();
                    std::this_thread::sleep_for(
                            std::chrono::milliseconds(static_cast<int>(monitoring_interval_ * 1000))
                    );
                }
            });
        }

        void PerformanceProfiler::stopRealTimeMonitoring() {
            real_time_monitoring_ = false;
            if (monitoring_thread_ && monitoring_thread_->joinable()) {
                monitoring_thread_->join();
            }
            monitoring_thread_.reset();
        }

        std::vector<std::string> PerformanceProfiler::detectBottlenecks(double threshold_percentage) const {
            std::lock_guard<std::mutex> lock(data_mutex_);
            std::vector<std::string> bottlenecks;

            double total_time = getTotalTime();
            if (total_time == 0.0) return bottlenecks;

            for (const auto& pair : profile_data_) {
                const auto& data = pair.second;
                double percentage = (data.total_time / total_time) * 100.0;

                if (percentage >= threshold_percentage) {
                    bottlenecks.push_back(data.name);
                }
            }

            // 按时间消耗排序
            std::sort(bottlenecks.begin(), bottlenecks.end(), [this](const std::string& a, const std::string& b) {
                auto it_a = profile_data_.find(a);
                auto it_b = profile_data_.find(b);
                return it_a->second.total_time > it_b->second.total_time;
            });

            return bottlenecks;
        }

        std::string PerformanceProfiler::getMostExpensiveOperation() const {
            std::lock_guard<std::mutex> lock(data_mutex_);

            if (profile_data_.empty()) return "";

            auto max_it = std::max_element(profile_data_.begin(), profile_data_.end(),
                                           [](const auto& a, const auto& b) {
                                               return a.second.total_time < b.second.total_time;
                                           });

            return max_it->first;
        }

        bool PerformanceProfiler::detectMemoryLeaks() const {
            auto memory_history = memory_monitor_->getMemoryHistory();
            if (memory_history.size() < 2) return false;

            // 简单的内存泄漏检测：如果内存使用持续增长
            size_t consecutive_increases = 0;
            for (size_t i = 1; i < memory_history.size(); ++i) {
                if (memory_history[i].second > memory_history[i-1].second) {
                    consecutive_increases++;
                } else {
                    consecutive_increases = 0;
                }

                if (consecutive_increases >= 5) {  // 连续5次增长
                    return true;
                }
            }

            return false;
        }

        double PerformanceProfiler::getOverallParallelEfficiency() const {
            std::lock_guard<std::mutex> lock(data_mutex_);

            double total_efficiency = 0.0;
            int count = 0;

            for (const auto& pair : profile_data_) {
                if (pair.second.parallel_efficiency > 0.0) {
                    total_efficiency += pair.second.parallel_efficiency;
                    count++;
                }
            }

            return (count > 0) ? total_efficiency / count : 0.0;
        }

        void PerformanceProfiler::runMicrobenchmark(const std::string& name,
                                                    const std::function<void()>& operation,
                                                    int iterations) {
            if (!enabled_) return;

            std::vector<double> times;
            times.reserve(iterations);

            for (int i = 0; i < iterations; ++i) {
                auto timer = createTimer(name + "_bench_" + std::to_string(i));
                operation();
                // timer 会在作用域结束时自动停止并记录时间
            }
        }

        // 私有方法实现

        void PerformanceProfiler::updateSystemMetrics() {
            if (!enabled_) return;

            memory_monitor_->recordMemoryUsage("系统监控");

            // 更新CPU使用率等系统指标
            double cpu_usage = getCurrentCPUUsage();
            int thread_count = getCurrentThreadCount();

            std::lock_guard<std::mutex> lock(data_mutex_);
            // 可以在这里更新全局系统指标
        }

        double PerformanceProfiler::getCurrentCPUUsage() {
            // 简化的CPU使用率获取，实际实现需要平台特定代码
#ifdef _WIN32
            FILETIME idle_time, kernel_time, user_time;
            if (GetSystemTimes(&idle_time, &kernel_time, &user_time)) {
                // 需要实现Windows特定的CPU使用率计算
                return 0.0; // 简化返回
            }
#elif defined(__linux__)
            // 读取 /proc/stat 来计算CPU使用率
            std::ifstream stat_file("/proc/stat");
            std::string cpu_line;
            if (std::getline(stat_file, cpu_line)) {
                // 解析CPU时间并计算使用率
                return 0.0; // 简化返回
            }
#endif
            return 0.0;
        }

        size_t PerformanceProfiler::getCurrentMemoryUsage() {
            return memory_monitor_->getCurrentMemoryUsage();
        }

        int PerformanceProfiler::getCurrentThreadCount() {
            return static_cast<int>(std::thread::hardware_concurrency());
        }

        std::string PerformanceProfiler::formatTime(double seconds) const {
            if (seconds < 1e-6) {
                return std::to_string(static_cast<int>(seconds * 1e9)) + " ns";
            } else if (seconds < 1e-3) {
                return std::to_string(static_cast<int>(seconds * 1e6)) + " μs";
            } else if (seconds < 1.0) {
                return std::to_string(static_cast<int>(seconds * 1e3)) + " ms";
            } else {
                return std::to_string(seconds) + " s";
            }
        }

        std::string PerformanceProfiler::formatMemory(size_t bytes) const {
            const char* units[] = {"B", "KB", "MB", "GB", "TB"};
            int unit_index = 0;
            double size = static_cast<double>(bytes);

            while (size >= 1024.0 && unit_index < 4) {
                size /= 1024.0;
                unit_index++;
            }

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
            return oss.str();
        }

        std::string PerformanceProfiler::formatPercentage(double percentage) const {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << (percentage * 100.0) << "%";
            return oss.str();
        }

        // ========================= PerformanceCounters 静态成员实现 =========================

        std::atomic<size_t> PerformanceCounters::total_particles_{0};
        std::atomic<size_t> PerformanceCounters::total_timesteps_{0};
        std::atomic<size_t> PerformanceCounters::cache_hits_{0};
        std::atomic<size_t> PerformanceCounters::cache_misses_{0};
        std::atomic<double> PerformanceCounters::total_iteration_time_{0.0};
        std::atomic<size_t> PerformanceCounters::iteration_count_{0};
        std::mutex PerformanceCounters::parallel_data_mutex_;
        std::vector<std::pair<int, double>> PerformanceCounters::parallel_times_;

        void PerformanceCounters::incrementParticleCount(size_t count) {
            total_particles_ += count;
        }

        void PerformanceCounters::incrementTimeSteps() {
            total_timesteps_++;
        }

        void PerformanceCounters::recordIterationTime(double time) {
            double current = total_iteration_time_.load(std::memory_order_relaxed);
            while (!total_iteration_time_.compare_exchange_weak(
                    current, current + time, std::memory_order_relaxed)) {
                /* retry */
            }
            iteration_count_++;
        }

        void PerformanceCounters::recordParallelTime(int thread_count, double time) {
            std::lock_guard<std::mutex> lock(parallel_data_mutex_);
            parallel_times_.emplace_back(thread_count, time);
        }

        void PerformanceCounters::recordCacheHit() {
            cache_hits_++;
        }

        void PerformanceCounters::recordCacheMiss() {
            cache_misses_++;
        }

        size_t PerformanceCounters::getTotalParticleCount() {
            return total_particles_;
        }

        size_t PerformanceCounters::getTotalTimeSteps() {
            return total_timesteps_;
        }

        double PerformanceCounters::getAverageIterationTime() {
            size_t count = iteration_count_;
            return (count > 0) ? (total_iteration_time_ / count) : 0.0;
        }

        double PerformanceCounters::getCacheHitRatio() {
            size_t hits = cache_hits_;
            size_t misses = cache_misses_;
            size_t total = hits + misses;
            return (total > 0) ? (static_cast<double>(hits) / total) : 0.0;
        }

        double PerformanceCounters::getAverageParallelEfficiency() {
            std::lock_guard<std::mutex> lock(parallel_data_mutex_);
            if (parallel_times_.empty()) return 0.0;

            double total_efficiency = 0.0;
            for (const auto& entry : parallel_times_) {
                // 简化的并行效率计算：理想时间 / 实际时间
                double ideal_time = entry.second / entry.first;  // 假设线性加速
                double efficiency = ideal_time / entry.second;
                total_efficiency += std::min(efficiency, 1.0);  // 效率不超过100%
            }

            return total_efficiency / parallel_times_.size();
        }

        void PerformanceCounters::reset() {
            total_particles_ = 0;
            total_timesteps_ = 0;
            cache_hits_ = 0;
            cache_misses_ = 0;
            total_iteration_time_ = 0.0;
            iteration_count_ = 0;

            std::lock_guard<std::mutex> lock(parallel_data_mutex_);
            parallel_times_.clear();
        }

        // ========================= GPU Profiler 实现 =========================

        std::unordered_map<std::string, GPUProfiler::GPUMetrics> GPUProfiler::gpu_metrics_;
        std::mutex GPUProfiler::gpu_mutex_;

        void GPUProfiler::startKernelTimer(const std::string& kernel_name) {
            // GPU计时实现，需要CUDA/OpenCL特定代码
            // 这里提供框架，实际使用时需要添加GPU特定的计时代码
        }

        void GPUProfiler::stopKernelTimer(const std::string& kernel_name) {
            // GPU计时停止实现
        }

        void GPUProfiler::recordMemoryTransfer(size_t bytes, bool host_to_device) {
            // 记录GPU内存传输
        }

        GPUProfiler::GPUMetrics GPUProfiler::getGPUMetrics(const std::string& kernel_name) {
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            auto it = gpu_metrics_.find(kernel_name);
            return (it != gpu_metrics_.end()) ? it->second : GPUMetrics{};
        }

        void GPUProfiler::printGPUSummary(std::ostream& os) {
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            os << "\n===== GPU性能摘要 =====\n";

            for (const auto& pair : gpu_metrics_) {
                const auto& metrics = pair.second;
                os << "Kernel: " << pair.first << "\n";
                os << "  执行时间: " << metrics.kernel_time << " ms\n";
                os << "  内存传输时间: " << metrics.memory_transfer_time << " ms\n";
                os << "  内存使用: " << (metrics.memory_usage / (1024*1024)) << " MB\n";
                os << "  GPU利用率: " << (metrics.gpu_utilization * 100) << "%\n";
            }
        }

    } // namespace Core
} // namespace OceanSim