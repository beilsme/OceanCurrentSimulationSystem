#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace OceanSim {
    namespace Utils {

/**
 * @brief 日志级别枚举
 */
        enum class LogLevel {
            DEBUG = 0,
            INFO = 1,
            WARNING = 2,
            ERROR = 3,
            CRITICAL = 4
        };

/**
 * @brief 线程安全的日志记录器类
 * 
 * 提供统一的日志记录接口，支持多种日志级别和输出目标
 * 在海洋洋流模拟系统中用于记录计算过程和性能监控
 */
        class Logger {
        public:
            /**
             * @brief 获取Logger单例实例
             * @return Logger实例引用
             */
            static Logger& getInstance();

            /**
             * @brief 设置日志文件路径
             * @param filepath 日志文件路径
             */
            void setLogFile(const std::string& filepath);

            /**
             * @brief 设置最小日志级别
             * @param level 最小日志级别
             */
            void setMinLogLevel(LogLevel level);

            /**
             * @brief 记录调试信息
             * @param message 日志消息
             */
            void debug(const std::string& message);

            /**
             * @brief 记录一般信息
             * @param message 日志消息
             */
            void info(const std::string& message);

            /**
             * @brief 记录警告信息
             * @param message 日志消息
             */
            void warning(const std::string& message);

            /**
             * @brief 记录错误信息
             * @param message 日志消息
             */
            void error(const std::string& message);

            /**
             * @brief 记录严重错误信息
             * @param message 日志消息
             */
            void critical(const std::string& message);

            /**
             * @brief 记录性能监控信息
             * @param operation 操作名称
             * @param duration 持续时间（毫秒）
             * @param details 详细信息
             */
            void logPerformance(const std::string& operation, double duration, const std::string& details = "");

            /**
             * @brief 刷新日志缓冲区
             */
            void flush();

            /**
             * @brief 关闭日志文件
             */
            void close();

        private:
            Logger() = default;
            ~Logger();

            Logger(const Logger&) = delete;
            Logger& operator=(const Logger&) = delete;

            /**
             * @brief 内部日志记录方法
             * @param level 日志级别
             * @param message 日志消息
             */
            void log(LogLevel level, const std::string& message);

            /**
             * @brief 获取日志级别字符串
             * @param level 日志级别
             * @return 级别字符串
             */
            std::string getLevelString(LogLevel level) const;

            /**
             * @brief 获取当前时间戳
             * @return 格式化的时间戳字符串
             */
            std::string getCurrentTimestamp() const;

        private:
            std::unique_ptr<std::ofstream> logFile_;    // 日志文件流
            LogLevel minLevel_ = LogLevel::INFO;        // 最小日志级别
            std::mutex logMutex_;                       // 线程同步互斥锁
            bool consoleOutput_ = true;                 // 是否输出到控制台
            std::string logFilePath_;                   // 日志文件路径
        };

/**
 * @brief 日志记录宏定义，提供便捷的日志记录接口
 */
#define LOG_DEBUG(msg) OceanSim::Utils::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) OceanSim::Utils::Logger::getInstance().info(msg)
#define LOG_WARNING(msg) OceanSim::Utils::Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) OceanSim::Utils::Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) OceanSim::Utils::Logger::getInstance().critical(msg)

/**
 * @brief 性能监控辅助类
 * 
 * 使用RAII模式自动记录代码块执行时间
 */
        class PerformanceTimer {
        public:
            explicit PerformanceTimer(const std::string& operation);
            ~PerformanceTimer();

            void addDetails(const std::string& details);

        private:
            std::string operation_;
            std::string details_;
            std::chrono::high_resolution_clock::time_point startTime_;
        };

/**
 * @brief 性能监控宏
 */
#define PERF_TIMER(op) OceanSim::Utils::PerformanceTimer __timer(op)

    } // namespace Utils
} // namespace OceanSim

#endif // LOGGER_H