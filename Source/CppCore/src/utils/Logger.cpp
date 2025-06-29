#include "utils/Logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace OceanSim {
    namespace Utils {

        Logger& Logger::getInstance() {
            static Logger instance;
            return instance;
        }

        Logger::~Logger() {
            close();
        }

        void Logger::setLogFile(const std::string& filepath) {
            std::lock_guard<std::mutex> lock(logMutex_);

            if (logFile_) {
                logFile_->close();
            }

            logFilePath_ = filepath;
            logFile_ = std::make_unique<std::ofstream>(filepath, std::ios::app);

            if (!logFile_->is_open()) {
                std::cerr << "Warning: Could not open log file: " << filepath << std::endl;
                logFile_.reset();
            } else {
                log(LogLevel::INFO, "Log file opened: " + filepath);
            }
        }

        void Logger::setMinLogLevel(LogLevel level) {
            std::lock_guard<std::mutex> lock(logMutex_);
            minLevel_ = level;
        }

        void Logger::debug(const std::string& message) {
            log(LogLevel::DEBUG, message);
        }

        void Logger::info(const std::string& message) {
            log(LogLevel::INFO, message);
        }

        void Logger::warning(const std::string& message) {
            log(LogLevel::WARNING, message);
        }

        void Logger::error(const std::string& message) {
            log(LogLevel::ERROR, message);
        }

        void Logger::critical(const std::string& message) {
            log(LogLevel::CRITICAL, message);
        }

        void Logger::logPerformance(const std::string& operation, double duration, const std::string& details) {
            std::ostringstream oss;
            oss << "PERF [" << operation << "] " << std::fixed << std::setprecision(3)
                << duration << "ms";

            if (!details.empty()) {
                oss << " - " << details;
            }

            log(LogLevel::INFO, oss.str());
        }

        void Logger::flush() {
            std::lock_guard<std::mutex> lock(logMutex_);

            if (logFile_) {
                logFile_->flush();
            }

            if (consoleOutput_) {
                std::cout.flush();
            }
        }

        void Logger::close() {
            std::lock_guard<std::mutex> lock(logMutex_);

            if (logFile_) {
                log(LogLevel::INFO, "Log file closing");
                logFile_->close();
                logFile_.reset();
            }
        }

        void Logger::log(LogLevel level, const std::string& message) {
            std::lock_guard<std::mutex> lock(logMutex_);

            if (level < minLevel_) {
                return;
            }

            std::string timestamp = getCurrentTimestamp();
            std::string levelStr = getLevelString(level);
            std::string logEntry = timestamp + " [" + levelStr + "] " + message;

            // 输出到控制台
            if (consoleOutput_) {
                std::cout << logEntry << std::endl;
            }

            // 输出到文件
            if (logFile_ && logFile_->is_open()) {
                *logFile_ << logEntry << std::endl;
                logFile_->flush();
            }
        }

        std::string Logger::getLevelString(LogLevel level) const {
            switch (level) {
                case LogLevel::DEBUG:    return "DEBUG";
                case LogLevel::INFO:     return "INFO ";
                case LogLevel::WARNING:  return "WARN ";
                case LogLevel::ERROR:    return "ERROR";
                case LogLevel::CRITICAL: return "CRIT ";
                default:                 return "UNKN ";
            }
        }

        std::string Logger::getCurrentTimestamp() const {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;

            std::ostringstream oss;
            oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

            return oss.str();
        }

// PerformanceTimer implementation
        PerformanceTimer::PerformanceTimer(const std::string& operation)
                : operation_(operation), startTime_(std::chrono::high_resolution_clock::now()) {
        }

        PerformanceTimer::~PerformanceTimer() {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    endTime - startTime_).count();

            Logger::getInstance().logPerformance(operation_, static_cast<double>(duration), details_);
        }

        void PerformanceTimer::addDetails(const std::string& details) {
            details_ = details;
        }

    } // namespace Utils
} // namespace OceanSim