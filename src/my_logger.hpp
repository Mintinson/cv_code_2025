/**
 * @file my_logger.hpp
 * @brief Thread-safe, colorized logger with source location tracking
 * @author wwf
 * @date 2025-12-14
 */

#pragma once

#include <chrono>
#include <filesystem>
#include <format>
#include <mutex>
#include <print>
#include <source_location>
#include <string_view>
#include <type_traits>

/**
 * @brief Log severity levels
 */
enum class LogLevel
{
    DEBUG,
    INFO,
    WARN,
    ERROR
};

/**
 * @brief ANSI color codes for terminal output
 */
namespace Color
{
constexpr std::string_view kReset  = "\033[0m";  ///< Reset to default color
constexpr std::string_view kRed    = "\033[31m"; ///< Red color (errors)
constexpr std::string_view kGreen  = "\033[32m"; ///< Green color (info)
constexpr std::string_view kYellow = "\033[33m"; ///< Yellow color (warnings)
constexpr std::string_view kBlue   = "\033[34m"; ///< Blue color
constexpr std::string_view kGray   = "\033[90m"; ///< Gray color (debug)
}

/**
 * @brief Log message wrapper with format string and source location
 * @tparam Args Format argument types
 */
/**
 * @brief Log message wrapper with format string and source location
 * @tparam Args Format argument types
 */
template <typename... Args>
struct LogMsg
{
    std::format_string<Args...> fmt; ///< Format string
    std::source_location loc;        ///< Source code location

    /**
     * @brief Construct from C-string literal
     * @param s Format string literal
     * @param l Source location (automatically captured)
     */
    consteval LogMsg(const char* s, std::source_location l = std::source_location::current())
        : fmt(s)
        , loc(l)
    {
    }

    /**
     * @brief Construct from string_view
     * @param s Format string view
     * @param l Source location (automatically captured)
     */
    consteval LogMsg(std::string_view s, std::source_location l = std::source_location::current())
        : fmt(s)
        , loc(l)
    {
    }
};

/**
 * @brief Concept for types that support lock/unlock operations
 * @tparam T Type to check
 */
template <typename T>
concept Lockable = requires(T lockable) {
    lockable.lock();
    lockable.unlock();
};

/**
 * @brief No-op mutex for single-threaded logging
 */
struct DummyMutex
{
    void lock() { }
    void unlock() { }
};
/**
 * @brief  Thread-safe, colorized logger with source location tracking
 *
 * @tparam Mutex a Lockable Type, default to No-op mutex for single-threaded logging
 */
template <Lockable Mutex = DummyMutex>
class Logger
{
public:
    // Singleton
    static Logger& get()
    {
        static Logger instance;
        return instance;
    }

    void setLevel(LogLevel level) { minLevel_ = level; }

    // General logging function
    template <typename... Args>
    void log(LogLevel level, LogMsg<Args...> msg, Args&&... args)
    {
        if (level < minLevel_) return;

        // 获取时间
        auto now = std::chrono::system_clock::now();

        // 提取文件名 (只保留文件名，去掉长路径)
        std::filesystem::path filePath(msg.loc.file_name());
        std::string fileName = filePath.filename().string();

        // 准备前缀数据
        std::string_view colorCode;
        std::string_view levelStr;

        switch (level) {
        case LogLevel::DEBUG:
            colorCode = Color::kGray;
            levelStr  = "DBG";
            break;
        case LogLevel::INFO:
            colorCode = Color::kGreen;
            levelStr  = "INF";
            break;
        case LogLevel::WARN:
            colorCode = Color::kYellow;
            levelStr  = "WRN";
            break;
        case LogLevel::ERROR:
            colorCode = Color::kRed;
            levelStr  = "ERR";
            break;
        }

        // 线程安全锁
        std::lock_guard<Mutex> lock(mutex_);

        // C++23 std::println 直接输出到 stderr (通常比 stdout 更适合日志)
        // 格式: [Time] [Level] [File:Line] UserMessage
        // 使用 {0:%T} 格式化 chrono 时间 (需编译器支持 C++20 chrono format，GCC13+/MSVC 17.6+ 支持)
        try {
            std::println(
                stderr,
                "{}[{:%T}] [{}] [{}:{}] {}{}",
                colorCode,
                //  std::chrono::floor<std::chrono::microseconds>(now), // 或者保留毫秒
                std::chrono::floor<std::chrono::microseconds>(now - lastLogTime_), // 或者保留毫秒
                levelStr,
                fileName,
                msg.loc.line(),
                std::format(msg.fmt, std::forward<Args>(args)...),
                Color::kReset);
        }
        catch (const std::exception& e) {
            // 极其罕见的情况：格式化失败
            std::println(stderr, "Log formatting error: {}", e.what());
        }
        lastLogTime_ = now;
    }

    /**
     * @brief Log a debug message
     * @tparam Args Format argument types
     * @param msg Log message with format string and source location
     * @param args Format arguments
     */
    template <typename... Args>
    void debug(std::type_identity_t<LogMsg<Args...>> msg, Args&&... args)
    {
        log(LogLevel::DEBUG, msg, std::forward<Args>(args)...);
    }

    /**
     * @brief Log an info message
     * @tparam Args Format argument types
     * @param msg Log message with format string and source location
     * @param args Format arguments
     */
    template <typename... Args>
    void info(std::type_identity_t<LogMsg<Args...>> msg, Args&&... args)
    {
        log(LogLevel::INFO, msg, std::forward<Args>(args)...);
    }

    /**
     * @brief Log a warning message
     * @tparam Args Format argument types
     * @param msg Log message with format string and source location
     * @param args Format arguments
     */
    template <typename... Args>
    void warn(std::type_identity_t<LogMsg<Args...>> msg, Args&&... args)
    {
        log(LogLevel::WARN, msg, std::forward<Args>(args)...);
    }

    /**
     * @brief Log an error message
     * @tparam Args Format argument types
     * @param msg Log message with format string and source location
     * @param args Format arguments
     */
    template <typename... Args>
    void error(std::type_identity_t<LogMsg<Args...>> msg, Args&&... args)
    {
        log(LogLevel::ERROR, msg, std::forward<Args>(args)...);
    }

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger()  = default;
    ~Logger() = default;

    Mutex mutex_;                        ///< Mutex for thread-safe logging
    LogLevel minLevel_ = LogLevel::INFO; ///< Minimum log level to display
    /// Last log timestamp for delta time calculation
    std::chrono::time_point<std::chrono::system_clock> lastLogTime_
        = std::chrono::system_clock::now();
};

/**
 * @brief Global logger instance (single-threaded)
 */
inline Logger<DummyMutex>& Log = Logger<DummyMutex>::get();