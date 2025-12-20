/**
 * @file my_cmd_parser.hpp
 * @brief Type-safe command-line argument parser with automatic type conversion
 * @author wwf
 * @date 2025-12-14
 */

#pragma once

#include <functional>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <charconv>

#include "my_logger.hpp"

/**
 * @brief Variant type for storing argument values of different types
 */
using ArgVariant = std::variant<std::monostate, bool, int, float, double, std::string>;

/**
 * @brief Simple but powerful command-line argument parser
 *
 * Features:
 * - Type-safe argument parsing with compile-time type checking
 * - Automatic type conversion using std::from_chars
 * - Support for bool, int, float, double, and string types
 * - Automatic help generation
 * - Default values
 * - User-provided argument tracking
 */
class MyCmdParser
{
public:
    /**
     * @brief Construct a command-line parser
     * @param programName Name of the program (shown in help)
     * @param description Program description (shown in help)
     */
    MyCmdParser(std::string programName, std::string description)
        : programName_(std::move(programName))
        , description_(std::move(description))
    {
    }

    /**
     * @brief Check if an argument was provided by the user
     * @param name Argument name (without leading dashes)
     * @return true if the argument was provided on the command line
     */
    [[nodiscard]] bool isUserProvided(std::string_view name) const
    {
        std::string key = normalizeKey(name);
        auto it         = arguments_.find(key);
        if (it != arguments_.end()) { return it->second.isUserProvided; }
        return false;
    }

    /**
     * @brief Add a typed argument with optional default value
     * @tparam T Argument type (bool, int, float, double, or std::string)
     * @param name Argument name (without leading dashes)
     * @param help Help text describing the argument
     * @param defaultValue Optional default value
     */
    template <typename T>
        requires std::is_same_v<T, bool> || std::is_arithmetic_v<T>
                 || std::is_same_v<T, std::string>
    void addArgument(std::string_view name, // NOLINT
                     std::string_view help,
                     std::optional<T> defaultValue = std::nullopt)
    {
        std::string key = normalizeKey(name);

        if (arguments_.contains(key)) {
            std::println(stderr, "Warning: Argument '{}' is already defined. Overwriting.", key);
        }

        Argument arg;
        arg.help   = help;
        arg.isFlag = std::is_same_v<T, bool>;

        // 设置默认值
        if (defaultValue.has_value()) { arg.value = *defaultValue; }
        else {
            // 如果是 bool，默认是 false
            if constexpr (std::is_same_v<T, bool>) arg.value = false;
        }

        // 核心优化：存储一个解析器 Lambda，在 parse 时立即转换类型
        arg.parser = [](const std::string_view& s, ArgVariant& v) -> bool {
            if constexpr (std::is_same_v<T, std::string>) {
                v = std::string(s);
                return true;
            }
            else if constexpr (std::is_same_v<T, bool>) {
                // 处理 bool (flag 不需要值，但在某些情况下可能传入 true/false)
                v = true;
                return true;
            }
            else {
                T temp;
                auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), temp); // NOLINT
                if (ec == std::errc()) {
                    v = temp;
                    return true;
                }
                return false;
            }
        };

        arguments_[key] = std::move(arg);
    }

    /**
     * @brief Add a boolean flag argument (defaults to false)
     * @param name Flag name (without leading dashes)
     * @param help Help text describing the flag
     */
    void addFlag(std::string_view name, std::string_view help)
    {
        addArgument<bool>(name, help, false);
    }

    /**
     * @brief Get the value of an argument
     * @tparam T Expected type of the argument value
     * @param name Argument name (without leading dashes)
     * @return Optional containing the value if found and of correct type, nullopt otherwise
     */
    template <typename T>
    std::optional<T> getValue(std::string_view name) const
    {
        std::string key = normalizeKey(name);
        auto it         = arguments_.find(key);

        if (it == arguments_.end()) { return std::nullopt; }

        // 直接检查 Variant 是否持有请求的类型
        if (auto* val = std::get_if<T>(&it->second.value)) { return *val; }

        // 如果 Variant 是 monostate (没被赋值且没默认值)，返回 nullopt
        if (std::holds_alternative<std::monostate>(it->second.value)) { return std::nullopt; }
        return std::nullopt;
    }

    /**
     * @brief Parse command-line arguments
     * @param argc Argument count from main()
     * @param argv Argument vector from main()
     * @return true if parsing succeeded, false if help was requested or an error occurred
     */
    bool parse(int argc, char** argv)
    {
        // 跳过程序名
        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i]; // NOLINT

            if (arg == "--") continue; // 处理分隔符

            if (arg == "-h" || arg == "--help") {
                printHelp();
                return false; // 或是 exit(0)
            }

            // 处理 Key (去掉 -- 或 -)
            std::string key = normalizeKey(arg);
            auto it         = arguments_.find(key);

            if (it == arguments_.end()) {
                Log.error("Error: Unknown argument '{}'", arg);
                printHelp();
                return false;
            }

            Argument& argObj = it->second;

            argObj.isUserProvided = true;

            if (argObj.isFlag) {
                // 如果是 Flag，直接设为 true (复用了 parser 逻辑，或者直接赋值)
                argObj.value = true;
            }
            else {
                // 检查是否还有下一个参数
                if (i + 1 >= argc) {
                    Log.error("Error: Argument '{}' requires a value", arg);
                    return false;
                }

                // 获取下一个参数作为值
                std::string_view valStr = argv[++i]; // NOLINT

                // 调用之前存储的 parser lambda 进行类型转换
                if (!argObj.parser(valStr, argObj.value)) {
                    Log.error("Error: Invalid value '{}' for argument '{}'", valStr, arg);
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Print help message to stdout
     */
    void printHelp() const
    {
        std::println("\n{}\n\nUsage: {} [options]\n\nOptions:", description_, programName_);

        for (const auto& [name, arg] : arguments_) {
            std::string flags = "--" + name;

            // 格式化默认值
            std::string defaultStr;
            std::visit(
                [&](auto&& val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (!std::is_same_v<T, std::monostate>) {
                        if constexpr (std::is_same_v<T, bool>) {
                            // Flag 通常不显示默认值 false，除非你需要
                        }
                        else {
                            defaultStr = std::format(" (default: {})", val);
                        }
                    }
                },
                arg.value);

            std::println("  {:<20} {}{}", flags, arg.help, defaultStr);
        }
        std::println("  {:<20} {}", "-h, --help", "Show this help message");
    }

private:
    /**
     * @brief Internal argument representation
     */
    struct Argument
    {
        std::string help;    ///< Help text
        bool isFlag = false; ///< True if this is a boolean flag
        ArgVariant value;    ///< Current value
        /// Type-erased parser function for converting string to typed value
        std::function<bool(const std::string_view&, ArgVariant&)> parser;
        bool isUserProvided = false; ///< True if user provided this argument
    };

    /**
     * @brief Normalize argument key by removing leading dashes
     * @param str Raw argument string (may include -- or -)
     * @return Normalized key without leading dashes
     */
    static std::string normalizeKey(std::string_view str)
    {
        size_t start = str.find_first_not_of('-');
        if (start == std::string_view::npos) return std::string(str);
        return std::string(str.substr(start));
    }

    std::string programName_;                                ///< Program name
    std::string description_;                                ///< Program description
    std::map<std::string, Argument, std::less<>> arguments_; ///< Registered arguments
};