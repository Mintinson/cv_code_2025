#include <iostream>
#include <ranges>
#include <string_view>
#include <version>

/**
 * @brief print C++ standard version information
 */
void print_cpp_version()
{
    std::cout << "=== C++ Standard Version ===" << std::endl;
#if defined(_MSC_VER) && !defined(__clang__) // __cplusplus is not reliable in MSVC see
    // https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170
    std::cout << "__cplusplus(MSVC): " << _MSVC_LANG << std::endl;

    std::cout << "Detected as: ";
    // MSVC 特例处理
    if (_MSVC_LANG >= 202302L)
        std::cout << "C++23 or later";
    else if (_MSVC_LANG >= 202002L)
        std::cout << "C++20";
    else if (_MSVC_LANG >= 201703L)
        std::cout << "C++17";
    else if (_MSVC_LANG >= 201402L)
        std::cout << "C++14";
    else if (_MSVC_LANG >= 201103L)
        std::cout << "C++11";
    else
        std::cout << "Pre-C++11";
#else

#if __cplusplus >= 202302L
    std::cout << "__cplusplus: " << __cplusplus << std::endl;

    std::cout << "Detected as: ";
    std::cout << "C++23 or later";
#elif __cplusplus >= 202002L
    std::cout << "C++20";
#elif __cplusplus >= 201703L
    std::cout << "C++17";
#elif __cplusplus >= 201402L
    std::cout << "C++14";
#elif __cplusplus >= 201103L
    std::cout << "C++11";
#else
    std::cout << "Pre-C++11";
#endif
#endif
    std::cout << std::endl << std::endl;
}

/**
 * @brief print compiler information
 */
void print_compiler_info()
{
    std::cout << "=== Compiler Information ===" << std::endl;

#if defined(__clang__)
    // Clang/LLVM
    std::cout << "Compiler: Clang/LLVM" << std::endl;
    std::cout << "Version: " << __clang_major__ << "." << __clang_minor__ << "."
              << __clang_patchlevel__ << std::endl;

#elif defined(__GNUC__) || defined(__GNUG__)
    // GNU GCC/G++
    std::cout << "Compiler: GNU GCC/G++" << std::endl;
    std::cout << "Version: " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__
              << std::endl;

#elif defined(_MSC_VER)
    // Microsoft Visual C++
    std::cout << "Compiler: Microsoft Visual C++" << std::endl;
    std::cout << "Version: " << _MSC_VER << " (_MSC_FULL_VER: " << _MSC_FULL_VER << ")"
              << std::endl;

    // 解析 MSVC 版本
    std::cout << "MSVC Year: ";
#if _MSC_VER >= 1940
    std::cout << "2022 (17.10+)";
#elif _MSC_VER >= 1930
    std::cout << "2022 (17.0-17.9)";
#elif _MSC_VER >= 1920
    std::cout << "2019";
#elif _MSC_VER >= 1910
    std::cout << "2017";
#elif _MSC_VER >= 1900
    std::cout << "2015";
#else
    std::cout << "Older than 2015";
#endif
    std::cout << std::endl;

#elif defined(__INTEL_COMPILER) || defined(__ICC)
    // Intel C++
    std::cout << "Compiler: Intel C++" << std::endl;
    std::cout << "Version: " << __INTEL_COMPILER << std::endl;

#else
    std::cout << "Compiler: Unknown" << std::endl;
#endif

    std::cout << std::endl;
}

/**
 * @brief print standard library information
 */
void print_stdlib_info()
{
    std::cout << "=== Standard Library ===" << std::endl;
    std::cout << "Current Standard Library: ";

#if defined(_LIBCPP_VERSION)
    std::cout << "LLVM libc++" << std::endl;
    std::cout << "Version: " << _LIBCPP_VERSION << std::endl;

#elif defined(__GLIBCXX__)
    std::cout << "GNU libstdc++" << std::endl;
    std::cout << "Version timestamp: " << __GLIBCXX__ << std::endl;

    std::cout << "Approximate GCC version: ";
#if __GLIBCXX__ >= 20230714
    std::cout << "13.x or later";
#elif __GLIBCXX__ >= 20220421
    std::cout << "12.x";
#elif __GLIBCXX__ >= 20210427
    std::cout << "11.x";
#elif __GLIBCXX__ >= 20200326
    std::cout << "10.x";
#elif __GLIBCXX__ >= 20190503
    std::cout << "9.x";
#else
    std::cout << "8.x or earlier";
#endif
    std::cout << std::endl;

#elif defined(_MSVC_STL_VERSION)
    // Microsoft MSVC STL
    std::cout << "Microsoft STL" << std::endl;
    std::cout << "Version: " << _MSVC_STL_VERSION << std::endl;

#else
    std::cout << "Unknown Standard Library" << std::endl;
#endif

    std::cout << std::endl;
}

/**
 * @brief print platform information
 */
void print_platform_info()
{
    std::cout << "=== Platform Information ===" << std::endl;

    std::cout << "Operating System: ";
#if defined(_WIN32) || defined(_WIN64)
    std::cout << "Windows";
#if defined(_WIN64)
    std::cout << " (64-bit)";
#else
    std::cout << " (32-bit)";
#endif
#elif defined(__linux__)
    std::cout << "Linux";
#elif defined(__APPLE__) && defined(__MACH__)
    std::cout << "macOS";
#elif defined(__unix__)
    std::cout << "Unix";
#else
    std::cout << "Unknown";
#endif
    std::cout << std::endl;

    std::cout << "Architecture: ";
#if defined(__x86_64__) || defined(_M_X64)
    std::cout << "x86_64 (64-bit)";
#elif defined(__i386__) || defined(_M_IX86)
    std::cout << "x86 (32-bit)";
#elif defined(__aarch64__) || defined(_M_ARM64)
    std::cout << "ARM64";
#elif defined(__arm__) || defined(_M_ARM)
    std::cout << "ARM";
#else
    std::cout << "Unknown";
#endif
    std::cout << std::endl;

    std::cout << std::endl;
}

/**
 * @brief check for C++ feature support macros and print results
 */
void print_feature_support()
{
    std::cout << "=== C++ Feature Support ===" << std::endl;

#ifdef __cpp_concepts
    std::cout << "Concepts (C++20): " << __cpp_concepts << std::endl;
#else
    std::cout << "Concepts (C++20): Not supported" << std::endl;
#endif

#ifdef __cpp_modules
    std::cout << "Modules (C++20): " << __cpp_modules << std::endl;
#else
    std::cout << "Modules (C++20): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_format
    std::cout << "std::format (C++20): " << __cpp_lib_format << std::endl;
#else
    std::cout << "std::format (C++20): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_ranges
    std::cout << "Ranges (C++20): " << __cpp_lib_ranges << std::endl;
#else
    std::cout << "Ranges (C++20): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_coroutine
    std::cout << "Coroutines (C++20): " << __cpp_lib_coroutine << std::endl;
#else
    std::cout << "Coroutines (C++20): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_print
    std::cout << "std::print (C++23): " << __cpp_lib_print << std::endl;
#else
    std::cout << "std::print (C++23): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_mdspan
    std::cout << "std::mdspan (C++23): " << __cpp_lib_mdspan << std::endl;
#else
    std::cout << "std::mdspan (C++23): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_ranges_zip
    std::cout << "std::ranges::zip (C++23): " << __cpp_lib_ranges_zip << std::endl;
#else
    std::cout << "std::ranges::zip (C++23): Not supported" << std::endl;
#endif

#ifdef __cpp_lib_ranges_to_container
    std::cout << "std::ranges::to (C++23): " << __cpp_lib_ranges_to_container << std::endl;
#else
    std::cout << "std::ranges::to (C++23): Not supported" << std::endl;
#endif

    std::cout << std::endl;
}

int main()
{
    std::cout << "============ C++ Compiler & Standard Library Checker ============" << std::endl;
    std::cout << std::endl;

    print_cpp_version();
    print_compiler_info();
    print_stdlib_info();
    print_platform_info();
    print_feature_support();

    std::cout << "================================================================" << std::endl;
    std::cout << "Check completed successfully!" << std::endl;

    return 0;
}