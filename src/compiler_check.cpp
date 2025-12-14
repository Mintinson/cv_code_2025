#include <iostream>

#include <vector> 
#include <type_traits>

void print_stdlib_info() {
    std::cout << "Current Standard Library: ";

#if defined(_LIBCPP_VERSION)
    // LLVM libc++
    std::cout << "LLVM libc++ (Version: " << _LIBCPP_VERSION << ")" << std::endl;

#elif defined(__GLIBCXX__)
    // GNU libstdc++
    std::cout << "GNU libstdc++ (Version timestamp: " << __GLIBCXX__ << ")" << std::endl;

#elif defined(_MSVC_STL_VERSION)
    // Microsoft MSVC STL
    std::cout << "Microsoft STL (Version: " << _MSVC_STL_VERSION << ")" << std::endl;

#else
    std::cout << "Unknown Standard Library" << std::endl;
#endif
}

int main() {
    print_stdlib_info();
    return 0;
}