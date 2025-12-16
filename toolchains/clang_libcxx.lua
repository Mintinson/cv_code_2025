-- toolchains/clang_libcxx.lua

toolchain("clang+libc++")
    set_kind("standalone")
    -- 描述：这是一个基于 clang 的工具链
    set_description("Clang toolchain with libc++ standard library")
    
    -- 显式指定工具集，让 xmake 知道底层使用 clang
    set_toolset("cc", "clang")
    set_toolset("cxx", "clang++")
    set_toolset("ld", "clang++")        -- 链接器 (通常用 clang++ 驱动来做链接)
    set_toolset("sh", "clang++")        -- 共享库链接器
    set_toolset("ar", "llvm-ar")        -- 静态库归档器 (替换 GNU ar)
    set_toolset("as", "clang")        -- 汇编器
    set_toolset("ranlib", "llvm-ranlib")
    set_toolset("strip", "llvm-strip")
    
    -- 保持系统默认的 ar/ranlib，不要覆盖
    -- set_toolset("ar", "ar") -- 注释掉，使用系统默认

    -- 加载时注入 flags
    on_load(function (toolchain)
        local flags = "-stdlib=libc++"
        
        -- C++ 编译选项
        toolchain:add("cxxflags", flags)
        -- 链接选项
        toolchain:add("ldflags", flags)
        -- 动态库链接选项
        toolchain:add("shflags", flags)

        -- 【显式链接 ABI 库】
        -- 通常 -stdlib=libc++ 会自动处理，但为了保险（特别是源码编译环境），
        -- 显式链接 libc++abi 和 libunwind 是个好习惯
        toolchain:add("ldflags", "-lc++abi")
        toolchain:add("shflags", "-lc++abi")
        toolchain:add("ldflags", "-lunwind") -- 如果你编译了 libunwind
        
        -- 可选：如果你在 Linux 上需要使用 lld 链接器
        -- 注意：仅对最终链接有效，不影响静态库创建
        toolchain:add("ldflags", "-fuse-ld=lld")

        toolchain:add("envs", "CC", "clang")
        toolchain:add("envs", "CXX", "clang++")
        toolchain:add("envs", "LD", "clang++")
        toolchain:add("envs", "AS", "clang") -- FFmpeg 的汇编经常需要这个
        toolchain:add("envs", "AR", "llvm-ar")
        toolchain:add("envs", "RANLIB", "llvm-ranlib")
    end)
toolchain_end()