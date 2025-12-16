add_rules("mode.debug", "mode.release")

includes("toolchains/clang_libcxx.lua")

local opencv_configs = {webp = true}

if get_config("toolchain") == "clang-cl" then
    opencv_configs.webp = false
end


add_requires("opencv 4.11.0", {configs = opencv_configs})
add_requires("yaml-cpp 0.8.0")

add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})  -- for intellisense like clangd

set_languages("c++23")


add_defines("PROJECT_ROOT=R\"($(projectdir))\"")  
add_defines("BUILD_DIR=R\"($(builddir))\"")
add_defines("SOURCE_DIR=R\"($(projectdir)/src)\"")
add_defines("IMAGE_DIR=R\"($(projectdir)/images)\"")

if is_plat("windows") then
    add_cxflags("/utf-8", {toolchain = "cl"})
end

-- -- 1. 定义检测选项 (只负责检测和定义宏)
option("use_cuda")
    set_default(false) -- 默认关闭，除非用户显式开启
    set_showmenu(true)
    set_description("Enable CUDA support (requires CUDA Toolkit)")
    -- 移除了 on_check 和 add_defines，防止全局配置失败
option_end()

-- targets
target("test_lib")
    set_kind("binary")
    add_files("src/compiler_check.cpp")
    set_languages("c++23")

target("test_opencv")
    set_kind("binary")
    add_files("src/test_opencv.cpp")
    set_languages("c++23")
    add_packages("opencv")

target("final_project")
    set_kind("binary")
    add_files("src/final_project.cpp")
    set_languages("c++23")
    add_packages("opencv", "yaml-cpp")

    -- 挂载选项（为了让用户可以在 menuconfig 中看到该选项关联到了此 target）
    add_options("use_cuda")


    -- -----------------------------------------------------------
    -- 2. 在 Target 内部根据开关状态进行检测和配置
    -- -----------------------------------------------------------
    -- 只有当用户显式开启了 use_cuda (xmake f --use_cuda=y) 时，才进入此逻辑
    if has_config("use_cuda") then
        
-- 1. 尝试添加 cuda 包（xmake 会自动查找，找不到可能会报 warning 或 error）
        add_packages("cuda")
        
        -- 2. 直接添加宏和文件
        add_defines("USE_CUDA")
        add_files("src/cuda_operators.cu", {languages = "c++20"})
        
        -- 3. 编译选项
        add_cugencodes("native")
        add_cuflags("-allow-unsupported-compiler", {force = true})
        if is_plat("windows") then
            add_cuflags("--compiler-options=/utf-8")
        end
    end
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

