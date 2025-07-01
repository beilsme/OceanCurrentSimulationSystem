# CMakeLists.txt - 海洋洋流模拟系统核心模块构建配置
cmake_minimum_required(VERSION 3.16)
project(OceanCurrentSimulationSystem VERSION 1.0.0 LANGUAGES CXX)

# 在所有目标上启用Position Independent Code，
# 以便静态库可被链接到共享库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# 构建选项
option(BUILD_TESTS "构建测试程序" ON)
option(BUILD_PYTHON_BINDINGS "构建Python绑定" ON)
option(BUILD_CSHARP_BINDINGS "构建C#绑定" ON)
option(ENABLE_PROFILING "启用性能分析" ON)
option(ENABLE_GPU_SUPPORT "启用GPU加速支持" OFF)

# 在所有目标上启用Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# 编译器特定设置
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -mavx2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# 平台特定设置（一定放在 find_package(OpenMP REQUIRED) 之前）
if(APPLE)
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
    include_directories(/opt/homebrew/opt/libomp/include)
    link_directories(/opt/homebrew/opt/libomp/lib)
endif()

find_package(OpenMP REQUIRED)


# 查找必需依赖库
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# 查找可选依赖库
find_package(TBB QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NETCDF QUIET netcdf)
    pkg_check_modules(HDF5 QUIET hdf5)
endif()

# 检查TBB可用性
if(TBB_FOUND)
    message(STATUS "找到Intel TBB，启用并行优化")
    add_definitions(-DUSE_TBB)
else()
    message(WARNING "未找到Intel TBB，将使用标准库线程")
endif()

# 检查NetCDF可用性
if(NETCDF_FOUND)
    message(STATUS "找到NetCDF库，启用NetCDF数据支持")
    add_definitions(-DUSE_NETCDF)
else()
    message(WARNING "未找到NetCDF库，NetCDF功能将被禁用")
endif()

# 检查HDF5可用性
if(HDF5_FOUND)
    message(STATUS "找到HDF5库，启用HDF5数据支持")
    add_definitions(-DUSE_HDF5)
else()
    message(WARNING "未找到HDF5库，HDF5功能将被禁用")
endif()

# GPU支持检查
if(ENABLE_GPU_SUPPORT)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        message(STATUS "找到CUDA，启用GPU加速支持")
        add_definitions(-DUSE_CUDA)
    else()
        message(WARNING "未找到CUDA，GPU功能将被禁用")
        set(ENABLE_GPU_SUPPORT OFF)
    endif()
endif()

# 包含目录
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen
)
# ======== 手动指定 Python 头文件 & 动态库 ========
#set(PY_INC "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/python/include")
#set(PY_LIB "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/python/lib/libpython3.12.dylib")
#
#include_directories(${PY_INC})                     # 让所有目标都能找到 <Python.h>
#link_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/python/lib")   # 供链接器搜索

# 允许外部自定义 Python 路径；若未指定则使用系统默认
set(PY_INC "")
set(PY_LIB "")
# ===============================================



# 平台特定设置
if(APPLE)
    # macOS上的OpenMP和依赖库路径
    if(EXISTS "/opt/homebrew/opt/libomp")
        include_directories(/opt/homebrew/opt/libomp/include)
        link_directories(/opt/homebrew/opt/libomp/lib)
    endif()
    if(EXISTS "/opt/homebrew/include")
        include_directories(/opt/homebrew/include)
        link_directories(/opt/homebrew/lib)
    endif()
elseif(WIN32)
    # Windows特定设置
    add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()




# 明确指定核心源文件
set(CORE_SOURCES
        # 核心模块
        src/core/AdvectionDiffusionSolver.cpp
        src/core/PerformanceProfiler.cpp
        src/core/ParticleSimulator.cpp
        src/core/CurrentFieldSolver.cpp

        # 算法模块
        src/algorithms/RungeKuttaSolver.cpp
        src/algorithms/FiniteDifferenceSolver.cpp
        src/algorithms/ParallelComputeEngine.cpp
        src/algorithms/VectorizedOperations.cpp

        # 数据结构
        src/data/GridDataStructure.cpp
        src/data/MemoryManager.cpp
        src/data/DataExporter.cpp

        # 工具函数
        src/utils/MathUtils.cpp
        src/utils/ThreadPool.cpp
        src/utils/Logger.cpp

        include/algorithms/FiniteDifferenceSolver.h
        include/algorithms/RungeKuttaSolver.h
        include/algorithms/ParallelComputeEngine.h
        include/algorithms/VectorizedOperations.h
        include/core/AdvectionDiffusionSolver.h
        include/core/CurrentFieldSolver.h
        include/core/ParticleSimulator.h
        include/core/PerformanceProfiler.h
        include/data/GridDataStructure.h
        include/data/MemoryManager.h
        include/data/DataExporter.h
        include/utils/MathUtils.h
        include/utils/ThreadPool.h
        include/utils/Logger.h

)

# GPU源文件（如果启用）
if(ENABLE_GPU_SUPPORT)
    set(GPU_SOURCES
            src/gpu/CudaComputeKernel.cu
            src/gpu/GpuMemoryManager.cu
    )
endif()

# 构建核心静态库
add_library(CppCore STATIC ${CORE_SOURCES})

# 链接核心库依赖
target_link_libraries(CppCore
        PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        Threads::Threads
)

# 链接可选依赖
if(TBB_FOUND)
    target_link_libraries(CppCore PUBLIC TBB::tbb)
endif()

if(NETCDF_FOUND)
    target_link_libraries(CppCore PUBLIC ${NETCDF_LIBRARIES})
    target_include_directories(CppCore PUBLIC ${NETCDF_INCLUDE_DIRS})
endif()

if(HDF5_FOUND)
    target_link_libraries(CppCore PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(CppCore PUBLIC ${HDF5_INCLUDE_DIRS})
    link_directories(${HDF5_LIBRARY_DIRS})
endif()

# GPU支持
if(ENABLE_GPU_SUPPORT)
    add_library(OceanSimGPU STATIC ${GPU_SOURCES})
    target_link_libraries(OceanSimGPU PUBLIC CUDA::cudart CUDA::cufft)
    target_link_libraries(CppCore PUBLIC OceanSimGPU)
endif()

# 数学库链接（Linux需要）
if(UNIX AND NOT APPLE)
    target_link_libraries(CppCore PUBLIC m)
endif()

# 设置目标属性
set_target_properties(CppCore PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        OUTPUT_NAME "oceansim_core"
)



# C#绑定动态库
if(BUILD_CSHARP_BINDINGS)
    set(CSHARP_BINDING_SOURCES
            bindings/csharp/CppCoreWrapper.cpp
            bindings/csharp/CppCoreInterop.cpp
            bindings/csharp/ParticleSimulatorWrapper.cpp
            bindings/csharp/GridDataWrapper.cpp
    )

    add_library(OceanSimCSharp SHARED ${CSHARP_BINDING_SOURCES})
    target_link_libraries(OceanSimCSharp PRIVATE CppCore)

    set_target_properties(OceanSimCSharp PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION 1
            OUTPUT_NAME "oceansim_csharp"
    )

    # C#绑定的特殊导出设置
    if(WIN32)
        set_target_properties(OceanSimCSharp PROPERTIES
                WINDOWS_EXPORT_ALL_SYMBOLS ON
        )
    endif()
endif()

# Python绑定模块
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        set(PYTHON_BINDING_SOURCES
                bindings/python/pybind_main.cpp
                bindings/python/pybind_grid.cpp
                bindings/python/pybind_algorithms.cpp
                bindings/python/pybind_parallel.cpp
                bindings/python/pybind_advection.cpp
                bindings/python/pybind_current_solver.cpp
                bindings/python/pybind_particle_sim.cpp
        )


        pybind11_add_module(oceansim ${PYTHON_BINDING_SOURCES})
        if(PY_INC)
            target_include_directories(oceansim PRIVATE ${PY_INC})
        endif()
        target_link_libraries(oceansim PRIVATE CppCore)
        if(PY_LIB)
            target_link_libraries(oceansim PRIVATE ${PY_LIB})
        endif()


    else()
        message(WARNING "未找到pybind11，Python绑定将被禁用")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# 测试程序
if(BUILD_TESTS)
    enable_testing()

    # 查找测试框架
    find_package(GTest QUIET)
    if(GTest_FOUND)
        set(TEST_SOURCES
                tests/unit_tests/test_math_utils.cpp
                tests/unit_tests/test_grid_structure.cpp
                tests/unit_tests/test_advection_solver.cpp
                tests/integration_tests/test_full_simulation.cpp
                tests/performance_tests/test_performance.cpp
        )

        add_executable(OceanSimTests ${TEST_SOURCES})
        target_link_libraries(OceanSimTests
                PRIVATE
                CppCore
                GTest::gtest_main
                GTest::gmock_main
        )

        # 注册测试
        add_test(NAME UnitTests COMMAND OceanSimTests)

        message(STATUS "测试程序已启用")
    else()
        message(WARNING "未找到Google Test，测试程序将被禁用")
    endif()
endif()

# 性能分析支持
if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    message(STATUS "性能分析已启用")
endif()

# 安装规则
install(TARGETS CppCore
        EXPORT CppCoreTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

if(BUILD_CSHARP_BINDINGS)
    install(TARGETS OceanSimCSharp
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
    )
endif()

if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    install(TARGETS oceansim
            LIBRARY DESTINATION lib/python
    )
endif()

# 安装头文件
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
)

# 安装CMake配置文件
install(EXPORT CppCoreTargets
        FILE CppCoreTargets.cmake
        NAMESPACE OceanSim::
        DESTINATION lib/cmake/CppCore
)

# 生成配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        CppCoreConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppCoreConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CppCoreConfig.cmake
        INSTALL_DESTINATION lib/cmake/CppCore
)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CppCoreConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/CppCoreConfigVersion.cmake
        DESTINATION lib/cmake/CppCore
)

# 构建信息输出
message(STATUS "=== 海洋洋流模拟系统构建配置 ===")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "编译器: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "OpenMP支持: ${OpenMP_FOUND}")
message(STATUS "Eigen版本: ${Eigen3_VERSION}")
message(STATUS "TBB支持: ${TBB_FOUND}")
message(STATUS "NetCDF支持: ${NETCDF_FOUND}")
message(STATUS "HDF5支持: ${HDF5_FOUND}")
message(STATUS "GPU支持: ${ENABLE_GPU_SUPPORT}")
message(STATUS "Python绑定: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "C#绑定: ${BUILD_CSHARP_BINDINGS}")
message(STATUS "测试程序: ${BUILD_TESTS}")
message(STATUS "性能分析: ${ENABLE_PROFILING}")
message(STATUS "=====================================")

# 生成编译数据库（用于IDE支持）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)