# CMakeLists.txt - C++核心模块构建配置
cmake_minimum_required(VERSION 3.16)
project(OceanCurrentSimulationCore VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# 查找依赖库
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# 包含目录
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tbb/include
)

# 源文件
file(GLOB_RECURSE CORE_SOURCES
        "src/core/*.cpp"
        "src/algorithms/*.cpp"
        "src/data/*.cpp"
        "src/utils/*.cpp"
)

# 绑定文件
file(GLOB CSHARP_BINDING_SOURCES "bindings/csharp/*.cpp")
file(GLOB PYTHON_BINDING_SOURCES "bindings/python/*.cpp")

# 核心静态库
add_library(OceanSimCore STATIC ${CORE_SOURCES}
        include/core/ParticleSimulator.h
        src/core/ParticleSimulator.cpp
        include/core/CurrentFieldSolver.h
        src/core/CurrentFieldSolver.cpp
        include/algorithms/RungeKuttaSolver.h
        include/data/GridDataStructure.h
        include/core/AdvectionDiffusionSolver.h
        include/core/PerformanceProfiler.h
        include/utils/MathUtils.h)
target_link_libraries(OceanSimCore
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        TBB::tbb
)

# C#绑定动态库
add_library(OceanSimCSharp SHARED ${CSHARP_BINDING_SOURCES}
        bindings/csharp/CppCoreWrapper.h
        bindings/csharp/CppCoreWrapper.cpp
        bindings/python/pybind_ocean_sim.cpp
        bindings/python/pybind_ocean_sim.h)
target_link_libraries(OceanSimCSharp OceanSimCore)

# Python绑定模块
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(OceanSimPython ${PYTHON_BINDING_SOURCES})
    target_link_libraries(OceanSimPython PRIVATE OceanSimCore)
endif()

# 安装规则
install(TARGETS OceanSimCore OceanSimCSharp
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)