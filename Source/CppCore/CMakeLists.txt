# CMakeLists.txt - C++核心模块构建配置
cmake_minimum_required(VERSION 3.16)
project(OceanCurrentSimulationCore VERSION 1.0.0 LANGUAGES CXX)

# 在所有目标上启用Position Independent Code，
# 以便静态库可被链接到共享库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
add_library(OceanSimCore STATIC ${CORE_SOURCES})
target_link_libraries(OceanSimCore
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        TBB::tbb
)

if(APPLE)
    target_include_directories(OceanSimCore PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_directories(OceanSimCore PRIVATE /opt/homebrew/opt/libomp/lib)
endif()

# C#绑定动态库
# C#绑定引擎不需要包含Python的绑定文件
add_library(OceanSimCSharp SHARED ${CSHARP_BINDING_SOURCES}
        bindings/csharp/CppCoreWrapper.cpp)
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