/**
 * @file pybind_parallel.cpp
 * @author beilsm
 * @version 1.0
 * @brief Parallel compute and vector operations bindings
 * @date 2025-07-01
 */
#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "algorithms/VectorizedOperations.h"
#include "algorithms/ParallelComputeEngine.h"

namespace py = pybind11;

void bind_parallel(py::module_ &m)
{
    using VecOp = OceanSimulation::Core::VectorizedOperations;
    py::enum_<VecOp::SimdType>(m, "SimdType")
            .value("None", VecOp::SimdType::None)
            .value("SSE", VecOp::SimdType::SSE)
            .value("AVX", VecOp::SimdType::AVX)
            .value("AVX2", VecOp::SimdType::AVX2)
            .value("AVX512", VecOp::SimdType::AVX512)
            .value("NEON", VecOp::SimdType::NEON);
    py::class_<VecOp::Config>(m, "VectorConfig")
            .def(py::init<>())
            .def_readwrite("preferredSimd", &VecOp::Config::preferredSimd)
            .def_readwrite("enableAutoAlignment", &VecOp::Config::enableAutoAlignment)
            .def_readwrite("alignmentBytes", &VecOp::Config::alignmentBytes)
            .def_readwrite("enableBoundsCheck", &VecOp::Config::enableBoundsCheck)
            .def_readwrite("enablePrefetch", &VecOp::Config::enablePrefetch);

    py::class_<VecOp, std::shared_ptr<VecOp>>(m, "VectorizedOperations")
            .def(py::init<const VecOp::Config &>())
            .def("vector_add", (void(VecOp::*)(const float*, const float*, float*, size_t))&VecOp::vectorAdd)
            .def("vector_sub", (void(VecOp::*)(const float*, const float*, float*, size_t))&VecOp::vectorSub)
            .def("vector_mul", (void(VecOp::*)(const float*, const float*, float*, size_t))&VecOp::vectorMul);

    using Engine = OceanSimulation::Core::ParallelComputeEngine;
    py::enum_<Engine::ExecutionPolicy>(m, "ExecutionPolicy")
            .value("Sequential", Engine::ExecutionPolicy::Sequential)
            .value("Parallel", Engine::ExecutionPolicy::Parallel)
            .value("Vectorized", Engine::ExecutionPolicy::Vectorized)
            .value("HybridParallel", Engine::ExecutionPolicy::HybridParallel);
    py::enum_<Engine::Priority>(m, "Priority")
            .value("Low", Engine::Priority::Low)
            .value("Normal", Engine::Priority::Normal)
            .value("High", Engine::Priority::High)
            .value("Critical", Engine::Priority::Critical);
    py::class_<Engine::Config>(m, "EngineConfig")
            .def(py::init<>())
            .def_readwrite("maxThreads", &Engine::Config::maxThreads)
            .def_readwrite("workStealingQueueSize", &Engine::Config::workStealingQueueSize)
            .def_readwrite("enableAffinity", &Engine::Config::enableAffinity)
            .def_readwrite("enableHyperthreading", &Engine::Config::enableHyperthreading)
            .def_readwrite("defaultPolicy", &Engine::Config::defaultPolicy)
            .def_readwrite("chunkSize", &Engine::Config::chunkSize)
            .def_readwrite("loadBalanceThreshold", &Engine::Config::loadBalanceThreshold);

    py::class_<Engine, std::shared_ptr<Engine>>(m, "ParallelComputeEngine")
            .def(py::init<const Engine::Config &>())
            .def("start", &Engine::start)
            .def("stop", &Engine::stop)
            .def("get_available_threads", &Engine::getAvailableThreads);
}