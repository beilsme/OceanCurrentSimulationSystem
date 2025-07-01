/**
 * @file pybind_algorithms.cpp
 * @author beilsm
 * @version 1.0
 * @brief Numerical algorithm bindings
 * @date 2025-07-01
 */
#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "algorithms/RungeKuttaSolver.h"
#include "algorithms/FiniteDifferenceSolver.h"

namespace py = pybind11;

void bind_algorithms(py::module_ &m)
{
    using RK = OceanSim::Algorithms::RungeKuttaSolver;
    py::enum_<RK::Method>(m, "RKMethod")
            .value("RK4", RK::Method::RK4)
            .value("RK45", RK::Method::RK45)
            .value("DOPRI5", RK::Method::DOPRI5)
            .value("RK8", RK::Method::RK8)
            .value("GAUSS_LEGENDRE", RK::Method::GAUSS_LEGENDRE);

    py::class_<RK::ButcherTableau>(m, "ButcherTableau")
            .def(py::init<>())
            .def_readwrite("A", &RK::ButcherTableau::A)
            .def_readwrite("b", &RK::ButcherTableau::b)
            .def_readwrite("c", &RK::ButcherTableau::c)
            .def_readwrite("b_err", &RK::ButcherTableau::b_err)
            .def_readwrite("order", &RK::ButcherTableau::order)
            .def_readwrite("is_implicit", &RK::ButcherTableau::is_implicit);

    py::class_<RK, std::shared_ptr<RK>>(m, "RungeKuttaSolver")
            .def(py::init<RK::Method>(), py::arg("method") = RK::Method::RK4)
            .def(py::init<const RK::ButcherTableau &>())
                    // 明确指定方法签名来解决重载问题
            .def("solve",
                 static_cast<Eigen::Vector3d(RK::*)(const Eigen::Vector3d&, const Eigen::Vector3d&, double, const RK::VectorFunction&) const>(&RK::solve),
                 "Solve vector ODE", py::arg("y0"), py::arg("dy0"), py::arg("dt"), py::arg("f"))
            .def("solve_scalar",
                 static_cast<double(RK::*)(double, double, double, const RK::ScalarFunction&) const>(&RK::solve),
                 "Solve scalar ODE", py::arg("y0"), py::arg("t0"), py::arg("dt"), py::arg("f"))
            .def("solve_adaptive", &RK::solveAdaptive,
                 "Solve with adaptive time step",
                 py::arg("y0"), py::arg("dy0"), py::arg("dt"), py::arg("f"),
                 py::arg("tolerance")=1e-6, py::arg("dt_min")=1e-10, py::arg("dt_max")=1.0)
            .def("solve_sequence", &RK::solveSequence,
                 "Solve time sequence",
                 py::arg("y0"), py::arg("t0"), py::arg("t_end"), py::arg("dt"), py::arg("f"))
            .def("set_method", &RK::setMethod, "Set RK method")
            .def("set_custom_tableau", &RK::setCustomTableau, "Set custom Butcher tableau")
            .def("set_tolerance", &RK::setTolerance, "Set tolerance")
            .def("set_max_iterations", &RK::setMaxIterations, "Set max iterations")
            .def("get_order", &RK::getOrder, "Get method order")
            .def("get_stages", &RK::getStages, "Get number of stages")
            .def("is_implicit", &RK::isImplicit, "Check if method is implicit")
                    // 静态方法
            .def_static("create_rk4", &RK::createRK4, "Create RK4 tableau")
            .def_static("create_rk45", &RK::createRK45, "Create RK45 tableau")
            .def_static("create_dopri5", &RK::createDOPRI5, "Create DOPRI5 tableau")
            .def_static("create_rk8", &RK::createRK8, "Create RK8 tableau")
            .def_static("create_gauss_legendre", &RK::createGaussLegendre,
                        "Create Gauss-Legendre tableau", py::arg("stages"));

    // 向量化求解器
    using VRK = OceanSim::Algorithms::VectorizedRKSolver;
    py::class_<VRK, std::shared_ptr<VRK>>(m, "VectorizedRKSolver")
            .def(py::init<RK::Method>(), py::arg("method") = RK::Method::RK4)
            .def("solve_batch", &VRK::solveBatch,
                 "Solve batch of initial value problems",
                 py::arg("y0_batch"), py::arg("dt"), py::arg("f"))
            .def("solve_batch_simd", &VRK::solveBatchSIMD,
                 "Solve batch with SIMD optimization",
                 py::arg("y_batch"), py::arg("dt"), py::arg("f"));

    // 有限差分求解器
    using FD = OceanSim::Algorithms::FiniteDifferenceSolver;
    py::class_<FD, std::shared_ptr<FD>>(m, "FiniteDifferenceSolver")
            .def(py::init<int, double>(),
                 "Constructor", py::arg("grid_size"), py::arg("time_step"))
            .def("solve_advection_diffusion", &FD::solveAdvectionDiffusion,
                 "Solve advection-diffusion equation",
                 py::arg("u"), py::arg("concentration"), py::arg("diffusivity"))
            .def("compute_spatial_derivative", &FD::computeSpatialDerivative,
                 "Compute spatial derivative",
                 py::arg("field"), py::arg("derivative"), py::arg("order")=1)
            .def("set_boundary_conditions", &FD::setBoundaryConditions,
                 "Set boundary conditions",
                 py::arg("boundary_type"), py::arg("boundary_values"))
            .def("get_grid_spacing", &FD::getGridSpacing, "Get grid spacing")
            .def("get_time_step", &FD::getTimeStep, "Get time step");
}
