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
            .def("solve", &RK::solve)
            .def("solve_adaptive", &RK::solveAdaptive, py::arg("y0"), py::arg("dy0"), py::arg("dt"), py::arg("f"),
                 py::arg("tolerance")=1e-6, py::arg("dt_min")=1e-10, py::arg("dt_max")=1.0)
            .def("solve_sequence", &RK::solveSequence)
            .def_static("create_rk4", &RK::createRK4)
            .def_static("create_rk45", &RK::createRK45)
            .def_static("create_dopri5", &RK::createDOPRI5)
            .def_static("create_rk8", &RK::createRK8);

    using FD = OceanSim::Algorithms::FiniteDifferenceSolver;
    py::class_<FD, std::shared_ptr<FD>>(m, "FiniteDifferenceSolver")
            .def(py::init<int,double>())
            .def("solve_advection_diffusion", &FD::solveAdvectionDiffusion)
            .def("compute_spatial_derivative", &FD::computeSpatialDerivative,
                 py::arg("field"), py::arg("derivative"), py::arg("order")=1)
            .def("set_boundary_conditions", &FD::setBoundaryConditions)
            .def("get_grid_spacing", &FD::getGridSpacing)
            .def("get_time_step", &FD::getTimeStep);
}
