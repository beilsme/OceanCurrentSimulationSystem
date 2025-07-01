#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <memory>
#include "core/AdvectionDiffusionSolver.h"
namespace py = pybind11;

void bind_advection(py::module_ &m)
{
    using Solver = OceanSim::Core::AdvectionDiffusionSolver;
    py::enum_<Solver::NumericalScheme>(m, "NumericalScheme")
            .value("UPWIND", Solver::NumericalScheme::UPWIND)
            .value("LAX_WENDROFF", Solver::NumericalScheme::LAX_WENDROFF)
            .value("TVD_SUPERBEE", Solver::NumericalScheme::TVD_SUPERBEE)
            .value("WENO5", Solver::NumericalScheme::WENO5)
            .value("QUICK", Solver::NumericalScheme::QUICK)
            .value("MUSCL", Solver::NumericalScheme::MUSCL);

    py::enum_<Solver::TimeIntegration>(m, "TimeIntegration")
            .value("EXPLICIT_EULER", Solver::TimeIntegration::EXPLICIT_EULER)
            .value("IMPLICIT_EULER", Solver::TimeIntegration::IMPLICIT_EULER)
            .value("CRANK_NICOLSON", Solver::TimeIntegration::CRANK_NICOLSON)
            .value("RUNGE_KUTTA_4", Solver::TimeIntegration::RUNGE_KUTTA_4)
            .value("ADAMS_BASHFORTH", Solver::TimeIntegration::ADAMS_BASHFORTH);

    py::enum_<Solver::BoundaryType>(m, "BoundaryType")
            .value("DIRICHLET", Solver::BoundaryType::DIRICHLET)
            .value("NEUMANN", Solver::BoundaryType::NEUMANN)
            .value("ROBIN", Solver::BoundaryType::ROBIN)
            .value("PERIODIC", Solver::BoundaryType::PERIODIC)
            .value("OUTFLOW", Solver::BoundaryType::OUTFLOW);

    py::class_<Solver, std::shared_ptr<Solver>>(m, "AdvectionDiffusionSolver")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>,
                         Solver::NumericalScheme,
                         Solver::TimeIntegration>(),
                 py::arg("grid"),
                 py::arg("scheme") = Solver::NumericalScheme::TVD_SUPERBEE,
                 py::arg("time_method") = Solver::TimeIntegration::RUNGE_KUTTA_4)
            .def("set_initial_condition", &Solver::setInitialCondition)
            .def("set_velocity_field", &Solver::setVelocityField,
                 py::arg("u_field"), py::arg("v_field"), py::arg("w_field") = "")
            .def("set_diffusion_coefficient", &Solver::setDiffusionCoefficient)
            .def("solve", &Solver::solve);
}