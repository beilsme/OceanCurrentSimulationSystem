#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include "core/AdvectionDiffusionSolver.h"  // TODO: verify path
namespace py = pybind11;

void bind_advection(py::module_ &m)
{
    using Solver = OceanSim::Core::AdvectionDiffusionSolver;
    py::class_<Solver>(m, "AdvectionDiffusionSolver")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>,
                         Solver::NumericalScheme,
                         Solver::TimeIntegration>(),
                 py::arg("grid"),
                 py::arg("scheme"),
                 py::arg("time_method"))
            .def("solve", &Solver::solve);
}