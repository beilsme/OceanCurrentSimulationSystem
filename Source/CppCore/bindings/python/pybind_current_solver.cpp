#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include "core/CurrentFieldSolver.h"  // TODO: verify path
namespace py = pybind11;

void bind_current_solver(py::module_ &m)
{
    using Solver = OceanSim::Core::CurrentFieldSolver;
    py::class_<Solver::PhysicalParameters>(m, "PhysicalParameters")
            .def(py::init<>());
    py::class_<Solver>(m, "CurrentFieldSolver")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>,
                         const Solver::PhysicalParameters &>(),
                 py::arg("grid"), py::arg("params") = Solver::PhysicalParameters())
            .def("initialize", &Solver::initialize)
            .def("step_forward", &Solver::stepForward);
}