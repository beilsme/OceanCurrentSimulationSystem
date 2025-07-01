/**
 * @file pybind_current_solver.cpp
 * @author beilsm
 * @version 1.1
 * @brief Bindings for CurrentFieldSolver
 * @date 2025-07-01
 * @details
 * - 功能: 绑定 CurrentFieldSolver 及其 OceanState
 * - 改进: 修复对 Eigen::MatrixXd 成员的绑定错误，使用 def_property
 * - 较上一版: 解决 def_readwrite 对 Eigen::MatrixXd 不匹配的问题
 * - 最新修改时间: 2025-07-01
 */

#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <memory>
#include "core/CurrentFieldSolver.h"

namespace py = pybind11;

void bind_current_solver(py::module_ &m)
{
    using Solver = OceanSim::Core::CurrentFieldSolver;

    py::class_<Solver::PhysicalParameters>(m, "PhysicalParameters")
            .def(py::init<>())
            .def_readwrite("gravity", &Solver::PhysicalParameters::gravity)
            .def_readwrite("coriolis_f", &Solver::PhysicalParameters::coriolis_f);

    py::class_<Solver::OceanState>(m, "OceanState")
            .def(py::init<int,int,int>(), py::arg("nx"), py::arg("ny"), py::arg("nz"))
            .def_property("u",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.u; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.u = value; })
            .def_property("v",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.v; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.v = value; })
            .def_property("w",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.w; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.w = value; });

    py::class_<Solver, std::shared_ptr<Solver>>(m, "CurrentFieldSolver")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>, const Solver::PhysicalParameters &>(),
                 py::arg("grid"), py::arg("params") = Solver::PhysicalParameters())
            .def("initialize", &Solver::initialize)
            .def("step_forward", &Solver::stepForward)
            .def("get_current_state", &Solver::getCurrentState, py::return_value_policy::reference_internal);
}
