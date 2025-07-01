// 更新后的 pybind_current_solver.cpp
#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <memory>
#include "core/CurrentFieldSolver.h"

namespace py = pybind11;

void bind_current_solver(py::module_ &m)
{
    using Solver = OceanSim::Core::CurrentFieldSolver;

    // 物理参数类 - 统一使用snake_case
    py::class_<Solver::PhysicalParameters>(m, "PhysicalParameters")
            .def(py::init<>())
            .def_readwrite("gravity", &Solver::PhysicalParameters::gravity)
            .def_readwrite("coriolis_f", &Solver::PhysicalParameters::coriolis_f)
            .def_readwrite("beta", &Solver::PhysicalParameters::beta)
            .def_readwrite("viscosity_h", &Solver::PhysicalParameters::viscosity_h)
            .def_readwrite("viscosity_v", &Solver::PhysicalParameters::viscosity_v)
            .def_readwrite("diffusivity_h", &Solver::PhysicalParameters::diffusivity_h)
            .def_readwrite("diffusivity_v", &Solver::PhysicalParameters::diffusivity_v)
            .def_readwrite("reference_density", &Solver::PhysicalParameters::reference_density);

    // 海洋状态类 - 统一属性访问方式
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
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.w = value; })
            .def_property("temperature",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.temperature; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.temperature = value; })
            .def_property("salinity",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.salinity; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.salinity = value; })
            .def_property("density",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.density; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.density = value; })
            .def_property("pressure",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.pressure; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.pressure = value; })
            .def_property("ssh",
                          [](Solver::OceanState &self) -> Eigen::Ref<Eigen::MatrixXd> { return self.ssh; },
                          [](Solver::OceanState &self, const Eigen::MatrixXd &value) { self.ssh = value; })
                    // 添加便捷方法
            .def("resize", &Solver::OceanState::resize, py::arg("nx"), py::arg("ny"), py::arg("nz"))
            .def("get_dimensions", [](const Solver::OceanState &self) {
                // 假设OceanState有获取维度的方法，如果没有需要添加
                return py::make_tuple(self.u.rows(), self.u.cols(), 1); // 简化示例
            });

    // 主求解器类 - 统一方法命名
    py::class_<Solver, std::shared_ptr<Solver>>(m, "CurrentFieldSolver")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>, const Solver::PhysicalParameters &>(),
                 py::arg("grid"), py::arg("params") = Solver::PhysicalParameters())

                    // 初始化和时间步进 - snake_case命名
            .def("initialize", &Solver::initialize, py::arg("initial_state"))
            .def("step_forward", &Solver::stepForward, py::arg("dt"))
            .def("step_forward_barotropic", &Solver::stepForwardBarotropic, py::arg("dt"))
            .def("step_forward_baroclinic", &Solver::stepForwardBaroclinic, py::arg("dt"))

                    // 状态访问
            .def("get_current_state",
                 static_cast<const Solver::OceanState&(Solver::*)() const>(&Solver::getCurrentState),
                 py::return_value_policy::reference_internal)
            .def("get_current_state_copy", [](Solver &self) {
                     return Solver::OceanState(self.getCurrentState());
                 }, py::return_value_policy::move,
                 "返回 OceanState 的拷贝，避免原始 solver 被销毁时出现悬空引用。")

                    // 诊断计算
            .def("compute_vorticity", &Solver::computeVorticity)
            .def("compute_divergence", &Solver::computeDivergence)
            .def("compute_kinetic_energy", &Solver::computeKineticEnergy)
            .def("compute_total_energy", &Solver::computeTotalEnergy)

                    // 质量守恒检查
            .def("check_mass_conservation", &Solver::checkMassConservation,
                 py::arg("tolerance") = 1e-10)
            .def("compute_mass_imbalance", &Solver::computeMassImbalance);
}
