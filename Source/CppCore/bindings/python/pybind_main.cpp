#include <pybind11/pybind11.h>
namespace py = pybind11;

// 先声明各子模块 bind 函数
void bind_particle_sim(py::module_ &);
void bind_current_solver(py::module_ &);
void bind_advection(py::module_ &);

PYBIND11_MODULE(oceansim, m)     // ★ 导出名统一为 oceansim
{
    m.doc() = "Ocean current simulation bindings";
    bind_particle_sim(m);
    bind_current_solver(m);
    bind_advection(m);
}
