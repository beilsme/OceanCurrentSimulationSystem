#include "pybind_bindings.h"

PYBIND11_MODULE(oceansim, m) {
    m.doc() = "Ocean current simulation bindings";

    m.attr("__version__") = "1.0.0";

    m.def("hello", []() {
        return "Ocean Simulation System Ready!";
    });

    m.def("test_add", [](float a, float b) {
        return a + b;
    }, "Test addition function");

    // 逐个测试绑定函数
    bind_grid(m);          // 先测试这个
    bind_algorithms(m);
    bind_parallel(m);
    bind_advection(m);
    bind_current_solver(m);
    bind_particle_sim(m);
}