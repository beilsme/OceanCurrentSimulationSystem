#include <pybind11/pybind11.h>

void bind_grid(pybind11::module_ &m)
{
    m.def("grid_test", []() { return "grid ok"; });
}
