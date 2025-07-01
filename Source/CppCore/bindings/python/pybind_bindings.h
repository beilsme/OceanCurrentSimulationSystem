#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_advection(py::module_ &);
void bind_current_solver(py::module_ &);
void bind_particle_sim(py::module_ &);