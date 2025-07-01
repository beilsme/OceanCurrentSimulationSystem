#include "pybind_bindings.h"

PYBIND11_MODULE(oceansim, m) {
    m.doc() = "Ocean current simulation bindings";
    bind_grid(m);
    bind_algorithms(m);
    bind_parallel(m);
    bind_advection(m);
    bind_current_solver(m);
    bind_particle_sim(m);
}