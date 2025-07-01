#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include "core/ParticleSimulator.h"  // TODO: verify path
namespace py = pybind11;

void bind_particle_sim(py::module_ &m)
{
    using Sim = OceanSim::Core::ParticleSimulator;
    py::class_<Sim::Particle>(m, "Particle")
            .def(py::init<>())
            .def(py::init<const Eigen::Vector3d &>())
            .def_readwrite("position", &Sim::Particle::position)
            .def_readwrite("velocity", &Sim::Particle::velocity);
    py::class_<Sim, std::shared_ptr<Sim>>(m, "ParticleSimulator")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>,
                    std::shared_ptr<OceanSim::Algorithms::RungeKuttaSolver>>())
            .def("initialize_particles", &Sim::initializeParticles)
            .def("step_forward", &Sim::stepForward);
}