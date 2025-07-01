
/**
 * @file pybind_particle_sim.cpp
 * @author beilsm
 * @version 1.0
 * @brief Bindings for ParticleSimulator
 * @date 2025-07-01
 */
#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <memory>
#include "core/ParticleSimulator.h"
namespace py = pybind11;

void bind_particle_sim(py::module_ &m)
{
    using Sim = OceanSim::Core::ParticleSimulator;
    py::class_<Sim::Particle>(m, "Particle")
            .def(py::init<>())
            .def(py::init<const Eigen::Vector3d &>())
            .def_readwrite("position", &Sim::Particle::position)
            .def_readwrite("velocity", &Sim::Particle::velocity)
            .def_readwrite("age", &Sim::Particle::age)
            .def_readwrite("id", &Sim::Particle::id)
            .def_readwrite("active", &Sim::Particle::active);
    py::class_<Sim, std::shared_ptr<Sim>>(m, "ParticleSimulator")
            .def(py::init<std::shared_ptr<OceanSim::Data::GridDataStructure>,
                    std::shared_ptr<OceanSim::Algorithms::RungeKuttaSolver>>())
            .def("initialize_particles", &Sim::initializeParticles)
            .def("step_forward", &Sim::stepForward)
            .def("get_particles", &Sim::getParticles, py::return_value_policy::reference_internal);
}