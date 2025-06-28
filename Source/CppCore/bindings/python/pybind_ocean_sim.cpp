// bindings/python/pybind_ocean_sim.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "core/ParticleSimulator.h"
#include "core/CurrentFieldSolver.h"
#include "core/AdvectionDiffusionSolver.h"
#include "data/GridDataStructure.h"
#include "algorithms/RungeKuttaSolver.h"
#include "algorithms/FiniteDifferenceSolver.h"
#include "utils/MathUtils.h"
#include "utils/Logger.h"

namespace py = pybind11;
using namespace OceanSim;

PYBIND11_MODULE(OceanSimPython, m) {
m.doc() = "OceanSim C++ Core Python Bindings";
m.attr("__version__") = "1.0.0";

// ========================= 初始化模块 =========================
m.def("initialize", &Utils::Logger::initialize, "Initialize the OceanSim library",
py::arg("name") = "OceanSim");

m.def("set_log_level", [](int level) {
// 设置日志级别的实现
}, "Set logging level");

// ========================= 数据结构绑定 =========================

// Vector3D 辅助类
py::class_<Eigen::Vector3d>(m, "Vector3D")
.def(py::init<double, double, double>())
.def(py::init<>())
.def_property("x",
[](const Eigen::Vector3d& v) { return v.x(); },
[](Eigen::Vector3d& v, double x) { v.x() = x; })
.def_property("y",
[](const Eigen::Vector3d& v) { return v.y(); },
[](Eigen::Vector3d& v, double y) { v.y() = y; })
.def_property("z",
[](const Eigen::Vector3d& v) { return v.z(); },
[](Eigen::Vector3d& v, double z) { v.z() = z; })
.def("norm", &Eigen::Vector3d::norm)
.def("normalized", &Eigen::Vector3d::normalized)
.def("__repr__", [](const Eigen::Vector3d& v) {
return "Vector3D(" + std::to_string(v.x()) + ", " +
std::to_string(v.y()) + ", " + std::to_string(v.z()) + ")";
});

// ========================= 网格数据结构 =========================

py::enum_<Data::GridDataStructure::CoordinateSystem>(m, "CoordinateSystem")
.value("CARTESIAN", Data::GridDataStructure::CoordinateSystem::CARTESIAN)
.value("SPHERICAL", Data::GridDataStructure::CoordinateSystem::SPHERICAL)
.value("HYBRID_SIGMA", Data::GridDataStructure::CoordinateSystem::HYBRID_SIGMA)
.value("ISOPYCNAL", Data::GridDataStructure::CoordinateSystem::ISOPYCNAL);

py::enum_<Data::GridDataStructure::GridType>(m, "GridType")
.value("REGULAR", Data::GridDataStructure::GridType::REGULAR)
.value("CURVILINEAR", Data::GridDataStructure::GridType::CURVILINEAR)
.value("UNSTRUCTURED", Data::GridDataStructure::GridType::UNSTRUCTURED);

py::enum_<Data::GridDataStructure::InterpolationMethod>(m, "InterpolationMethod")
.value("LINEAR", Data::GridDataStructure::InterpolationMethod::LINEAR)
.value("CUBIC", Data::GridDataStructure::InterpolationMethod::CUBIC)
.value("BILINEAR", Data::GridDataStructure::InterpolationMethod::BILINEAR)
.value("TRILINEAR", Data::GridDataStructure::InterpolationMethod::TRILINEAR)
.value("CONSERVATIVE", Data::GridDataStructure::InterpolationMethod::CONSERVATIVE);

py::class_<Data::GridDataStructure, std::shared_ptr<Data::GridDataStructure>>(m, "GridDataStructure")
.def(py::init<int, int, int, Data::GridDataStructure::CoordinateSystem, Data::GridDataStructure::GridType>(),
        py::arg("nx"), py::arg("ny"), py::arg("nz"),
        py::arg("coord_sys") = Data::GridDataStructure::CoordinateSystem::CARTESIAN,
py::arg("grid_type") = Data::GridDataStructure::GridType::REGULAR)

.def("set_spacing", py::overload_cast<double, double, const std::vector<double>&>
(&Data::GridDataStructure::setSpacing))
.def("set_origin", &Data::GridDataStructure::setOrigin)
.def("set_bounds", &Data::GridDataStructure::setBounds)

.def("add_field_2d", &Data::GridDataStructure::addField)
.def("add_field_3d", py::overload_cast<const std::string&, const std::vector<Eigen::MatrixXd>&>
(&Data::GridDataStructure::addField))
.def("add_vector_field", &Data::GridDataStructure::addVectorField)

.def("get_field_2d", &Data::GridDataStructure::getField2D,
py::return_value_policy::reference_internal)
.def("get_field_3d", py::overload_cast<const std::string&>
(&Data::GridDataStructure::getField3D, py::const_),
py::return_value_policy::reference_internal)

.def("interpolate_scalar", &Data::GridDataStructure::interpolateScalar,
py::arg("position"), py::arg("field"),
py::arg("method") = Data::GridDataStructure::InterpolationMethod::TRILINEAR)
.def("interpolate_vector", &Data::GridDataStructure::interpolateVector,
py::arg("position"), py::arg("field"),
py::arg("method") = Data::GridDataStructure::InterpolationMethod::TRILINEAR)

.def("compute_gradient", py::overload_cast<const std::string&, const Eigen::Vector3d&>
(&Data::GridDataStructure::computeGradient, py::const_))
.def("compute_divergence", &Data::GridDataStructure::computeDivergence)
.def("compute_curl", &Data::GridDataStructure::computeCurl)

.def("get_dimensions", &Data::GridDataStructure::getDimensions)
.def("get_spacing", &Data::GridDataStructure::getSpacing)
.def("get_origin", &Data::GridDataStructure::getOrigin)
.def("get_bounds", &Data::GridDataStructure::getBounds)

.def("has_field", &Data::GridDataStructure::hasField)
.def("get_field_names", &Data::GridDataStructure::getFieldNames)
.def("clear_field", &Data::GridDataStructure::clearField)
.def("get_memory_usage", &Data::GridDataStructure::getMemoryUsage)

.def("load_from_netcdf", &Data::GridDataStructure::loadFromNetCDF)
.def("save_to_netcdf", &Data::GridDataStructure::saveToNetCDF)

.def("get_min_value", &Data::GridDataStructure::getMinValue)
.def("get_max_value", &Data::GridDataStructure::getMaxValue)
.def("get_mean_value", &Data::GridDataStructure::getMeanValue);

// ========================= 粒子模拟器 =========================

py::class_<Core::ParticleSimulator::Particle>(m, "Particle")
.def(py::init<>())
.def(py::init<const Eigen::Vector3d&>())
.def_readwrite("position", &Core::ParticleSimulator::Particle::position)
.def_readwrite("velocity", &Core::ParticleSimulator::Particle::velocity)
.def_readwrite("age", &Core::ParticleSimulator::Particle::age)
.def_readwrite("id", &Core::ParticleSimulator::Particle::id)
.def_readwrite("active", &Core::ParticleSimulator::Particle::active);

py::class_<Core::ParticleSimulator, std::shared_ptr<Core::ParticleSimulator>>(m, "ParticleSimulator")
.def(py::init<std::shared_ptr<Data::GridDataStructure>,
        std::shared_ptr<Algorithms::RungeKuttaSolver>>())

.def("initialize_particles", &Core::ParticleSimulator::initializeParticles)
.def("initialize_random_particles", &Core::ParticleSimulator::initializeRandomParticles)

.def("step_forward", &Core::ParticleSimulator::stepForward)
.def("step_forward_adaptive", &Core::ParticleSimulator::stepForwardAdaptive,
py::arg("dt"), py::arg("tolerance") = 1e-6)

.def("set_num_threads", &Core::ParticleSimulator::setNumThreads)
.def("enable_vectorization", &Core::ParticleSimulator::enableVectorization)

.def("get_particles", &Core::ParticleSimulator::getParticles,
py::return_value_policy::reference_internal)
.def("get_trajectories", &Core::ParticleSimulator::getTrajectories)
.def("get_active_particle_count", &Core::ParticleSimulator::getActiveParticleCount)
.def("get_computation_time", &Core::ParticleSimulator::getComputationTime)

.def("set_boundary_conditions", &Core::ParticleSimulator::setBoundaryConditions)
.def("enable_diffusion", &Core::ParticleSimulator::enableDiffusion)
.def("set_source_term", &Core::ParticleSimulator::setSourceTerm);

// ========================= 洋流场求解器 =========================

py::class_<Core::CurrentFieldSolver::PhysicalParameters>(m, "PhysicalParameters")
.def(py::init<>())
.def_readwrite("gravity", &Core::CurrentFieldSolver::PhysicalParameters::gravity)
.def_readwrite("coriolis_f", &Core::CurrentFieldSolver::PhysicalParameters::coriolis_f)
.def_readwrite("beta", &Core::CurrentFieldSolver::PhysicalParameters::beta)
.def_readwrite("viscosity_h", &Core::CurrentFieldSolver::PhysicalParameters::viscosity_h)
.def_readwrite("viscosity_v", &Core::CurrentFieldSolver::PhysicalParameters::viscosity_v)
.def_readwrite("diffusivity_h", &Core::CurrentFieldSolver::PhysicalParameters::diffusivity_h)
.def_readwrite("diffusivity_v", &Core::CurrentFieldSolver::PhysicalParameters::diffusivity_v)
.def_readwrite("reference_density", &Core::CurrentFieldSolver::PhysicalParameters::reference_density);

py::class_<Core::CurrentFieldSolver::OceanState>(m, "OceanState")
.def(py::init<>())
.def(py::init<int, int, int>())
.def_readwrite("u", &Core::CurrentFieldSolver::OceanState::u)
.def_readwrite("v", &Core::CurrentFieldSolver::OceanState::v)
.def_readwrite("w", &Core::CurrentFieldSolver::OceanState::w)
.def_readwrite("temperature", &Core::CurrentFieldSolver::OceanState::temperature)
.def_readwrite("salinity", &Core::CurrentFieldSolver::OceanState::salinity)
.def_readwrite("density", &Core::CurrentFieldSolver::OceanState::density)
.def_readwrite("pressure", &Core::CurrentFieldSolver::OceanState::pressure)
.def_readwrite("ssh", &Core::CurrentFieldSolver::OceanState::ssh)
.def("resize", &Core::CurrentFieldSolver::OceanState::resize);

py::class_<Core::CurrentFieldSolver, std::shared_ptr<Core::CurrentFieldSolver>>(m, "CurrentFieldSolver")
.def(py::init<std::shared_ptr<Data::GridDataStructure>,
const Core::CurrentFieldSolver::PhysicalParameters&>(),
py::arg("grid"), py::arg("params") = Core::CurrentFieldSolver::PhysicalParameters{})

.def("initialize", &Core::CurrentFieldSolver::initialize)
.def("set_bottom_topography", &Core::CurrentFieldSolver::setBottomTopography)
.def("set_wind_stress", &Core::CurrentFieldSolver::setWindStress)

.def("step_forward", &Core::CurrentFieldSolver::stepForward)
.def("step_forward_barotropic", &Core::CurrentFieldSolver::stepForwardBarotropic)
.def("step_forward_baroclinic", &Core::CurrentFieldSolver::stepForwardBaroclinic)

.def("solve_navier_stokes", &Core::CurrentFieldSolver::solveNavierStokes)
.def("solve_continuity_equation", &Core::CurrentFieldSolver::solveContinuityEquation)
.def("solve_temperature_equation", &Core::CurrentFieldSolver::solveTemperatureEquation)
.def("solve_salinity_equation", &Core::CurrentFieldSolver::solveSalinityEquation)
.def("compute_density", &Core::CurrentFieldSolver::computeDensity)

.def("get_current_state", py::overload_cast<>(&Core::CurrentFieldSolver::getCurrentState),
        py::return_value_policy::reference_internal)

.def("compute_vorticity", &Core::CurrentFieldSolver::computeVorticity)
.def("compute_divergence", &Core::CurrentFieldSolver::computeDivergence)
.def("compute_kinetic_energy", &Core::CurrentFieldSolver::computeKineticEnergy)
.def("compute_total_energy", &Core::CurrentFieldSolver::computeTotalEnergy)

.def("check_mass_conservation", &Core::CurrentFieldSolver::checkMassConservation,
py::arg("tolerance") = 1e-10)
.def("compute_mass_imbalance", &Core::CurrentFieldSolver::computeMassImbalance);

// ========================= 平流扩散求解器 =========================

py::enum_<Core::AdvectionDiffusionSolver::NumericalScheme>(m, "NumericalScheme")
.value("UPWIND", Core::AdvectionDiffusionSolver::NumericalScheme::UPWIND)
.value("LAX_WENDROFF", Core::AdvectionDiffusionSolver::NumericalScheme::LAX_WENDROFF)
.value("TVD_SUPERBEE", Core::AdvectionDiffusionSolver::NumericalScheme::TVD_SUPERBEE)
.value("WENO5", Core::AdvectionDiffusionSolver::NumericalScheme::WENO5)
.value("QUICK", Core::AdvectionDiffusionSolver::NumericalScheme::QUICK)
.value("MUSCL", Core::AdvectionDiffusionSolver::NumericalScheme::MUSCL);

py::enum_<Core::AdvectionDiffusionSolver::TimeIntegration>(m, "TimeIntegration")
.value("EXPLICIT_EULER", Core::AdvectionDiffusionSolver::TimeIntegration::EXPLICIT_EULER)
.value("IMPLICIT_EULER", Core::AdvectionDiffusionSolver::TimeIntegration::IMPLICIT_EULER)
.value("CRANK_NICOLSON", Core::AdvectionDiffusionSolver::TimeIntegration::CRANK_NICOLSON)
.value("RUNGE_KUTTA_4", Core::AdvectionDiffusionSolver::TimeIntegration::RUNGE_KUTTA_4)
.value("ADAMS_BASHFORTH", Core::AdvectionDiffusionSolver::TimeIntegration::ADAMS_BASHFORTH);

py::enum_<Core::AdvectionDiffusionSolver::BoundaryType>(m, "BoundaryType")
.value("DIRICHLET", Core::AdvectionDiffusionSolver::BoundaryType::DIRICHLET)
.value("NEUMANN", Core::AdvectionDiffusionSolver::BoundaryType::NEUMANN)
.value("ROBIN", Core::AdvectionDiffusionSolver::BoundaryType::ROBIN)
.value("PERIODIC", Core::AdvectionDiffusionSolver::BoundaryType::PERIODIC)
.value("OUTFLOW", Core::AdvectionDiffusionSolver::BoundaryType::OUTFLOW);

py::class_<Core::AdvectionDiffusionSolver, std::shared_ptr<Core::AdvectionDiffusionSolver>>(m, "AdvectionDiffusionSolver")
.def(py::init<std::shared_ptr<Data::GridDataStructure>,
        Core::AdvectionDiffusionSolver::NumericalScheme,
        Core::AdvectionDiffusionSolver::TimeIntegration>(),
py::arg("grid"),
py::arg("scheme") = Core::AdvectionDiffusionSolver::NumericalScheme::TVD_SUPERBEE,
py::arg("time_method") = Core::AdvectionDiffusionSolver::TimeIntegration::RUNGE_KUTTA_4)

.def("set_initial_condition", &Core::AdvectionDiffusionSolver::setInitialCondition)
.def("set_velocity_field", &Core::AdvectionDiffusionSolver::setVelocityField,
py::arg("u_field"), py::arg("v_field"), py::arg("w_field") = "")

.def("set_diffusion_coefficient", &Core::AdvectionDiffusionSolver::setDiffusionCoefficient)
.def("set_diffusion_tensor", &Core::AdvectionDiffusionSolver::setDiffusionTensor)
.def("set_reaction_term", &Core::AdvectionDiffusionSolver::setReactionTerm)
.def("set_source_term", &Core::AdvectionDiffusionSolver::setSourceTerm)

.def("set_boundary_condition", &Core::AdvectionDiffusionSolver::setBoundaryCondition,
py::arg("type"), py::arg("boundary_id"), py::arg("value") = 0.0)

.def("solve", &Core::AdvectionDiffusionSolver::solve)
.def("solve_to_steady_state", &Core::AdvectionDiffusionSolver::solveToSteadyState,
py::arg("tolerance") = 1e-6, py::arg("max_iterations") = 10000)
.def("solve_time_sequence", &Core::AdvectionDiffusionSolver::solveTimeSequence,
py::arg("t_start"), py::arg("t_end"), py::arg("dt"), py::arg("output_prefix") = "")

.def("enable_adaptive_timestep", &Core::AdvectionDiffusionSolver::enableAdaptiveTimeStep,
py::arg("enable"), py::arg("cfl_target") = 0.5)
.def("compute_optimal_timestep", &Core::AdvectionDiffusionSolver::computeOptimalTimeStep)

.def("get_solution", py::overload_cast<>(&Core::AdvectionDiffusionSolver::getSolution),
        py::return_value_policy::reference_internal)
.def("get_max_concentration", &Core::AdvectionDiffusionSolver::getMaxConcentration)
.def("get_total_mass", &Core::AdvectionDiffusionSolver::getTotalMass)

.def("check_mass_conservation", &Core::AdvectionDiffusionSolver::checkMassConservation,
py::arg("tolerance") = 1e-8)
.def("compute_mass_balance", &Core::AdvectionDiffusionSolver::computeMassBalance)

.def("compute_peclet_number", &Core::AdvectionDiffusionSolver::computePecletNumber)
.def("compute_courant_number", &Core::AdvectionDiffusionSolver::computeCourantNumber)

.def("enable_profiling", &Core::AdvectionDiffusionSolver::enableProfiling)
.def("get_computation_time", &Core::AdvectionDiffusionSolver::getComputationTime)
.def("get_iteration_count", &Core::AdvectionDiffusionSolver::getIterationCount);

// ========================= 龙格-库塔求解器 =========================

py::enum_<Algorithms::RungeKuttaSolver::Method>(m, "RKMethod")
.value("RK4", Algorithms::RungeKuttaSolver::Method::RK4)
.value("RK45", Algorithms::RungeKuttaSolver::Method::RK45)
.value("DOPRI5", Algorithms::RungeKuttaSolver::Method::DOPRI5)
.value("RK8", Algorithms::RungeKuttaSolver::Method::RK8)
.value("GAUSS_LEGENDRE", Algorithms::RungeKuttaSolver::Method::GAUSS_LEGENDRE);

py::class_<Algorithms::RungeKuttaSolver::ButcherTableau>(m, "ButcherTableau")
.def(py::init<>())
.def_readwrite("A", &Algorithms::RungeKuttaSolver::ButcherTableau::A)
.def_readwrite("b", &Algorithms::RungeKuttaSolver::ButcherTableau::b)
.def_readwrite("c", &Algorithms::RungeKuttaSolver::ButcherTableau::c)
.def_readwrite("b_err", &Algorithms::RungeKuttaSolver::ButcherTableau::b_err)
.def_readwrite("order", &Algorithms::RungeKuttaSolver::ButcherTableau::order)
.def_readwrite("is_implicit", &Algorithms::RungeKuttaSolver::ButcherTableau::is_implicit);

py::class_<Algorithms::RungeKuttaSolver, std::shared_ptr<Algorithms::RungeKuttaSolver>>(m, "RungeKuttaSolver")
.def(py::init<Algorithms::RungeKuttaSolver::Method>(),
        py::arg("method") = Algorithms::RungeKuttaSolver::Method::RK4)
.def(py::init<const Algorithms::RungeKuttaSolver::ButcherTableau&>())

.def("solve", py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&, double,
        const Algorithms::RungeKuttaSolver::VectorFunction&>
(&Algorithms::RungeKuttaSolver::solve, py::const_))

.def("solve_adaptive", &Algorithms::RungeKuttaSolver::solveAdaptive,
py::arg("y0"), py::arg("dy0"), py::arg("dt"), py::arg("f"),
py::arg("tolerance") = 1e-6, py::arg("dt_min") = 1e-10, py::arg("dt_max") = 1.0)

.def("solve_sequence", &Algorithms::RungeKuttaSolver::solveSequence)

.def("set_method", &Algorithms::RungeKuttaSolver::setMethod)
.def("set_custom_tableau", &Algorithms::RungeKuttaSolver::setCustomTableau)
.def("set_tolerance", &Algorithms::RungeKuttaSolver::setTolerance)
.def("set_max_iterations", &Algorithms::RungeKuttaSolver::setMaxIterations)

.def("get_order", &Algorithms::RungeKuttaSolver::getOrder)
.def("get_stages", &Algorithms::RungeKuttaSolver::getStages)
.def("is_implicit", &Algorithms::RungeKuttaSolver::isImplicit)

.def_static("create_rk4", &Algorithms::RungeKuttaSolver::createRK4)
.def_static("create_rk45", &Algorithms::RungeKuttaSolver::createRK45)
.def_static("create_dopri5", &Algorithms::RungeKuttaSolver::createDOPRI5)
.def_static("create_rk8", &Algorithms::RungeKuttaSolver::createRK8)
.def_static("create_gauss_legendre", &Algorithms::RungeKuttaSolver::createGaussLegendre);

// ========================= 向量化求解器 =========================

py::class_<Algorithms::VectorizedRKSolver>(m, "VectorizedRKSolver")
.def(py::init<Algorithms::RungeKuttaSolver::Method>(),
        py::arg("method") = Algorithms::RungeKuttaSolver::Method::RK4)
.def("solve_batch", &Algorithms::VectorizedRKSolver::solveBatch)
.def("solve_batch_simd", &Algorithms::VectorizedRKSolver::solveBatchSIMD);

// ========================= 工具函数 =========================

py::module utils = m.def_submodule("utils", "Utility functions");

// 数学工具
utils.def("degrees_to_radians", [](double degrees) {
return degrees * M_PI / 180.0;
});

utils.def("radians_to_degrees", [](double radians) {
return radians * 180.0 / M_PI;
});

utils.def("haversine_distance", [](double lat1, double lon1, double lat2, double lon2) {
// 地球半径（千米）
const double R = 6371.0;

double dlat = (lat2 - lat1) * M_PI / 180.0;
double dlon = (lon2 - lon1) * M_PI / 180.0;
lat1 *= M_PI / 180.0;
lat2 *= M_PI / 180.0;

double a = sin(dlat/2) * sin(dlat/2) +
           cos(lat1) * cos(lat2) * sin(dlon/2) * sin(dlon/2);
double c = 2 * atan2(sqrt(a), sqrt(1-a));

return R * c;
}, "Calculate great circle distance between two points");

// 性能监控
utils.def("get_memory_usage", []() {
// 获取当前内存使用量的简单实现
return 0.0; // MB
});

utils.def("get_cpu_count", []() {
return std::thread::hardware_concurrency();
});

// ========================= 常量定义 =========================

m.attr("EARTH_RADIUS") = 6371000.0;  // 地球半径（米）
m.attr("GRAVITY") = 9.81;            // 重力加速度
m.attr("SEAWATER_DENSITY") = 1025.0; // 海水密度
m.attr("PI") = M_PI;

// ========================= 异常处理 =========================

py::register_exception<std::runtime_error>(m, "OceanSimError");
py::register_exception<std::invalid_argument>(m, "InvalidArgumentError");
py::register_exception<std::out_of_range>(m, "OutOfRangeError");
}