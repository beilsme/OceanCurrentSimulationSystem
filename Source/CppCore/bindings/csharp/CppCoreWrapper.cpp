// bindings/csharp/CppCoreWrapper.cpp
#include "CppCoreWrapper.h"
#include "core/ParticleSimulator.h"
#include "core/CurrentFieldSolver.h"
#include "core/AdvectionDiffusionSolver.h"
#include "data/GridDataStructure.h"
#include "algorithms/RungeKuttaSolver.h"
#include "utils/Logger.h"
#include <unordered_map>
#include <memory>
#include <cstring>
#include <thread>

using namespace OceanSim;

// ========================= 全局状态管理 =========================

static ErrorCode g_last_error = ERROR_NONE;
static std::string g_error_message;
static bool g_initialized = false;

// 句柄管理
static std::unordered_map<void*, std::shared_ptr<Core::ParticleSimulator>> g_particle_simulators;
static std::unordered_map<void*, std::shared_ptr<Core::CurrentFieldSolver>> g_current_solvers;
static std::unordered_map<void*, std::shared_ptr<Data::GridDataStructure>> g_grid_data;
static std::unordered_map<void*, std::shared_ptr<Core::AdvectionDiffusionSolver>> g_ad_solvers;

// 辅助函数
void SetError(ErrorCode code, const std::string& message) {
    g_last_error = code;
    g_error_message = message;
}

template<typename T>
T* GetHandle(void* handle, const std::unordered_map<void*, std::shared_ptr<T>>& container) {
    auto it = container.find(handle);
    if (it != container.end()) {
        return it->second.get();
    }
    SetError(ERROR_INVALID_HANDLE, "Invalid handle provided");
    return nullptr;
}

// ========================= 初始化和清理 =========================

OCEANSIM_API void InitializeOceanSimCore() {
    if (!g_initialized) {
        Utils::Logger::initialize("OceanSimCore");
        Utils::Logger::info("OceanSim C++ Core initialized");
        g_initialized = true;
    }
}

OCEANSIM_API void ShutdownOceanSimCore() {
    if (g_initialized) {
        g_particle_simulators.clear();
        g_current_solvers.clear();
        g_grid_data.clear();
        g_ad_solvers.clear();
        Utils::Logger::info("OceanSim C++ Core shutdown");
        g_initialized = false;
    }
}

OCEANSIM_API const char* GetVersionString() {
    return "OceanSim C++ Core v1.0.0";
}

OCEANSIM_API int GetThreadCount() {
    return static_cast<int>(std::thread::hardware_concurrency());
}

OCEANSIM_API void SetThreadCount(int thread_count) {
    omp_set_num_threads(thread_count);
}

// ========================= 错误处理接口 =========================

OCEANSIM_API ErrorCode GetLastError() {
    return g_last_error;
}

OCEANSIM_API const char* GetErrorMessage(ErrorCode error_code) {
    switch (error_code) {
        case ERROR_NONE: return "No error";
        case ERROR_INVALID_HANDLE: return "Invalid handle";
        case ERROR_INVALID_PARAMETER: return "Invalid parameter";
        case ERROR_MEMORY_ALLOCATION: return "Memory allocation failed";
        case ERROR_FILE_IO: return "File I/O error";
        case ERROR_NUMERICAL_INSTABILITY: return "Numerical instability detected";
        case ERROR_CONVERGENCE_FAILURE: return "Convergence failure";
        default: return "Unknown error";
    }
}

OCEANSIM_API void ClearError() {
    g_last_error = ERROR_NONE;
    g_error_message.clear();
}

// ========================= 网格数据结构接口 =========================

OCEANSIM_API GridDataHandle CreateGridData(int nx, int ny, int nz,
                                           CoordinateSystemType coord_sys,
                                           GridTypeEnum grid_type) {
    try {
        auto coord_system = static_cast<Data::GridDataStructure::CoordinateSystem>(coord_sys);
        auto grid_type_enum = static_cast<Data::GridDataStructure::GridType>(grid_type);

        auto grid = std::make_shared<Data::GridDataStructure>(nx, ny, nz, coord_system, grid_type_enum);
        void* handle = grid.get();
        g_grid_data[handle] = grid;

        return handle;
    } catch (const std::exception& e) {
        SetError(ERROR_MEMORY_ALLOCATION, e.what());
        return nullptr;
    }
}

OCEANSIM_API void DestroyGridData(GridDataHandle handle) {
    auto it = g_grid_data.find(handle);
    if (it != g_grid_data.end()) {
        g_grid_data.erase(it);
    }
}

OCEANSIM_API void SetGridSpacing(GridDataHandle handle, double dx, double dy,
                                 const double* dz, int nz) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid) {
        std::vector<double> dz_vec(dz, dz + nz);
        grid->setSpacing(dx, dy, dz_vec);
    }
}

OCEANSIM_API void SetGridOrigin(GridDataHandle handle, const Vector3D* origin) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && origin) {
        Eigen::Vector3d eigen_origin(origin->x, origin->y, origin->z);
        grid->setOrigin(eigen_origin);
    }
}

OCEANSIM_API void AddScalarField2D(GridDataHandle handle, const char* name,
                                   const double* data, int nx, int ny) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && name && data) {
        Eigen::MatrixXd field(nx, ny);
        std::memcpy(field.data(), data, nx * ny * sizeof(double));
        grid->addField(name, field);
    }
}

OCEANSIM_API void AddVectorField(GridDataHandle handle, const char* name,
                                 const double* u_data, const double* v_data,
                                 const double* w_data, int nx, int ny, int nz) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && name && u_data && v_data && w_data) {
        std::vector<Eigen::MatrixXd> components(3);
        int size = nx * ny * nz;

        components[0].resize(nx, ny);
        components[1].resize(nx, ny);
        components[2].resize(nx, ny);

        std::memcpy(components[0].data(), u_data, size * sizeof(double));
        std::memcpy(components[1].data(), v_data, size * sizeof(double));
        std::memcpy(components[2].data(), w_data, size * sizeof(double));

        grid->addVectorField(name, components);
    }
}

OCEANSIM_API double InterpolateScalar(GridDataHandle handle, const char* field_name,
                                      const Vector3D* position, InterpolationMethodType method) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && field_name && position) {
        Eigen::Vector3d pos(position->x, position->y, position->z);
        auto interp_method = static_cast<Data::GridDataStructure::InterpolationMethod>(method);
        return grid->interpolateScalar(pos, field_name, interp_method);
    }
    return 0.0;
}

OCEANSIM_API Vector3D InterpolateVector(GridDataHandle handle, const char* field_name,
                                        const Vector3D* position, InterpolationMethodType method) {
    Vector3D result = {0.0, 0.0, 0.0};
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && field_name && position) {
        Eigen::Vector3d pos(position->x, position->y, position->z);
        auto interp_method = static_cast<Data::GridDataStructure::InterpolationMethod>(method);
        Eigen::Vector3d vec = grid->interpolateVector(pos, field_name, interp_method);
        result.x = vec.x();
        result.y = vec.y();
        result.z = vec.z();
    }
    return result;
}

OCEANSIM_API int HasField(GridDataHandle handle, const char* field_name) {
    auto* grid = GetHandle(handle, g_grid_data);
    if (grid && field_name) {
        return grid->hasField(field_name) ? 1 : 0;
    }
    return 0;
}

// ========================= 粒子模拟器接口 =========================

OCEANSIM_API ParticleSimulatorHandle CreateParticleSimulator(
        int nx, int ny, int nz, double dx, double dy, double dz) {
    try {
        auto grid = std::make_shared<Data::GridDataStructure>(nx, ny, nz);
        std::vector<double> dz_vec(nz, dz);
        grid->setSpacing(dx, dy, dz_vec);

        auto solver = std::make_shared<Algorithms::RungeKuttaSolver>();
        auto particle_sim = std::make_shared<Core::ParticleSimulator>(grid, solver);

        void* handle = particle_sim.get();
        g_particle_simulators[handle] = particle_sim;

        return handle;
    } catch (const std::exception& e) {
        SetError(ERROR_MEMORY_ALLOCATION, e.what());
        return nullptr;
    }
}

OCEANSIM_API void DestroyParticleSimulator(ParticleSimulatorHandle handle) {
    auto it = g_particle_simulators.find(handle);
    if (it != g_particle_simulators.end()) {
        g_particle_simulators.erase(it);
    }
}

OCEANSIM_API void InitializeParticles(ParticleSimulatorHandle handle,
                                      const Vector3D* positions, int count) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim && positions && count > 0) {
        std::vector<Eigen::Vector3d> pos_vec;
        pos_vec.reserve(count);

        for (int i = 0; i < count; ++i) {
            pos_vec.emplace_back(positions[i].x, positions[i].y, positions[i].z);
        }

        sim->initializeParticles(pos_vec);
    }
}

OCEANSIM_API void InitializeRandomParticles(ParticleSimulatorHandle handle,
                                            int count, const Vector3D* bounds_min,
                                            const Vector3D* bounds_max) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim && bounds_min && bounds_max && count > 0) {
        Eigen::Vector3d min_bounds(bounds_min->x, bounds_min->y, bounds_min->z);
        Eigen::Vector3d max_bounds(bounds_max->x, bounds_max->y, bounds_max->z);

        sim->initializeRandomParticles(count, min_bounds, max_bounds);
    }
}

OCEANSIM_API void StepParticlesForward(ParticleSimulatorHandle handle, double dt) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim) {
        sim->stepForward(dt);
    }
}

OCEANSIM_API int GetParticleCount(ParticleSimulatorHandle handle) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim) {
        return static_cast<int>(sim->getParticles().size());
    }
    return 0;
}

OCEANSIM_API void GetParticles(ParticleSimulatorHandle handle,
                               ParticleData* particles, int max_count) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim && particles) {
        const auto& sim_particles = sim->getParticles();
        int count = std::min(max_count, static_cast<int>(sim_particles.size()));

        for (int i = 0; i < count; ++i) {
            const auto& p = sim_particles[i];
            particles[i].position.x = p.position.x();
            particles[i].position.y = p.position.y();
            particles[i].position.z = p.position.z();
            particles[i].velocity.x = p.velocity.x();
            particles[i].velocity.y = p.velocity.y();
            particles[i].velocity.z = p.velocity.z();
            particles[i].age = p.age;
            particles[i].id = p.id;
            particles[i].active = p.active ? 1 : 0;
        }
    }
}

OCEANSIM_API void EnableParticleDiffusion(ParticleSimulatorHandle handle,
                                          double diffusion_coefficient) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim) {
        sim->enableDiffusion(diffusion_coefficient);
    }
}

OCEANSIM_API double GetParticleComputationTime(ParticleSimulatorHandle handle) {
    auto* sim = GetHandle(handle, g_particle_simulators);
    if (sim) {
        return sim->getComputationTime();
    }
    return 0.0;
}

// ========================= 洋流场求解器接口 =========================

OCEANSIM_API CurrentFieldSolverHandle CreateCurrentFieldSolver(
        int nx, int ny, int nz, double dx, double dy, double dz,
        const PhysicalParameters* params) {
    try {
        auto grid = std::make_shared<Data::GridDataStructure>(nx, ny, nz);
        std::vector<double> dz_vec(nz, dz);
        grid->setSpacing(dx, dy, dz_vec);

        Core::CurrentFieldSolver::PhysicalParameters cpp_params;
        if (params) {
            cpp_params.gravity = params->gravity;
            cpp_params.coriolis_f = params->coriolis_f;
            cpp_params.beta = params->beta;
            cpp_params.viscosity_h = params->viscosity_h;
            cpp_params.viscosity_v = params->viscosity_v;
            cpp_params.diffusivity_h = params->diffusivity_h;
            cpp_params.diffusivity_v = params->diffusivity_v;
            cpp_params.reference_density = params->reference_density;
        }

        auto solver = std::make_shared<Core::CurrentFieldSolver>(grid, cpp_params);
        void* handle = solver.get();
        g_current_solvers[handle] = solver;

        return handle;
    } catch (const std::exception& e) {
        SetError(ERROR_MEMORY_ALLOCATION, e.what());
        return nullptr;
    }
}

OCEANSIM_API void DestroyCurrentFieldSolver(CurrentFieldSolverHandle handle) {
    auto it = g_current_solvers.find(handle);
    if (it != g_current_solvers.end()) {
        g_current_solvers.erase(it);
    }
}

OCEANSIM_API void SetBottomTopography(CurrentFieldSolverHandle handle,
                                      const double* bottom_depth, int nx, int ny) {
    auto* solver = GetHandle(handle, g_current_solvers);
    if (solver && bottom_depth) {
        Eigen::MatrixXd topography(nx, ny);
        std::memcpy(topography.data(), bottom_depth, nx * ny * sizeof(double));
        solver->setBottomTopography(topography);
    }
}

OCEANSIM_API void StepCurrentFieldForward(CurrentFieldSolverHandle handle, double dt) {
    auto* solver = GetHandle(handle, g_current_solvers);
    if (solver) {
        solver->stepForward(dt);
    }
}

OCEANSIM_API double GetTotalEnergy(CurrentFieldSolverHandle handle) {
    auto* solver = GetHandle(handle, g_current_solvers);
    if (solver) {
        return solver->computeTotalEnergy();
    }
    return 0.0;
}

OCEANSIM_API int CheckMassConservation(CurrentFieldSolverHandle handle, double tolerance) {
    auto* solver = GetHandle(handle, g_current_solvers);
    if (solver) {
        return solver->checkMassConservation(tolerance) ? 1 : 0;
    }
    return 0;
}

// ========================= 平流扩散求解器接口 =========================

OCEANSIM_API AdvectionDiffusionHandle CreateAdvectionDiffusionSolver(
        GridDataHandle grid_handle, NumericalSchemeType scheme, TimeIntegrationType time_method) {
    try {
        auto* grid_ptr = GetHandle(grid_handle, g_grid_data);
        if (!grid_ptr) return nullptr;

        auto grid_shared = g_grid_data[grid_handle];
        auto num_scheme = static_cast<Core::AdvectionDiffusionSolver::NumericalScheme>(scheme);
        auto time_int = static_cast<Core::AdvectionDiffusionSolver::TimeIntegration>(time_method);

        auto solver = std::make_shared<Core::AdvectionDiffusionSolver>(grid_shared, num_scheme, time_int);
        void* handle = solver.get();
        g_ad_solvers[handle] = solver;

        return handle;
    } catch (const std::exception& e) {
        SetError(ERROR_MEMORY_ALLOCATION, e.what());
        return nullptr;
    }
}

OCEANSIM_API void DestroyAdvectionDiffusionSolver(AdvectionDiffusionHandle handle) {
    auto it = g_ad_solvers.find(handle);
    if (it != g_ad_solvers.end()) {
        g_ad_solvers.erase(it);
    }
}

OCEANSIM_API void SetDiffusionCoefficient(AdvectionDiffusionHandle handle,
                                          double diffusion_coeff) {
    auto* solver = GetHandle(handle, g_ad_solvers);
    if (solver) {
        solver->setDiffusionCoefficient(diffusion_coeff);
    }
}

OCEANSIM_API void SolveAdvectionDiffusion(AdvectionDiffusionHandle handle, double dt) {
    auto* solver = GetHandle(handle, g_ad_solvers);
    if (solver) {
        solver->solve(dt);
    }
}

OCEANSIM_API double GetMaxConcentration(AdvectionDiffusionHandle handle) {
    auto* solver = GetHandle(handle, g_ad_solvers);
    if (solver) {
        return solver->getMaxConcentration();
    }
    return 0.0;
}

OCEANSIM_API double ComputePecletNumber(AdvectionDiffusionHandle handle) {
    auto* solver = GetHandle(handle, g_ad_solvers);
    if (solver) {
        return solver->computePecletNumber();
    }
    return 0.0;
}

// ========================= 性能分析接口 =========================

OCEANSIM_API void GetPerformanceMetrics(void* handle, PerformanceMetrics* metrics) {
    if (!metrics) return;

    // 默认初始化
    metrics->computation_time = 0.0;
    metrics->memory_usage_mb = 0.0;
    metrics->iteration_count = 0;
    metrics->efficiency_ratio = 0.0;

    // 尝试从不同类型的求解器获取性能指标
    auto particle_it = g_particle_simulators.find(handle);
    if (particle_it != g_particle_simulators.end()) {
        metrics->computation_time = particle_it->second->getComputationTime();
        return;
    }

    auto current_it = g_current_solvers.find(handle);
    if (current_it != g_current_solvers.end()) {
        // CurrentFieldSolver的性能指标
        return;
    }

    auto ad_it = g_ad_solvers.find(handle);
    if (ad_it != g_ad_solvers.end()) {
        metrics->computation_time = ad_it->second->getComputationTime();
        metrics->iteration_count = ad_it->second->getIterationCount();
        return;
    }
}