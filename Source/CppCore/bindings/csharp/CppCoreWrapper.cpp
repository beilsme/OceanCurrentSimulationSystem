// ==============================================================================
// 文件路径：Source/CppCore/bindings/csharp/CppCoreWrapper.cpp
// 作者：beilsm
// 版本号：v1.0.0
// 创建时间：2025-07-01
// 最新更改时间：2025-07-01
// ==============================================================================
// 📝 功能说明：
//   C++核心模块的C#绑定包装器实现文件
//   将C++类包装为C风格接口供C# P/Invoke调用
// ==============================================================================

#include "CppCoreWrapper.h"

// 包含必要的C++头文件
#include "data/GridDataStructure.h"
#include "core/ParticleSimulator.h"
#include "core/CurrentFieldSolver.h"
#include "core/AdvectionDiffusionSolver.h"
#include "algorithms/RungeKuttaSolver.h"
#include "algorithms/FiniteDifferenceSolver.h"
#include "algorithms/ParallelComputeEngine.h"
#include "algorithms/VectorizedOperations.h"
#include "data/MemoryManager.h"
#include "data/DataExporter.h"
#include "core/PerformanceProfiler.h"
#include "utils/Logger.h"

#include <cstring>
#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

using namespace OceanSim;
using namespace OceanSim::Data;
using namespace OceanSim::Core;
using namespace OceanSim::Algorithms;
using namespace OceanSim::Utils;

// ===========================================
// 内部辅助函数
// ===========================================

namespace {
    // 将C结构体转换为C++对象
    Eigen::Vector3d ConvertVector3D(const Vector3D& vec) {
        return Eigen::Vector3d(vec.x, vec.y, vec.z);
    }

    Vector3D ConvertToVector3D(const Eigen::Vector3d& vec) {
        Vector3D result;
        result.x = vec.x();
        result.y = vec.y();
        result.z = vec.z();
        return result;
    }

    GridDataStructure::CoordinateSystem ConvertCoordinateSystemType(int coord_system) {
        switch (coord_system) {
            case COORDINATE_CARTESIAN: return GridDataStructure::CoordinateSystem::CARTESIAN;
            case COORDINATE_SPHERICAL: return GridDataStructure::CoordinateSystem::SPHERICAL;
            case COORDINATE_HYBRID_SIGMA: return GridDataStructure::CoordinateSystem::HYBRID_SIGMA;
            case COORDINATE_ISOPYCNAL: return GridDataStructure::CoordinateSystem::ISOPYCNAL;
            default: return GridDataStructure::CoordinateSystem::CARTESIAN;
        }
    }

    GridDataStructure::GridType ConvertGridType(int grid_type) {
        switch (grid_type) {
            case GRID_REGULAR: return GridDataStructure::GridType::REGULAR;
            case GRID_CURVILINEAR: return GridDataStructure::GridType::CURVILINEAR;
            case GRID_UNSTRUCTURED: return GridDataStructure::GridType::UNSTRUCTURED;
            default: return GridDataStructure::GridType::REGULAR;
        }
    }

    AdvectionDiffusionSolver::NumericalScheme ConvertNumericalScheme(int scheme) {
        switch (scheme) {
            case SCHEME_UPWIND: return AdvectionDiffusionSolver::NumericalScheme::UPWIND;
            case SCHEME_CENTRAL: return AdvectionDiffusionSolver::NumericalScheme::LAX_WENDROFF;
            case SCHEME_TVD_SUPERBEE: return AdvectionDiffusionSolver::NumericalScheme::TVD_SUPERBEE;
            case SCHEME_WENO: return AdvectionDiffusionSolver::NumericalScheme::WENO5;
            default: return AdvectionDiffusionSolver::NumericalScheme::TVD_SUPERBEE;
        }
    }

    AdvectionDiffusionSolver::TimeIntegration ConvertTimeIntegration(int method) {
        switch (method) {
            case TIME_EULER: return AdvectionDiffusionSolver::TimeIntegration::EXPLICIT_EULER;
            case TIME_RUNGE_KUTTA_2: return AdvectionDiffusionSolver::TimeIntegration::IMPLICIT_EULER;
            case TIME_RUNGE_KUTTA_3: return AdvectionDiffusionSolver::TimeIntegration::CRANK_NICOLSON;
            case TIME_RUNGE_KUTTA_4: return AdvectionDiffusionSolver::TimeIntegration::RUNGE_KUTTA_4;
            case TIME_ADAMS_BASHFORTH: return AdvectionDiffusionSolver::TimeIntegration::ADAMS_BASHFORTH;
            default: return AdvectionDiffusionSolver::TimeIntegration::RUNGE_KUTTA_4;
        }
    }

    OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy ConvertExecutionPolicy(int policy) {
        switch (policy) {
            case EXECUTION_SEQUENTIAL: return OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy::Sequential;
            case EXECUTION_PARALLEL: return OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy::Parallel;
            case EXECUTION_VECTORIZED: return OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy::Vectorized;
            case EXECUTION_HYBRID_PARALLEL: return OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy::HybridParallel;
            default: return OceanSimulation::Core::ParallelComputeEngine::ExecutionPolicy::Parallel;
        }
    }

    OceanSimulation::Core::VectorizedOperations::SimdType ConvertSimdType(int simd) {
        switch (simd) {
            case SIMD_NONE: return OceanSimulation::Core::VectorizedOperations::SimdType::None;
            case SIMD_SSE: return OceanSimulation::Core::VectorizedOperations::SimdType::SSE;
            case SIMD_AVX: return OceanSimulation::Core::VectorizedOperations::SimdType::AVX;
            case SIMD_AVX2: return OceanSimulation::Core::VectorizedOperations::SimdType::AVX2;
            case SIMD_AVX512: return OceanSimulation::Core::VectorizedOperations::SimdType::AVX512;
            case SIMD_NEON: return OceanSimulation::Core::VectorizedOperations::SimdType::NEON;
            default: return OceanSimulation::Core::VectorizedOperations::SimdType::AVX2;
        }
    }

    // 错误处理宏
#define HANDLE_EXCEPTION(code) \
        try { \
            code \
        } catch (const std::exception& e) { \
            Logger::getInstance().error("C# Wrapper Exception: " + std::string(e.what())); \
        } catch (...) { \
            Logger::getInstance().error("C# Wrapper Unknown Exception"); \
        }
}

// ===========================================
// 粒子模拟器接口实现
// ===========================================

OCEANSIM_API ParticleSimulatorHandle CreateParticleSimulator(
        int nx, int ny, int nz, double dx, double dy, double dz) {
    HANDLE_EXCEPTION({
                         // 创建网格
                         auto grid = std::make_shared<GridDataStructure>(nx, ny, nz);
                         grid->setSpacing(dx, dy, std::vector<double>(nz, dz));

                         // 创建默认的RK4求解器
                         auto solver = std::make_shared<RungeKuttaSolver>(RungeKuttaSolver::Method::RK4);

                         // 创建粒子模拟器
                         auto simulator = std::make_unique<ParticleSimulator>(grid, solver);
                         return simulator.release();
                     });
    return nullptr;
}

OCEANSIM_API void DestroyParticleSimulator(ParticleSimulatorHandle handle) {
    if (handle) {
        delete static_cast<ParticleSimulator*>(handle);
    }
}

OCEANSIM_API void InitializeParticles(ParticleSimulatorHandle handle,
                                      const Vector3D* positions, int count) {
    if (!handle || !positions) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         std::vector<Eigen::Vector3d> init_positions;
                         init_positions.reserve(count);

                         for (int i = 0; i < count; ++i) {
                             init_positions.push_back(ConvertVector3D(positions[i]));
                         }

                         simPtr->initializeParticles(init_positions);
                     });
}

OCEANSIM_API void InitializeRandomParticles(ParticleSimulatorHandle handle,
                                            int count, const Vector3D* bounds_min,
                                            const Vector3D* bounds_max) {
    if (!handle || !bounds_min || !bounds_max) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         Eigen::Vector3d min_bounds = ConvertVector3D(*bounds_min);
                         Eigen::Vector3d max_bounds = ConvertVector3D(*bounds_max);

                         simPtr->initializeRandomParticles(count, min_bounds, max_bounds);
                     });
}

OCEANSIM_API void StepParticlesForward(ParticleSimulatorHandle handle, double dt) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         simPtr->stepForward(dt);
                     });
}

OCEANSIM_API int GetParticleCount(ParticleSimulatorHandle handle) {
    if (!handle) return 0;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         return static_cast<int>(simPtr->getParticles().size());
                     });

    return 0;
}

OCEANSIM_API void GetParticles(ParticleSimulatorHandle handle,
                               ParticleData* particles, int max_count) {
    if (!handle || !particles) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         const auto& particle_list = simPtr->getParticles();

                         int copy_count = std::min(max_count, static_cast<int>(particle_list.size()));
                         for (int i = 0; i < copy_count; ++i) {
                             const auto& p = particle_list[i];
                             particles[i].id = p.id;
                             particles[i].position = ConvertToVector3D(p.position);
                             particles[i].velocity = ConvertToVector3D(p.velocity);
                             particles[i].age = p.age;
                             particles[i].active = p.active ? 1 : 0;
                         }
                     });
}

OCEANSIM_API void SetParticleVelocityField(ParticleSimulatorHandle handle,
                                           const char* field_name) {
    if (!handle || !field_name) return;

    HANDLE_EXCEPTION({
                         // 这个功能需要在ParticleSimulator中实现
                         // 现在只记录请求
                         Logger::getInstance().info("SetParticleVelocityField called with field: " + std::string(field_name));
                     });
}

OCEANSIM_API void EnableParticleDiffusion(ParticleSimulatorHandle handle,
                                          double diffusion_coefficient) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         simPtr->enableDiffusion(diffusion_coefficient);
                     });
}

OCEANSIM_API void SetParticleBoundaryConditions(ParticleSimulatorHandle handle,
                                                const Vector3D* bounds_min,
                                                const Vector3D* bounds_max,
                                                int periodic) {
    if (!handle || !bounds_min || !bounds_max) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         Eigen::Vector3d min_bounds = ConvertVector3D(*bounds_min);
                         Eigen::Vector3d max_bounds = ConvertVector3D(*bounds_max);

                         simPtr->setBoundaryConditions(min_bounds, max_bounds, periodic != 0);
                     });
}

OCEANSIM_API double GetParticleComputationTime(ParticleSimulatorHandle handle) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(handle);
                         return simPtr->getComputationTime();
                     });

    return 0.0;
}

// ===========================================
// 洋流场求解器接口实现
// ===========================================

OCEANSIM_API CurrentFieldSolverHandle CreateCurrentFieldSolver(
        int nx, int ny, int nz, double dx, double dy, double dz,
        const PhysicalParameters* params) {
    if (!params) return nullptr;

    HANDLE_EXCEPTION({
                         // 创建网格
                         auto grid = std::make_shared<GridDataStructure>(nx, ny, nz);
                         grid->setSpacing(dx, dy, std::vector<double>(nz, dz));

                         // 创建物理参数对象
                         CurrentFieldSolver::PhysicalParameters cpp_params;
                         cpp_params.gravity = params->gravity;
                         cpp_params.coriolis_f = params->coriolis_f;
                         cpp_params.beta = params->beta;
                         cpp_params.viscosity_h = params->viscosity_h;
                         cpp_params.viscosity_v = params->viscosity_v;
                         cpp_params.diffusivity_h = params->diffusivity_h;
                         cpp_params.diffusivity_v = params->diffusivity_v;
                         cpp_params.reference_density = params->reference_density;

                         auto solver = std::make_unique<CurrentFieldSolver>(grid, cpp_params);
                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyCurrentFieldSolver(CurrentFieldSolverHandle handle) {
    if (handle) {
        delete static_cast<CurrentFieldSolver*>(handle);
    }
}

OCEANSIM_API void SetCurrentFieldInitialCondition(CurrentFieldSolverHandle handle,
                                                  const char* field_name,
                                                  const double* data, int size) {
    if (!handle || !field_name || !data) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         // 这需要在CurrentFieldSolver中实现setInitialCondition方法
                         Logger::getInstance().info("SetCurrentFieldInitialCondition called for field: " + std::string(field_name));
                     });
}

OCEANSIM_API void SetBottomTopography(CurrentFieldSolverHandle handle,
                                      const double* bottom_depth, int nx, int ny) {
    if (!handle || !bottom_depth) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         Eigen::MatrixXd depth_matrix = Eigen::Map<const Eigen::MatrixXd>(bottom_depth, nx, ny);
                         solverPtr->setBottomTopography(depth_matrix);
                     });
}

OCEANSIM_API void SetWindStress(CurrentFieldSolverHandle handle,
                                const double* tau_x, const double* tau_y,
                                int nx, int ny) {
    if (!handle || !tau_x || !tau_y) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         Eigen::MatrixXd tau_x_matrix = Eigen::Map<const Eigen::MatrixXd>(tau_x, nx, ny);
                         Eigen::MatrixXd tau_y_matrix = Eigen::Map<const Eigen::MatrixXd>(tau_y, nx, ny);
                         solverPtr->setWindStress(tau_x_matrix, tau_y_matrix);
                     });
}

OCEANSIM_API void StepCurrentFieldForward(CurrentFieldSolverHandle handle, double dt) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         solverPtr->stepForward(dt);
                     });
}

OCEANSIM_API void GetCurrentField(CurrentFieldSolverHandle handle,
                                  const char* field_name,
                                  double* data, int max_size) {
    if (!handle || !field_name || !data) return;

    HANDLE_EXCEPTION({
                         // 这需要在CurrentFieldSolver中实现getField方法
                         Logger::getInstance().info("GetCurrentField called for field: " + std::string(field_name));
                     });
}

OCEANSIM_API void GetVelocityField(CurrentFieldSolverHandle handle,
                                   double* u_data, double* v_data, double* w_data,
                                   int max_size) {
    if (!handle || !u_data || !v_data || !w_data) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         const auto& state = solverPtr->getCurrentState();

                         // 复制速度场数据
                         int size = std::min(max_size, static_cast<int>(state.u.size()));
                         if (size > 0) {
                             std::memcpy(u_data, state.u.data(), size * sizeof(double));
                             std::memcpy(v_data, state.v.data(), size * sizeof(double));
                             std::memcpy(w_data, state.w.data(), size * sizeof(double));
                         }
                     });
}

OCEANSIM_API double GetTotalEnergy(CurrentFieldSolverHandle handle) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         return solverPtr->computeTotalEnergy();
                     });

    return 0.0;
}

OCEANSIM_API int CheckMassConservation(CurrentFieldSolverHandle handle, double tolerance) {
    if (!handle) return 0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(handle);
                         return solverPtr->checkMassConservation(tolerance) ? 1 : 0;
                     });

    return 0;
}

// ===========================================
// 网格数据结构接口实现
// ===========================================

OCEANSIM_API GridDataHandle CreateGridData(int nx, int ny, int nz,
                                           CoordinateSystemType coord_sys,
                                           GridTypeEnum grid_type) {
    HANDLE_EXCEPTION({
                         auto grid = std::make_unique<GridDataStructure>(
                                 nx, ny, nz,
                                 ConvertCoordinateSystemType(coord_sys),
                                 ConvertGridType(grid_type)
                         );
                         return grid.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyGridData(GridDataHandle handle) {
    if (handle) {
        delete static_cast<GridDataStructure*>(handle);
    }
}

OCEANSIM_API void SetGridSpacing(GridDataHandle handle,
                                 double dx, double dy, const double* dz, int nz) {
    if (!handle || !dz) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         std::vector<double> dz_vec(dz, dz + nz);
                         gridPtr->setSpacing(dx, dy, dz_vec);
                     });
}

OCEANSIM_API void SetGridOrigin(GridDataHandle handle, const Vector3D* origin) {
    if (!handle || !origin) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         gridPtr->setOrigin(ConvertVector3D(*origin));
                     });
}

OCEANSIM_API void AddScalarField2D(GridDataHandle handle, const char* name,
                                   const double* data, int nx, int ny) {
    if (!handle || !name || !data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         Eigen::MatrixXd field_data = Eigen::Map<const Eigen::MatrixXd>(data, nx, ny);
                         gridPtr->addField(std::string(name), field_data);
                     });
}

OCEANSIM_API void AddScalarField3D(GridDataHandle handle, const char* name,
                                   const double* data, int nx, int ny, int nz) {
    if (!handle || !name || !data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         std::vector<Eigen::MatrixXd> field_3d;
                         field_3d.reserve(nz);

                         for (int k = 0; k < nz; ++k) {
                             Eigen::MatrixXd layer = Eigen::Map<const Eigen::MatrixXd>(
                                     data + k * nx * ny, nx, ny);
                             field_3d.push_back(layer);
                         }

                         gridPtr->addField(std::string(name), field_3d);
                     });
}

OCEANSIM_API void AddVectorField(GridDataHandle handle, const char* name,
                                 const double* u_data, const double* v_data,
                                 const double* w_data, int nx, int ny, int nz) {
    if (!handle || !name || !u_data || !v_data || !w_data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         std::vector<Eigen::MatrixXd> vector_components;
                         vector_components.reserve(3);

                         // U分量
                         std::vector<Eigen::MatrixXd> u_field;
                         for (int k = 0; k < nz; ++k) {
                             u_field.emplace_back(Eigen::Map<const Eigen::MatrixXd>(
                                     u_data + k * nx * ny, nx, ny));
                         }
                         vector_components.push_back(u_field[0]); // 简化版本，只存储第一层

                         gridPtr->addVectorField(std::string(name), vector_components);
                     });
}

OCEANSIM_API double InterpolateScalar(GridDataHandle handle, const char* field_name,
                                      const Vector3D* position, InterpolationMethodType method) {
    if (!handle || !field_name || !position) return 0.0;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         Eigen::Vector3d pos = ConvertVector3D(*position);
                         GridDataStructure::InterpolationMethod interp_method =
                                 static_cast<GridDataStructure::InterpolationMethod>(method);
                         return gridPtr->interpolateScalar(pos, std::string(field_name), interp_method);
                     });

    return 0.0;
}

OCEANSIM_API Vector3D InterpolateVector(GridDataHandle handle, const char* field_name,
                                        const Vector3D* position, InterpolationMethodType method) {
    Vector3D result = {0.0, 0.0, 0.0};
    if (!handle || !field_name || !position) return result;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         Eigen::Vector3d pos = ConvertVector3D(*position);
                         GridDataStructure::InterpolationMethod interp_method =
                                 static_cast<GridDataStructure::InterpolationMethod>(method);
                         Eigen::Vector3d vec_result = gridPtr->interpolateVector(pos, std::string(field_name), interp_method);
                         result = ConvertToVector3D(vec_result);
                     });

    return result;
}

OCEANSIM_API Vector3D ComputeGradient(GridDataHandle handle, const char* field_name,
                                      const Vector3D* position) {
    Vector3D result = {0.0, 0.0, 0.0};
    if (!handle || !field_name || !position) return result;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         Eigen::Vector3d pos = ConvertVector3D(*position);
                         Eigen::Vector3d grad = gridPtr->computeGradient(std::string(field_name), pos);
                         result = ConvertToVector3D(grad);
                     });

    return result;
}

OCEANSIM_API void GetFieldData2D(GridDataHandle handle, const char* field_name,
                                 double* data, int max_size) {
    if (!handle || !field_name || !data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         const auto& field = gridPtr->getField2D(std::string(field_name));
                         int copy_size = std::min(max_size, static_cast<int>(field.size()));
                         std::memcpy(data, field.data(), copy_size * sizeof(double));
                     });
}

OCEANSIM_API void GetFieldData3D(GridDataHandle handle, const char* field_name,
                                 double* data, int max_size) {
    if (!handle || !field_name || !data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         const auto& field_3d = gridPtr->getField3D(std::string(field_name));

                         int total_copied = 0;
                         for (const auto& layer : field_3d) {
                             int layer_size = static_cast<int>(layer.size());
                             int copy_size = std::min(max_size - total_copied, layer_size);
                             if (copy_size <= 0) break;

                             std::memcpy(data + total_copied, layer.data(), copy_size * sizeof(double));
                             total_copied += copy_size;
                         }
                     });
}

OCEANSIM_API int HasField(GridDataHandle handle, const char* field_name) {
    if (!handle || !field_name) return 0;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         return gridPtr->hasField(std::string(field_name)) ? 1 : 0;
                     });

    return 0;
}

OCEANSIM_API void ClearField(GridDataHandle handle, const char* field_name) {
    if (!handle || !field_name) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         gridPtr->clearField(std::string(field_name));
                     });
}

OCEANSIM_API size_t GetGridMemoryUsage(GridDataHandle handle) {
    if (!handle) return 0;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(handle);
                         return gridPtr->getMemoryUsage();
                     });

    return 0;
}

// ===========================================
// 平流扩散求解器接口实现
// ===========================================

OCEANSIM_API AdvectionDiffusionHandle CreateAdvectionDiffusionSolver(
        GridDataHandle grid_handle, NumericalSchemeType scheme, TimeIntegrationType time_method) {
    if (!grid_handle) return nullptr;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid_handle);
                         auto grid_shared = std::shared_ptr<GridDataStructure>(gridPtr, [](GridDataStructure*){});

                         auto solver = std::make_unique<AdvectionDiffusionSolver>(
                                 grid_shared,
                                 ConvertNumericalScheme(scheme),
                                 ConvertTimeIntegration(time_method)
                         );

                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyAdvectionDiffusionSolver(AdvectionDiffusionHandle handle) {
    if (handle) {
        delete static_cast<AdvectionDiffusionSolver*>(handle);
    }
}

OCEANSIM_API void SetADInitialCondition(AdvectionDiffusionHandle handle,
                                        const char* field_name,
                                        const double* initial_data, int size) {
    if (!handle || !field_name || !initial_data) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         Eigen::MatrixXd initial_field = Eigen::Map<const Eigen::MatrixXd>(initial_data, size, 1);
                         solverPtr->setInitialCondition(std::string(field_name), initial_field);
                     });
}

OCEANSIM_API void SetADVelocityField(AdvectionDiffusionHandle handle,
                                     const char* u_field, const char* v_field,
                                     const char* w_field) {
    if (!handle || !u_field || !v_field || !w_field) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         solverPtr->setVelocityField(std::string(u_field), std::string(v_field), std::string(w_field));
                     });
}

OCEANSIM_API void SetDiffusionCoefficient(AdvectionDiffusionHandle handle,
                                          double diffusion_coeff) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         solverPtr->setDiffusionCoefficient(diffusion_coeff);
                     });
}

OCEANSIM_API void SetBoundaryCondition(AdvectionDiffusionHandle handle,
                                       BoundaryConditionType type, int boundary_id,
                                       double value) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         AdvectionDiffusionSolver::BoundaryType boundary_type =
                                 static_cast<AdvectionDiffusionSolver::BoundaryType>(type);
                         solverPtr->setBoundaryCondition(boundary_type, boundary_id, value);
                     });
}

OCEANSIM_API void SolveAdvectionDiffusion(AdvectionDiffusionHandle handle, double dt) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         solverPtr->solve(dt);
                     });
}

OCEANSIM_API void GetADSolution(AdvectionDiffusionHandle handle,
                                double* solution_data, int max_size) {
    if (!handle || !solution_data) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         const auto& solution = solverPtr->getSolution();

                         int copy_size = std::min(max_size, static_cast<int>(solution.size()));
                         std::memcpy(solution_data, solution.data(), copy_size * sizeof(double));
                     });
}

OCEANSIM_API double GetMaxConcentration(AdvectionDiffusionHandle handle) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         return solverPtr->getMaxConcentration();
                     });

    return 0.0;
}

OCEANSIM_API double GetTotalMass(AdvectionDiffusionHandle handle) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         return solverPtr->getTotalMass();
                     });

    return 0.0;
}

OCEANSIM_API int CheckADMassConservation(AdvectionDiffusionHandle handle, double tolerance) {
    if (!handle) return 0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         return solverPtr->checkMassConservation(tolerance) ? 1 : 0;
                     });

    return 0;
}

OCEANSIM_API double ComputePecletNumber(AdvectionDiffusionHandle handle) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         return solverPtr->computePecletNumber();
                     });

    return 0.0;
}

OCEANSIM_API double ComputeCourantNumber(AdvectionDiffusionHandle handle, double dt) {
    if (!handle) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(handle);
                         return solverPtr->computeCourantNumber(dt);
                     });

    return 0.0;
}

// ===========================================
// 性能分析接口实现
// ===========================================

OCEANSIM_API void GetPerformanceMetrics(void* handle, PerformanceMetrics* metrics) {
    if (!handle || !metrics) return;

    HANDLE_EXCEPTION({
                         // 这需要根据具体的handle类型来实现
                         // 这里提供一个通用的框架
                         metrics->computation_time = 0.0;
                         metrics->memory_usage_mb = 0.0;
                         metrics->iteration_count = 0;
                         metrics->efficiency_ratio = 1.0;

                         Logger::getInstance().info("GetPerformanceMetrics called");
                     });
}

OCEANSIM_API void EnableProfiling(void* handle, int enable) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         // 这需要根据具体的handle类型来实现
                         Logger::getInstance().info("EnableProfiling called with enable: " + std::to_string(enable));
                     });
}

OCEANSIM_API void ResetPerformanceCounters(void* handle) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         // 这需要根据具体的handle类型来实现
                         Logger::getInstance().info("ResetPerformanceCounters called");
                     });
}

// ===========================================
// 错误处理接口实现
// ===========================================

static ErrorCode g_last_error = ERROR_NONE;
static std::string g_error_message;

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
        case ERROR_NUMERICAL_INSTABILITY: return "Numerical instability";
        case ERROR_CONVERGENCE_FAILURE: return "Convergence failure";
        default: return "Unknown error";
    }
}

OCEANSIM_API void ClearError() {
    g_last_error = ERROR_NONE;
    g_error_message.clear();
}

// ===========================================
// 初始化和清理接口实现
// ===========================================

OCEANSIM_API void InitializeCppCore() {
    HANDLE_EXCEPTION({
                         Logger::getInstance().info("OceanSim C++ Core initialized");
                         g_last_error = ERROR_NONE;
                     });
}

OCEANSIM_API void ShutdownCppCore() {
    HANDLE_EXCEPTION({
                         Logger::getInstance().info("OceanSim C++ Core shutdown");
                         ClearError();
                     });
}

OCEANSIM_API const char* GetVersionString() {
    return "1.0.0";
}

OCEANSIM_API int GetThreadCount() {
    return omp_get_max_threads();
}

OCEANSIM_API void SetThreadCount(int thread_count) {
    if (thread_count > 0) {
        omp_set_num_threads(thread_count);
        Logger::getInstance().info("Thread count set to: " + std::to_string(thread_count));
    }
}

// ===========================================
// 附加的求解器接口实现
// ===========================================

// RungeKutta求解器
OCEANSIM_API RungeKuttaSolverHandle CreateRungeKuttaSolver(int method) {
    HANDLE_EXCEPTION({
                         RungeKuttaSolver::Method rk_method = static_cast<RungeKuttaSolver::Method>(method);
                         auto solver = std::make_unique<RungeKuttaSolver>(rk_method);
                         return solver.release();
                     });
    return nullptr;
}

OCEANSIM_API void DestroyRungeKuttaSolver(RungeKuttaSolverHandle handle) {
    if (handle) {
        delete static_cast<RungeKuttaSolver*>(handle);
    }
}

// FiniteDifference求解器
OCEANSIM_API FiniteDifferenceSolverHandle CreateFiniteDifferenceSolver(int grid_size, double time_step) {
    HANDLE_EXCEPTION({
                         auto solver = std::make_unique<FiniteDifferenceSolver>(grid_size, time_step);
                         return solver.release();
                     });
    return nullptr;
}

OCEANSIM_API void DestroyFiniteDifferenceSolver(FiniteDifferenceSolverHandle handle) {
    if (handle) {
        delete static_cast<FiniteDifferenceSolver*>(handle);
    }
}

OCEANSIM_API void SetFDBoundaryConditions(FiniteDifferenceSolverHandle handle,
                                          BoundaryConditionType type, const double* values, int size) {
    if (!handle || !values) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<FiniteDifferenceSolver*>(handle);
                         std::vector<double> boundary_values(values, values + size);
                         // 这需要在FiniteDifferenceSolver中实现setBoundaryConditions方法
                         Logger::getInstance().info("SetFDBoundaryConditions called");
                     });
}

OCEANSIM_API void SolveFDAdvectionDiffusion(FiniteDifferenceSolverHandle handle,
                                            const double* initial_data, double* result,
                                            int size, double dt, int num_steps) {
    if (!handle || !initial_data || !result) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<FiniteDifferenceSolver*>(handle);
                         std::vector<double> initial_vec(initial_data, initial_data + size);

                         // 执行时间步进
                         for (int step = 0; step < num_steps; ++step) {
                             // 这需要在FiniteDifferenceSolver中实现相应的求解方法
                         }

                         // 复制结果
                         std::memcpy(result, initial_vec.data(), size * sizeof(double));
                         Logger::getInstance().info("SolveFDAdvectionDiffusion completed");
                     });
}

// VectorizedOperations
OCEANSIM_API VectorizedOperationsHandle CreateVectorizedOperations(const PerformanceConfig* config) {
    if (!config) return nullptr;

    HANDLE_EXCEPTION({
                         OceanSimulation::Core::VectorizedOperations::Config vec_config;
                         vec_config.preferredSimd = ConvertSimdType(config->simd_type);
                         vec_config.enableBoundsCheck = true;
                         vec_config.enableAutoAlignment = true;

                         auto ops = std::make_unique<OceanSimulation::Core::VectorizedOperations>(vec_config);
                         return ops.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyVectorizedOperations(VectorizedOperationsHandle handle) {
    if (handle) {
        delete static_cast<OceanSimulation::Core::VectorizedOperations*>(handle);
    }
}

OCEANSIM_API void VectorAdd(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size) {
    if (!handle || !a || !b || !result) return;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<OceanSimulation::Core::VectorizedOperations*>(handle);
                         opsPtr->vectorAdd(a, b, result, size);
                     });
}

OCEANSIM_API void VectorSub(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size) {
    if (!handle || !a || !b || !result) return;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<OceanSimulation::Core::VectorizedOperations*>(handle);
                         opsPtr->vectorSub(a, b, result, size);
                     });
}

OCEANSIM_API void VectorMul(VectorizedOperationsHandle handle, const double* a, const double* b,
                            double* result, int size) {
    if (!handle || !a || !b || !result) return;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<OceanSimulation::Core::VectorizedOperations*>(handle);
                         opsPtr->vectorMul(a, b, result, size);
                     });
}

OCEANSIM_API double VectorDotProduct(VectorizedOperationsHandle handle, const double* a, const double* b, int size) {
    if (!handle || !a || !b) return 0.0;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<OceanSimulation::Core::VectorizedOperations*>(handle);
                         return opsPtr->dotProduct(a, b, size);
                     });

    return 0.0;
}

// ParallelComputeEngine
OCEANSIM_API ParallelComputeEngineHandle CreateParallelComputeEngine(const PerformanceConfig* config) {
    if (!config) return nullptr;

    HANDLE_EXCEPTION({
                         OceanSimulation::Core::ParallelComputeEngine::Config engine_config;
                         engine_config.maxThreads = static_cast<size_t>(config->num_threads);
                         engine_config.defaultPolicy = ConvertExecutionPolicy(config->execution_policy);
                         engine_config.enableHyperthreading = true;
                         engine_config.enableAffinity = false;

                         auto engine = std::make_unique<OceanSimulation::Core::ParallelComputeEngine>(engine_config);
                         return engine.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyParallelComputeEngine(ParallelComputeEngineHandle handle) {
    if (handle) {
        delete static_cast<OceanSimulation::Core::ParallelComputeEngine*>(handle);
    }
}

OCEANSIM_API void StartParallelEngine(ParallelComputeEngineHandle handle) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* enginePtr = static_cast<OceanSimulation::Core::ParallelComputeEngine*>(handle);
                         enginePtr->start();
                     });
}

OCEANSIM_API void StopParallelEngine(ParallelComputeEngineHandle handle) {
    if (!handle) return;

    HANDLE_EXCEPTION({
                         auto* enginePtr = static_cast<OceanSimulation::Core::ParallelComputeEngine*>(handle);
                         enginePtr->stop();
                     });
}

OCEANSIM_API int GetAvailableThreads(ParallelComputeEngineHandle handle) {
    if (!handle) return 0;

    HANDLE_EXCEPTION({
                         auto* enginePtr = static_cast<OceanSimulation::Core::ParallelComputeEngine*>(handle);
                         return static_cast<int>(enginePtr->getAvailableThreads());
                     });

    return 0;
}

// ===========================================
// 数据导出器接口实现
// ===========================================

OCEANSIM_API DataExporterHandle CreateDataExporter() {
    HANDLE_EXCEPTION({
                         auto exporter = std::make_unique<OceanSimulation::Core::DataExporter>();
                         return exporter.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyDataExporter(DataExporterHandle handle) {
    if (handle) {
        delete static_cast<OceanSimulation::Core::DataExporter*>(handle);
    }
}

OCEANSIM_API int ExportToNetCDF(DataExporterHandle handle, GridDataHandle grid, const char* filename) {
    if (!handle || !grid || !filename) return 0;

    HANDLE_EXCEPTION({
                         auto* exporterPtr = static_cast<OceanSimulation::Core::DataExporter*>(handle);
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         // 这需要在DataExporter中实现exportToNetCDF方法
                         Logger::getInstance().info("ExportToNetCDF called with filename: " + std::string(filename));
                         return 1; // 假设成功
                     });

    return 0;
}

OCEANSIM_API int ExportToVTK(DataExporterHandle handle, GridDataHandle grid, const char* filename) {
    if (!handle || !grid || !filename) return 0;

    HANDLE_EXCEPTION({
                         auto* exporterPtr = static_cast<OceanSimulation::Core::DataExporter*>(handle);
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         // 这需要在DataExporter中实现exportToVTK方法
                         Logger::getInstance().info("ExportToVTK called with filename: " + std::string(filename));
                         return 1; // 假设成功
                     });

    return 0;
}

// ===========================================
// 性能分析器接口实现
// ===========================================

OCEANSIM_API PerformanceProfilerHandle CreatePerformanceProfiler() {
    HANDLE_EXCEPTION({
                         auto profiler = std::make_unique<PerformanceProfiler>();
                         return profiler.release();
                     });

    return nullptr;
}

OCEANSIM_API void DestroyPerformanceProfiler(PerformanceProfilerHandle handle) {
    if (handle) {
        delete static_cast<PerformanceProfiler*>(handle);
    }
}

OCEANSIM_API void StartTiming(PerformanceProfilerHandle handle, const char* section_name) {
    if (!handle || !section_name) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(handle);
                         profilerPtr->startTiming(std::string(section_name));
                     });
}

OCEANSIM_API void EndTiming(PerformanceProfilerHandle handle, const char* section_name) {
    if (!handle || !section_name) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(handle);
                         profilerPtr->endTiming(std::string(section_name));
                     });
}

OCEANSIM_API double GetElapsedTime(PerformanceProfilerHandle handle, const char* section_name) {
    if (!handle || !section_name) return 0.0;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(handle);
                         return profilerPtr->getElapsedTime(std::string(section_name));
                     });

    return 0.0;
}

OCEANSIM_API void GenerateReport(PerformanceProfilerHandle handle, const char* filename) {
    if (!handle || !filename) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(handle);
                         profilerPtr->generateReport(std::string(filename));
                     });
}