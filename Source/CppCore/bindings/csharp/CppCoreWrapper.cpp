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

using namespace OceanSim;

// ===========================================
// 内部辅助函数
// ===========================================

namespace {
    // 将C结构体转换为C++对象
    Vector3D ConvertVector3D(const Vector3D& vec) {
        return Vector3D(vec.x, vec.y, vec.z);
    }

    Vector3D ConvertToVector3D(const Vector3D& vec) {
        Vector3D result;
        result.x = vec.x();
        result.y = vec.y();
        result.z = vec.z();
        return result;
    }

    CoordinateSystem ConvertCoordinateSystem(int coord_system) {
        switch (coord_system) {
            case COORDINATE_CARTESIAN: return CoordinateSystem::CARTESIAN;
            case COORDINATE_SPHERICAL: return CoordinateSystem::SPHERICAL;
            case COORDINATE_HYBRID_SIGMA: return CoordinateSystem::HYBRID_SIGMA;
            case COORDINATE_ISOPYCNAL: return CoordinateSystem::ISOPYCNAL;
            default: return CoordinateSystem::CARTESIAN;
        }
    }

    GridType ConvertGridType(int grid_type) {
        switch (grid_type) {
            case GRID_REGULAR: return GridType::REGULAR;
            case GRID_CURVILINEAR: return GridType::CURVILINEAR;
            case GRID_UNSTRUCTURED: return GridType::UNSTRUCTURED;
            default: return GridType::REGULAR;
        }
    }

    NumericalScheme ConvertNumericalScheme(int scheme) {
        switch (scheme) {
            case SCHEME_UPWIND: return NumericalScheme::UPWIND;
            case SCHEME_CENTRAL: return NumericalScheme::CENTRAL;
            case SCHEME_TVD_SUPERBEE: return NumericalScheme::TVD_SUPERBEE;
            case SCHEME_WENO: return NumericalScheme::WENO;
            default: return NumericalScheme::TVD_SUPERBEE;
        }
    }

    TimeIntegration ConvertTimeIntegration(int method) {
        switch (method) {
            case TIME_EULER: return TimeIntegration::EULER;
            case TIME_RUNGE_KUTTA_2: return TimeIntegration::RUNGE_KUTTA_2;
            case TIME_RUNGE_KUTTA_3: return TimeIntegration::RUNGE_KUTTA_3;
            case TIME_RUNGE_KUTTA_4: return TimeIntegration::RUNGE_KUTTA_4;
            case TIME_ADAMS_BASHFORTH: return TimeIntegration::ADAMS_BASHFORTH;
            default: return TimeIntegration::RUNGE_KUTTA_4;
        }
    }

    ExecutionPolicy ConvertExecutionPolicy(int policy) {
        switch (policy) {
            case EXECUTION_SEQUENTIAL: return ExecutionPolicy::Sequential;
            case EXECUTION_PARALLEL: return ExecutionPolicy::Parallel;
            case EXECUTION_VECTORIZED: return ExecutionPolicy::Vectorized;
            case EXECUTION_HYBRID_PARALLEL: return ExecutionPolicy::HybridParallel;
            default: return ExecutionPolicy::Parallel;
        }
    }

    SimdType ConvertSimdType(int simd) {
        switch (simd) {
            case SIMD_NONE: return SimdType::None;
            case SIMD_SSE: return SimdType::SSE;
            case SIMD_AVX: return SimdType::AVX;
            case SIMD_AVX2: return SimdType::AVX2;
            case SIMD_AVX512: return SimdType::AVX512;
            case SIMD_NEON: return SimdType::NEON;
            default: return SimdType::AVX2;
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
// 网格数据结构接口实现
// ===========================================

OCEANSIM_API GridHandle Grid_Create(const GridConfig* config) {
    if (!config) return nullptr;

    HANDLE_EXCEPTION({
                         auto grid = std::make_unique<GridDataStructure>(
                                 config->nx, config->ny, config->nz,
                                 config->dx, config->dy, config->dz,
                                 ConvertVector3D(config->origin),
                                 ConvertCoordinateSystem(config->coordinate_system),
                                 ConvertGridType(config->grid_type)
                         );
                         return grid.release();
                     });

    return nullptr;
}

OCEANSIM_API void Grid_Destroy(GridHandle grid) {
    if (grid) {
        delete static_cast<GridDataStructure*>(grid);
    }
}

OCEANSIM_API void Grid_GetDimensions(GridHandle grid, int* nx, int* ny, int* nz) {
    if (!grid || !nx || !ny || !nz) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         auto dims = gridPtr->getDimensions();
                         *nx = dims[0];
                         *ny = dims[1];
                         *nz = dims[2];
                     });
}

OCEANSIM_API void Grid_SetScalarField(GridHandle grid, const double* data, int size, const char* field_name) {
    if (!grid || !data || !field_name) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         std::vector<double> field_data(data, data + size);
                         gridPtr->setScalarField(std::string(field_name), field_data);
                     });
}

OCEANSIM_API void Grid_SetVectorField(GridHandle grid, const double* u_data, const double* v_data,
                                      const double* w_data, int size, const char* field_name) {
    if (!grid || !u_data || !v_data || !w_data || !field_name) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         std::vector<Vector3D> vector_field;
                         vector_field.reserve(size);

                         for (int i = 0; i < size; ++i) {
                             vector_field.emplace_back(u_data[i], v_data[i], w_data[i]);
                         }

                         gridPtr->setVectorField(std::string(field_name), vector_field);
                     });
}

OCEANSIM_API void Grid_GetScalarField(GridHandle grid, const char* field_name, double* data, int size) {
    if (!grid || !field_name || !data) return;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         auto field_data = gridPtr->getScalarField(std::string(field_name));

                         int copy_size = std::min(size, static_cast<int>(field_data.size()));
                         std::memcpy(data, field_data.data(), copy_size * sizeof(double));
                     });
}

OCEANSIM_API double Grid_Interpolate(GridHandle grid, const Vector3D* position,
                                     const char* field_name, int method) {
    if (!grid || !position || !field_name) return 0.0;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         Vector3D pos = ConvertVector3D(*position);
                         InterpolationMethod interp_method = static_cast<InterpolationMethod>(method);
                         return gridPtr->interpolate(std::string(field_name), pos, interp_method);
                     });

    return 0.0;
}

// ===========================================
// 粒子模拟器接口实现
// ===========================================

OCEANSIM_API ParticleSimulatorHandle ParticleSim_Create(GridHandle grid, RungeKuttaSolverHandle solver_handle) {
    if (!grid || !solver_handle) return nullptr;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);
                         auto* solverPtr = static_cast<RungeKuttaSolver*>(solver_handle);

                         auto simulator = std::make_unique<ParticleSimulator>(*gridPtr, *solverPtr);
                         return simulator.release();
                     });

    return nullptr;
}

OCEANSIM_API void ParticleSim_Destroy(ParticleSimulatorHandle simulator) {
    if (simulator) {
        delete static_cast<ParticleSimulator*>(simulator);
    }
}

OCEANSIM_API void ParticleSim_InitializeParticles(ParticleSimulatorHandle simulator,
                                                  const Vector3D* positions, int count) {
    if (!simulator || !positions) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(simulator);
                         std::vector<Vector3D> init_positions;
                         init_positions.reserve(count);

                         for (int i = 0; i < count; ++i) {
                             init_positions.push_back(ConvertVector3D(positions[i]));
                         }

                         simPtr->initializeParticles(init_positions);
                     });
}

OCEANSIM_API void ParticleSim_StepForward(ParticleSimulatorHandle simulator, double time_step) {
    if (!simulator) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(simulator);
                         simPtr->stepForward(time_step);
                     });
}

OCEANSIM_API void ParticleSim_GetParticles(ParticleSimulatorHandle simulator, ParticleData* particles, int count) {
    if (!simulator || !particles) return;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(simulator);
                         auto particle_list = simPtr->getParticles();

                         int copy_count = std::min(count, static_cast<int>(particle_list.size()));
                         for (int i = 0; i < copy_count; ++i) {
                             const auto& p = particle_list[i];
                             particles[i].id = p.getId();
                             particles[i].position = ConvertToVector3D(p.getPosition());
                             particles[i].velocity = ConvertToVector3D(p.getVelocity());
                             particles[i].age = p.getAge();
                             particles[i].active = p.isActive() ? 1 : 0;
                         }
                     });
}

OCEANSIM_API int ParticleSim_GetParticleCount(ParticleSimulatorHandle simulator) {
    if (!simulator) return 0;

    HANDLE_EXCEPTION({
                         auto* simPtr = static_cast<ParticleSimulator*>(simulator);
                         return static_cast<int>(simPtr->getParticles().size());
                     });

    return 0;
}

// ===========================================
// 洋流场求解器接口实现
// ===========================================

OCEANSIM_API CurrentFieldSolverHandle CurrentSolver_Create(GridHandle grid, const PhysicalParameters* params) {
    if (!grid || !params) return nullptr;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         // 创建物理参数对象
                         PhysicalParameters cpp_params;
                         cpp_params.density = params->density;
                         cpp_params.viscosity = params->viscosity;
                         cpp_params.gravity = params->gravity;
                         cpp_params.coriolis_param = params->coriolis_param;
                         cpp_params.wind_stress_x = params->wind_stress_x;
                         cpp_params.wind_stress_y = params->wind_stress_y;

                         auto solver = std::make_unique<CurrentFieldSolver>(*gridPtr, cpp_params);
                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void CurrentSolver_Destroy(CurrentFieldSolverHandle solver) {
    if (solver) {
        delete static_cast<CurrentFieldSolver*>(solver);
    }
}

OCEANSIM_API void CurrentSolver_ComputeVelocityField(CurrentFieldSolverHandle solver, double time_step) {
    if (!solver) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(solver);
                         solverPtr->computeVelocityField(time_step);
                     });
}

OCEANSIM_API void CurrentSolver_SetBoundaryConditions(CurrentFieldSolverHandle solver, int boundary_type,
                                                      const double* values, int size) {
    if (!solver || !values) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(solver);
                         std::vector<double> boundary_values(values, values + size);
                         // Note: This would need to be implemented in the actual CurrentFieldSolver class
                         // solverPtr->setBoundaryConditions(static_cast<BoundaryType>(boundary_type), boundary_values);
                     });
}

OCEANSIM_API double CurrentSolver_ComputeKineticEnergy(CurrentFieldSolverHandle solver) {
    if (!solver) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(solver);
                         return solverPtr->computeKineticEnergy();
                     });

    return 0.0;
}

OCEANSIM_API double CurrentSolver_CheckMassConservation(CurrentFieldSolverHandle solver) {
    if (!solver) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<CurrentFieldSolver*>(solver);
                         return solverPtr->checkMassConservation();
                     });

    return 0.0;
}

// ===========================================
// 平流扩散求解器接口实现
// ===========================================

OCEANSIM_API AdvectionDiffusionSolverHandle AdvectionSolver_Create(GridHandle grid, const SolverParameters* params) {
    if (!grid || !params) return nullptr;

    HANDLE_EXCEPTION({
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         auto solver = std::make_unique<AdvectionDiffusionSolver>(
                                 *gridPtr,
                                 ConvertNumericalScheme(params->scheme_type),
                                 ConvertTimeIntegration(params->integration_method)
                         );

                         solver->setDiffusionCoefficient(params->diffusion_coeff);

                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void AdvectionSolver_Destroy(AdvectionDiffusionSolverHandle solver) {
    if (solver) {
        delete static_cast<AdvectionDiffusionSolver*>(solver);
    }
}

OCEANSIM_API void AdvectionSolver_SetInitialCondition(AdvectionDiffusionSolverHandle solver,
                                                      const double* initial_field, int size) {
    if (!solver || !initial_field) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(solver);
                         std::vector<double> initial_data(initial_field, initial_field + size);
                         solverPtr->setInitialCondition(initial_data);
                     });
}

OCEANSIM_API void AdvectionSolver_SetVelocityField(AdvectionDiffusionSolverHandle solver,
                                                   const double* u_field, const double* v_field,
                                                   const double* w_field, int size) {
    if (!solver || !u_field || !v_field || !w_field) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(solver);
                         std::vector<Vector3D> velocity_field;
                         velocity_field.reserve(size);

                         for (int i = 0; i < size; ++i) {
                             velocity_field.emplace_back(u_field[i], v_field[i], w_field[i]);
                         }

                         solverPtr->setVelocityField(velocity_field);
                     });
}

OCEANSIM_API void AdvectionSolver_Solve(AdvectionDiffusionSolverHandle solver, double time_end,
                                        double* output_field, int size) {
    if (!solver || !output_field) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<AdvectionDiffusionSolver*>(solver);
                         auto result = solverPtr->solve(time_end);

                         int copy_size = std::min(size, static_cast<int>(result.size()));
                         std::memcpy(output_field, result.data(), copy_size * sizeof(double));
                     });
}

// ===========================================
// 龙格-库塔求解器接口实现
// ===========================================

OCEANSIM_API RungeKuttaSolverHandle RungeKutta_Create(int order, double time_step) {
    HANDLE_EXCEPTION({
                         auto solver = std::make_unique<RungeKuttaSolver>(order, time_step);
                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void RungeKutta_Destroy(RungeKuttaSolverHandle solver) {
    if (solver) {
        delete static_cast<RungeKuttaSolver*>(solver);
    }
}

OCEANSIM_API void RungeKutta_SetTimeStep(RungeKuttaSolverHandle solver, double time_step) {
    if (!solver) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<RungeKuttaSolver*>(solver);
                         solverPtr->setTimeStep(time_step);
                     });
}

OCEANSIM_API double RungeKutta_GetTimeStep(RungeKuttaSolverHandle solver) {
    if (!solver) return 0.0;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<RungeKuttaSolver*>(solver);
                         return solverPtr->getTimeStep();
                     });

    return 0.0;
}

// ===========================================
// 有限差分求解器接口实现
// ===========================================

OCEANSIM_API FiniteDifferenceSolverHandle FiniteDiff_Create(int grid_size, double spacing) {
    HANDLE_EXCEPTION({
                         auto solver = std::make_unique<FiniteDifferenceSolver>(grid_size, spacing);
                         return solver.release();
                     });

    return nullptr;
}

OCEANSIM_API void FiniteDiff_Destroy(FiniteDifferenceSolverHandle solver) {
    if (solver) {
        delete static_cast<FiniteDifferenceSolver*>(solver);
    }
}

OCEANSIM_API void FiniteDiff_ComputeFirstDerivative(FiniteDifferenceSolverHandle solver,
                                                    const double* input, double* output,
                                                    int size, int direction) {
    if (!solver || !input || !output) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<FiniteDifferenceSolver*>(solver);
                         std::vector<double> input_data(input, input + size);
                         auto result = solverPtr->computeFirstDerivative(input_data, direction);

                         int copy_size = std::min(size, static_cast<int>(result.size()));
                         std::memcpy(output, result.data(), copy_size * sizeof(double));
                     });
}

OCEANSIM_API void FiniteDiff_ComputeSecondDerivative(FiniteDifferenceSolverHandle solver,
                                                     const double* input, double* output,
                                                     int size, int direction) {
    if (!solver || !input || !output) return;

    HANDLE_EXCEPTION({
                         auto* solverPtr = static_cast<FiniteDifferenceSolver*>(solver);
                         std::vector<double> input_data(input, input + size);
                         auto result = solverPtr->computeSecondDerivative(input_data, direction);

                         int copy_size = std::min(size, static_cast<int>(result.size()));
                         std::memcpy(output, result.data(), copy_size * sizeof(double));
                     });
}

// ===========================================
// 向量化运算接口实现
// ===========================================

OCEANSIM_API VectorizedOperationsHandle VectorOps_Create(const PerformanceConfig* config) {
    if (!config) return nullptr;

    HANDLE_EXCEPTION({
                         VectorConfig vec_config;
                         vec_config.execution_policy = ConvertExecutionPolicy(config->execution_policy);
                         vec_config.simd_type = ConvertSimdType(config->simd_type);
                         vec_config.num_threads = config->num_threads;

                         auto ops = std::make_unique<VectorizedOperations>(vec_config);
                         return ops.release();
                     });

    return nullptr;
}

OCEANSIM_API void VectorOps_Destroy(VectorizedOperationsHandle ops) {
    if (ops) {
        delete static_cast<VectorizedOperations*>(ops);
    }
}

OCEANSIM_API void VectorOps_Add(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size) {
    if (!ops || !a || !b || !result) return;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<VectorizedOperations*>(ops);
                         opsPtr->vectorAdd(a, b, result, size);
                     });
}

OCEANSIM_API void VectorOps_Sub(VectorizedOperationsHandle ops, const double* a, const double* b,
                                double* result, int size) {
    if (!ops || !a || !b || !result) return;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<VectorizedOperations*>(ops);
                         opsPtr->vectorSub(a, b, result, size);
                     });
}

OCEANSIM_API double VectorOps_DotProduct(VectorizedOperationsHandle ops, const double* a, const double* b, int size) {
    if (!ops || !a || !b) return 0.0;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<VectorizedOperations*>(ops);
                         return opsPtr->dotProduct(a, b, size);
                     });

    return 0.0;
}

OCEANSIM_API double VectorOps_Norm(VectorizedOperationsHandle ops, const double* a, int size) {
    if (!ops || !a) return 0.0;

    HANDLE_EXCEPTION({
                         auto* opsPtr = static_cast<VectorizedOperations*>(ops);
                         return opsPtr->vectorNorm(a, size);
                     });

    return 0.0;
}

// ===========================================
// 并行计算引擎接口实现
// ===========================================

OCEANSIM_API ParallelComputeEngineHandle ParallelEngine_Create(const PerformanceConfig* config) {
    if (!config) return nullptr;

    HANDLE_EXCEPTION({
                         EngineConfig engine_config;
                         engine_config.execution_policy = ConvertExecutionPolicy(config->execution_policy);
                         engine_config.num_threads = config->num_threads;

                         auto engine = std::make_unique<ParallelComputeEngine>(engine_config);
                         return engine.release();
                     });

    return nullptr;
}

OCEANSIM_API void ParallelEngine_Destroy(ParallelComputeEngineHandle engine) {
    if (engine) {
        delete static_cast<ParallelComputeEngine*>(engine);
    }
}

OCEANSIM_API void ParallelEngine_SetThreadCount(ParallelComputeEngineHandle engine, int num_threads) {
    if (!engine) return;

    HANDLE_EXCEPTION({
                         auto* enginePtr = static_cast<ParallelComputeEngine*>(engine);
                         enginePtr->setThreadCount(num_threads);
                     });
}

OCEANSIM_API int ParallelEngine_GetThreadCount(ParallelComputeEngineHandle engine) {
    if (!engine) return 0;

    HANDLE_EXCEPTION({
                         auto* enginePtr = static_cast<ParallelComputeEngine*>(engine);
                         return enginePtr->getThreadCount();
                     });

    return 0;
}

// ===========================================
// 数据导出器接口实现
// ===========================================

OCEANSIM_API DataExporterHandle DataExporter_Create(void) {
    HANDLE_EXCEPTION({
                         auto exporter = std::make_unique<DataExporter>();
                         return exporter.release();
                     });

    return nullptr;
}

OCEANSIM_API void DataExporter_Destroy(DataExporterHandle exporter) {
    if (exporter) {
        delete static_cast<DataExporter*>(exporter);
    }
}

OCEANSIM_API int DataExporter_ExportToNetCDF(DataExporterHandle exporter, GridHandle grid, const char* filename) {
    if (!exporter || !grid || !filename) return 0;

    HANDLE_EXCEPTION({
                         auto* exporterPtr = static_cast<DataExporter*>(exporter);
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         bool success = exporterPtr->exportToNetCDF(*gridPtr, std::string(filename));
                         return success ? 1 : 0;
                     });

    return 0;
}

OCEANSIM_API int DataExporter_ExportToVTK(DataExporterHandle exporter, GridHandle grid, const char* filename) {
    if (!exporter || !grid || !filename) return 0;

    HANDLE_EXCEPTION({
                         auto* exporterPtr = static_cast<DataExporter*>(exporter);
                         auto* gridPtr = static_cast<GridDataStructure*>(grid);

                         bool success = exporterPtr->exportToVTK(*gridPtr, std::string(filename));
                         return success ? 1 : 0;
                     });

    return 0;
}

// ===========================================
// 性能分析器接口实现
// ===========================================

OCEANSIM_API PerformanceProfilerHandle Profiler_Create(void) {
    HANDLE_EXCEPTION({
                         auto profiler = std::make_unique<PerformanceProfiler>();
                         return profiler.release();
                     });

    return nullptr;
}

OCEANSIM_API void Profiler_Destroy(PerformanceProfilerHandle profiler) {
    if (profiler) {
        delete static_cast<PerformanceProfiler*>(profiler);
    }
}

OCEANSIM_API void Profiler_StartTiming(PerformanceProfilerHandle profiler, const char* section_name) {
    if (!profiler || !section_name) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(profiler);
                         profilerPtr->startTiming(std::string(section_name));
                     });
}

OCEANSIM_API void Profiler_EndTiming(PerformanceProfilerHandle profiler, const char* section_name) {
    if (!profiler || !section_name) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(profiler);
                         profilerPtr->endTiming(std::string(section_name));
                     });
}

OCEANSIM_API double Profiler_GetElapsedTime(PerformanceProfilerHandle profiler, const char* section_name) {
    if (!profiler || !section_name) return 0.0;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(profiler);
                         return profilerPtr->getElapsedTime(std::string(section_name));
                     });

    return 0.0;
}

OCEANSIM_API void Profiler_GenerateReport(PerformanceProfilerHandle profiler, const char* filename) {
    if (!profiler || !filename) return;

    HANDLE_EXCEPTION({
                         auto* profilerPtr = static_cast<PerformanceProfiler*>(profiler);
                         profilerPtr->generateReport(std::string(filename));
                     });
}

// ===========================================
// 工具函数接口实现
// ===========================================

OCEANSIM_API const char* OceanSim_GetVersion(void) {
    return "1.0.0";
}

OCEANSIM_API int OceanSim_Initialize(void) {
    HANDLE_EXCEPTION({
                         Logger::getInstance().info("OceanSim C# Wrapper initialized");
                         return 1;
                     });

    return 0;
}

OCEANSIM_API void OceanSim_Cleanup(void) {
    HANDLE_EXCEPTION({
                         Logger::getInstance().info("OceanSim C# Wrapper cleanup");
                     });
}

OCEANSIM_API void OceanSim_SetLogLevel(int level) {
    HANDLE_EXCEPTION({
                         Logger::LogLevel log_level;
                         switch (level) {
                             case 0: log_level = Logger::LogLevel::DEBUG; break;
                             case 1: log_level = Logger::LogLevel::INFO; break;
                             case 2: log_level = Logger::LogLevel::WARNING; break;
                             case 3: log_level = Logger::LogLevel::ERROR; break;
                             default: log_level = Logger::LogLevel::INFO; break;
                         }
                         Logger::getInstance().setLevel(log_level);
                     });
}