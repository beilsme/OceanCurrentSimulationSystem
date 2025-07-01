import pytest
import importlib

oceansim = pytest.importorskip("oceansim")

def test_module_import():
    assert oceansim is not None

def test_grid_structure():
    grid = oceansim.GridDataStructure(4, 4, 1)
    assert grid.get_dimensions() == [4, 4, 1]


def test_runge_kutta_solver():
    solver = oceansim.RungeKuttaSolver(oceansim.RKMethod.RK4)
    result = solver.solve([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.1, lambda y, t: [0.0, 0.0, 0.0])
    assert len(result) == 3


def test_finite_difference_solver():
    fd = oceansim.FiniteDifferenceSolver(10, 0.1)
    assert fd.get_time_step() == 0.1


def test_vectorized_operations():
    cfg = oceansim.VectorConfig()
    vec = oceansim.VectorizedOperations(cfg)
    assert vec is not None


def test_parallel_compute_engine():
    cfg = oceansim.EngineConfig()
    engine = oceansim.ParallelComputeEngine(cfg)
    assert engine.get_available_threads() >= 1


def test_advection_solver(grid=None):
    grid = oceansim.GridDataStructure(4, 4, 1)
    solver = oceansim.AdvectionDiffusionSolver(grid, oceansim.NumericalScheme.UPWIND, oceansim.TimeIntegration.EXPLICIT_EULER)
    assert solver is not None


def test_current_field_solver():
    grid = oceansim.GridDataStructure(4, 4, 1)
    params = oceansim.PhysicalParameters()
    solver = oceansim.CurrentFieldSolver(grid, params)
    assert solver is not None

def test_particle_simulator():
    grid = oceansim.GridDataStructure(4, 4, 1)
    rk = oceansim.RungeKuttaSolver(oceansim.RKMethod.RK4)
    sim = oceansim.ParticleSimulator(grid, rk)
    sim.initialize_particles([])
    sim.step_forward(0.1)
    assert isinstance(sim.get_particles(), list)