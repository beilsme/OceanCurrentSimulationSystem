import pytest

@pytest.fixture
def oceansim():
    import oceansim
    return oceansim


def test_module_import(oceansim):
    """
    模块导入测试
    """
    assert oceansim is not None


def test_physical_parameters(oceansim):
    """
    仅测试物理参数结构体
    """
    params = oceansim.PhysicalParameters()
    assert params is not None


def test_particle_class(oceansim):
    """
    测试Particle类
    """
    p = oceansim.Particle()
    # 由于Eigen绑定，必须用Eigen类型，暂时只检查可创建
    assert p is not None
