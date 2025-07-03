#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "data/GridDataStructure.h"

namespace py = pybind11;

void bind_grid(py::module_ &m)
{
    using Grid = OceanSim::Data::GridDataStructure;

    py::enum_<Grid::CoordinateSystem>(m, "CoordinateSystem")
        .value("CARTESIAN", Grid::CoordinateSystem::CARTESIAN)
        .value("SPHERICAL", Grid::CoordinateSystem::SPHERICAL)
        .value("HYBRID_SIGMA", Grid::CoordinateSystem::HYBRID_SIGMA)
        .value("ISOPYCNAL", Grid::CoordinateSystem::ISOPYCNAL);

    py::enum_<Grid::GridType>(m, "GridType")
        .value("REGULAR", Grid::GridType::REGULAR)
        .value("CURVILINEAR", Grid::GridType::CURVILINEAR)
        .value("UNSTRUCTURED", Grid::GridType::UNSTRUCTURED);

    py::enum_<Grid::InterpolationMethod>(m, "InterpolationMethod")
        .value("LINEAR", Grid::InterpolationMethod::LINEAR)
        .value("CUBIC", Grid::InterpolationMethod::CUBIC)
        .value("BILINEAR", Grid::InterpolationMethod::BILINEAR)
        .value("TRILINEAR", Grid::InterpolationMethod::TRILINEAR)
        .value("CONSERVATIVE", Grid::InterpolationMethod::CONSERVATIVE);

    py::class_<Grid, std::shared_ptr<Grid>>(m, "GridDataStructure")
        .def(py::init<int, int, int, Grid::CoordinateSystem, Grid::GridType>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("coord_system") = Grid::CoordinateSystem::CARTESIAN,
             py::arg("grid_type") = Grid::GridType::REGULAR)
        .def("get_dimensions", &Grid::getDimensions)
        .def("add_field2d",
             static_cast<void (Grid::*)(const std::string&, const Eigen::MatrixXd&)>(&Grid::addField),
             py::arg("name"), py::arg("data"))
        .def("add_field3d",
             static_cast<void (Grid::*)(const std::string&, const std::vector<Eigen::MatrixXd>&)>(&Grid::addField),
             py::arg("name"), py::arg("data"))
        .def("add_vector_field", &Grid::addVectorField,
             py::arg("name"), py::arg("components"));
}