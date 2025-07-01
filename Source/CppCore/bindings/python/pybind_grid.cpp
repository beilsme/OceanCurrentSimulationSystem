/**
 * @file pybind_grid.cpp
 * @author beilsm
 * @version 1.0
 * @brief GridDataStructure bindings
 * @date 2025-07-01
 */
#include "pybind_bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "data/GridDataStructure.h"

namespace py = pybind11;

void bind_grid(py::module_ &m)
{
    using Grid = OceanSim::Data::GridDataStructure;
    py::class_<Grid, std::shared_ptr<Grid>>(m, "GridDataStructure")
            .def(py::init<int,int,int, Grid::CoordinateSystem, Grid::GridType>(),
                 py::arg("nx"), py::arg("ny"), py::arg("nz"),
                 py::arg("coord_sys") = Grid::CoordinateSystem::CARTESIAN,
                 py::arg("grid_type") = Grid::GridType::REGULAR)
            .def("set_spacing", (void(Grid::*)(double,double,const std::vector<double>&))&Grid::setSpacing,
                 py::arg("dx"), py::arg("dy"), py::arg("dz"))
            .def("set_origin", &Grid::setOrigin)
            .def("set_bounds", &Grid::setBounds)
            .def("add_field2d", &Grid::addField)
            .def("get_field_names", &Grid::getFieldNames)
            .def("has_field", &Grid::hasField)
            .def("get_dimensions", &Grid::getDimensions)
            .def("get_spacing", &Grid::getSpacing)
            .def("transform_to_grid", &Grid::transformToGrid)
            .def("transform_to_world", &Grid::transformToWorld)
            .def("interpolate_scalar", &Grid::interpolateScalar,
                 py::arg("pos"), py::arg("field"),
                 py::arg("method") = Grid::InterpolationMethod::TRILINEAR)
            .def("interpolate_vector", &Grid::interpolateVector,
                 py::arg("pos"), py::arg("field"),
                 py::arg("method") = Grid::InterpolationMethod::TRILINEAR);

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
}