// src/data/GridDataStructure.cpp
#include "data/GridDataStructure.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace OceanSim {
    namespace Data {

        GridDataStructure::GridDataStructure(int nx, int ny, int nz,
                                             CoordinateSystem coord_sys,
                                             GridType grid_type)
                : nx_(nx), ny_(ny), nz_(nz), coord_system_(coord_sys), grid_type_(grid_type),
                  origin_(Eigen::Vector3d::Zero()), has_transform_(false), thread_safe_(false) {

            min_bounds_ = Eigen::Vector3d::Zero();
            max_bounds_ = Eigen::Vector3d(nx-1, ny-1, nz-1);
            transform_matrix_ = Eigen::Matrix3d::Identity();

            initializeSpacing();
        }

        void GridDataStructure::initializeSpacing() {
            dx_.resize(nx_, 1.0);
            dy_.resize(ny_, 1.0);
            dz_.resize(nz_, 1.0);
            updateBounds();
        }

        void GridDataStructure::setSpacing(double dx, double dy, const std::vector<double>& dz) {
            if (dz.size() != static_cast<size_t>(nz_)) {
                throw std::invalid_argument("dz size must match nz");
            }

            std::fill(dx_.begin(), dx_.end(), dx);
            std::fill(dy_.begin(), dy_.end(), dy);
            dz_ = dz;
            updateBounds();
        }

        void GridDataStructure::setSpacing(const std::vector<double>& dx,
                                           const std::vector<double>& dy,
                                           const std::vector<double>& dz) {
            if (dx.size() != static_cast<size_t>(nx_) ||
                dy.size() != static_cast<size_t>(ny_) ||
                dz.size() != static_cast<size_t>(nz_)) {
                throw std::invalid_argument("Spacing vector sizes must match grid dimensions");
            }

            dx_ = dx;
            dy_ = dy;
            dz_ = dz;
            updateBounds();
        }

        void GridDataStructure::updateBounds() {
            double total_x = std::accumulate(dx_.begin(), dx_.end(), 0.0);
            double total_y = std::accumulate(dy_.begin(), dy_.end(), 0.0);
            double total_z = std::accumulate(dz_.begin(), dz_.end(), 0.0);

            max_bounds_ = origin_ + Eigen::Vector3d(total_x, total_y, total_z);
        }

        void GridDataStructure::setOrigin(const Eigen::Vector3d& origin) {
            origin_ = origin;
            updateBounds();
        }

        void GridDataStructure::setBounds(const Eigen::Vector3d& min_bounds,
                                          const Eigen::Vector3d& max_bounds) {
            min_bounds_ = min_bounds;
            max_bounds_ = max_bounds;
            origin_ = min_bounds;

            // 重新计算间距
            for (int i = 0; i < nx_; ++i) {
                dx_[i] = (max_bounds.x() - min_bounds.x()) / nx_;
            }
            for (int j = 0; j < ny_; ++j) {
                dy_[j] = (max_bounds.y() - min_bounds.y()) / ny_;
            }
            for (int k = 0; k < nz_; ++k) {
                dz_[k] = (max_bounds.z() - min_bounds.z()) / nz_;
            }
        }

        void GridDataStructure::addField(const std::string& name, const Eigen::MatrixXd& data_2d) {
            if (data_2d.rows() != ny_ || data_2d.cols() != nx_) {
                throw std::invalid_argument("2D field dimensions must match grid");
            }

            if (thread_safe_) {
                std::lock_guard<std::mutex> lock(global_mutex_);
                scalar_fields_2d_[name] = data_2d;
                if (field_mutexes_.find(name) == field_mutexes_.end()) {
                    field_mutexes_[name] = std::make_unique<std::mutex>();
                }
            } else {
                scalar_fields_2d_[name] = data_2d;
            }
        }

        void GridDataStructure::addField(const std::string& name,
                                         const std::vector<Eigen::MatrixXd>& data_3d) {
            if (data_3d.size() != static_cast<size_t>(nz_)) {
                throw std::invalid_argument("3D field depth must match grid nz");
            }

            for (const auto& layer : data_3d) {
                if (layer.rows() != ny_ || layer.cols() != nx_) {
                    throw std::invalid_argument("3D field layer dimensions must match grid");
                }
            }

            if (thread_safe_) {
                std::lock_guard<std::mutex> lock(global_mutex_);
                scalar_fields_3d_[name] = data_3d;
                if (field_mutexes_.find(name) == field_mutexes_.end()) {
                    field_mutexes_[name] = std::make_unique<std::mutex>();
                }
            } else {
                scalar_fields_3d_[name] = data_3d;
            }
        }

        void GridDataStructure::addVectorField(const std::string& name,
                                               const std::vector<Eigen::MatrixXd>& components) {
            if (components.size() != 3) {
                throw std::invalid_argument("Vector field must have 3 components");
            }

            // 验证每个分量的维度
            for (size_t comp = 0; comp < 3; ++comp) {
                if (components[comp].rows() != ny_ || components[comp].cols() != nx_) {
                    throw std::invalid_argument("Vector field component dimensions must match grid");
                }
            }

            if (thread_safe_) {
                std::lock_guard<std::mutex> lock(global_mutex_);
                vector_fields_[name] = components;
                if (field_mutexes_.find(name) == field_mutexes_.end()) {
                    field_mutexes_[name] = std::make_unique<std::mutex>();
                }
            } else {
                vector_fields_[name] = components;
            }
        }

        const Eigen::MatrixXd& GridDataStructure::getField2D(const std::string& name) const {
            auto it = scalar_fields_2d_.find(name);
            if (it == scalar_fields_2d_.end()) {
                throw std::runtime_error("Field not found: " + name);
            }
            return it->second;
        }

        const std::vector<Eigen::MatrixXd>& GridDataStructure::getField3D(const std::string& name) const {
            auto it = scalar_fields_3d_.find(name);
            if (it == scalar_fields_3d_.end()) {
                throw std::runtime_error("3D field not found: " + name);
            }
            return it->second;
        }

        std::vector<Eigen::MatrixXd>& GridDataStructure::getField3D(const std::string& name) {
            auto it = scalar_fields_3d_.find(name);
            if (it == scalar_fields_3d_.end()) {
                throw std::runtime_error("3D field not found: " + name);
            }
            return it->second;
        }

        double GridDataStructure::getValue(const std::string& field, int i, int j, int k) const {
            if (!isValidIndex(i, j, k)) {
                return 0.0;
            }

            if (thread_safe_) {
                lockField(field);
            }

            double value = 0.0;

            if (k == 0) {
                // 2D字段
                auto it = scalar_fields_2d_.find(field);
                if (it != scalar_fields_2d_.end()) {
                    value = it->second(j, i);
                }
            } else {
                // 3D字段
                auto it = scalar_fields_3d_.find(field);
                if (it != scalar_fields_3d_.end() && k < static_cast<int>(it->second.size())) {
                    value = it->second[k](j, i);
                }
            }

            if (thread_safe_) {
                unlockField(field);
            }

            return value;
        }

        void GridDataStructure::setValue(const std::string& field, int i, int j, int k, double value) {
            if (!isValidIndex(i, j, k)) {
                return;
            }

            if (thread_safe_) {
                lockField(field);
            }

            if (k == 0) {
                // 2D字段
                auto it = scalar_fields_2d_.find(field);
                if (it != scalar_fields_2d_.end()) {
                    it->second(j, i) = value;
                }
            } else {
                // 3D字段
                auto it = scalar_fields_3d_.find(field);
                if (it != scalar_fields_3d_.end() && k < static_cast<int>(it->second.size())) {
                    it->second[k](j, i) = value;
                }
            }

            if (thread_safe_) {
                unlockField(field);
            }
        }

        Eigen::Vector3d GridDataStructure::getVector(const std::string& field, int i, int j, int k) const {
            if (!isValidIndex(i, j, k)) {
                return Eigen::Vector3d::Zero();
            }

            if (thread_safe_) {
                lockField(field);
            }

            Eigen::Vector3d vector = Eigen::Vector3d::Zero();

            auto it = vector_fields_.find(field);
            if (it != vector_fields_.end() && it->second.size() == 3) {
                vector.x() = it->second[0](j, i);  // x分量
                vector.y() = it->second[1](j, i);  // y分量
                vector.z() = it->second[2](j, i);  // z分量
            }

            if (thread_safe_) {
                unlockField(field);
            }

            return vector;
        }

        double GridDataStructure::interpolateScalar(const Eigen::Vector3d& position,
                                                    const std::string& field,
                                                    InterpolationMethod metho