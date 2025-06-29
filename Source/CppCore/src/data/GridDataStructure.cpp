// src/data/GridDataStructure.cpp
#include "data/GridDataStructure.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

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
                                                    InterpolationMethod method) const {
            // 转换到网格坐标
            auto grid_coords = worldToGridCoords(position);
            double x = std::get<0>(grid_coords);
            double y = std::get<1>(grid_coords);
            double z = std::get<2>(grid_coords);

            // 检查边界
            if (x < 0 || x >= nx_ - 1 || y < 0 || y >= ny_ - 1 || z < 0 || z >= nz_ - 1) {
                return 0.0;
            }

            switch (method) {
                case InterpolationMethod::TRILINEAR: {
                    auto it = scalar_fields_3d_.find(field);
                    if (it != scalar_fields_3d_.end()) {
                        return trilinearInterpolation(it->second, x, y, z);
                    }
                    break;
                }
                case InterpolationMethod::BILINEAR: {
                    auto it = scalar_fields_2d_.find(field);
                    if (it != scalar_fields_2d_.end()) {
                        return bilinearInterpolation(it->second, x, y);
                    }
                    break;
                }
                default:
                    throw std::invalid_argument("Unsupported interpolation method");
            }

            return 0.0;
        }

        Eigen::Vector3d GridDataStructure::interpolateVector(const Eigen::Vector3d& position,
                                                             const std::string& field,
                                                             InterpolationMethod method) const {
            auto it = vector_fields_.find(field);
            if (it == vector_fields_.end()) {
                return Eigen::Vector3d::Zero();
            }

            Eigen::Vector3d result;
            for (int comp = 0; comp < 3; ++comp) {
                // 将每个分量作为标量字段进行插值
                auto grid_coords = worldToGridCoords(position);
                double x = std::get<0>(grid_coords);
                double y = std::get<1>(grid_coords);
                double z = std::get<2>(grid_coords);

                if (method == InterpolationMethod::BILINEAR) {
                    result[comp] = bilinearInterpolation(it->second[comp], x, y);
                } else {
                    // 对于三线性插值，需要构造临时的3D字段
                    std::vector<Eigen::MatrixXd> temp_3d(nz_);
                    for (int k = 0; k < nz_; ++k) {
                        temp_3d[k] = it->second[comp];
                    }
                    result[comp] = trilinearInterpolation(temp_3d, x, y, z);
                }
            }

            return result;
        }

        // 梯度计算
        Eigen::Vector3d GridDataStructure::computeGradient(const std::string& field,
                                                           const Eigen::Vector3d& position) const {
            auto indices = worldToGridIndices(position);
            return computeGradient(field, std::get<0>(indices), std::get<1>(indices), std::get<2>(indices));
        }

        Eigen::Vector3d GridDataStructure::computeGradient(const std::string& field,
                                                           int i, int j, int k) const {
            if (!isValidIndex(i, j, k)) {
                return Eigen::Vector3d::Zero();
            }

            auto it = scalar_fields_3d_.find(field);
            if (it == scalar_fields_3d_.end()) {
                return Eigen::Vector3d::Zero();
            }

            Eigen::Vector3d gradient;

            // 计算x方向导数
            gradient.x() = computeDerivative(it->second, i, j, k, 0);

            // 计算y方向导数  
            gradient.y() = computeDerivative(it->second, i, j, k, 1);

            // 计算z方向导数
            gradient.z() = computeDerivative(it->second, i, j, k, 2);

            return gradient;
        }

        // 散度和旋度
        double GridDataStructure::computeDivergence(const std::string& vector_field,
                                                    const Eigen::Vector3d& position) const {
            auto it = vector_fields_.find(vector_field);
            if (it == vector_fields_.end()) {
                return 0.0;
            }

            auto indices = worldToGridIndices(position);
            int i = std::get<0>(indices);
            int j = std::get<1>(indices);
            int k = std::get<2>(indices);

            if (!isValidIndex(i, j, k)) {
                return 0.0;
            }

            // 构造临时3D字段用于计算导数
            std::vector<Eigen::MatrixXd> u_field(nz_), v_field(nz_), w_field(nz_);
            for (int layer = 0; layer < nz_; ++layer) {
                u_field[layer] = it->second[0];
                v_field[layer] = it->second[1];
                w_field[layer] = it->second[2];
            }

            double du_dx = computeDerivative(u_field, i, j, k, 0);
            double dv_dy = computeDerivative(v_field, i, j, k, 1);
            double dw_dz = computeDerivative(w_field, i, j, k, 2);

            return du_dx + dv_dy + dw_dz;
        }

        Eigen::Vector3d GridDataStructure::computeCurl(const std::string& vector_field,
                                                       const Eigen::Vector3d& position) const {
            auto it = vector_fields_.find(vector_field);
            if (it == vector_fields_.end()) {
                return Eigen::Vector3d::Zero();
            }

            auto indices = worldToGridIndices(position);
            int i = std::get<0>(indices);
            int j = std::get<1>(indices);
            int k = std::get<2>(indices);

            if (!isValidIndex(i, j, k)) {
                return Eigen::Vector3d::Zero();
            }

            // 构造临时3D字段
            std::vector<Eigen::MatrixXd> u_field(nz_), v_field(nz_), w_field(nz_);
            for (int layer = 0; layer < nz_; ++layer) {
                u_field[layer] = it->second[0];
                v_field[layer] = it->second[1];
                w_field[layer] = it->second[2];
            }

            Eigen::Vector3d curl;

            // curl = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
            curl.x() = computeDerivative(w_field, i, j, k, 1) - computeDerivative(v_field, i, j, k, 2);
            curl.y() = computeDerivative(u_field, i, j, k, 2) - computeDerivative(w_field, i, j, k, 0);
            curl.z() = computeDerivative(v_field, i, j, k, 0) - computeDerivative(u_field, i, j, k, 1);

            return curl;
        }

        // 边界处理
        void GridDataStructure::setBoundaryConditions(const std::string& field,
                                                      const std::string& boundary_type) {
            boundary_conditions_[field] = boundary_type;
        }

        void GridDataStructure::applyPeriodicBoundary(const std::string& field, int axis) {
            // 实现周期边界条件
            auto it_2d = scalar_fields_2d_.find(field);
            auto it_3d = scalar_fields_3d_.find(field);

            if (it_2d != scalar_fields_2d_.end()) {
                if (axis == 0) { // x方向周期
                    for (int j = 0; j < ny_; ++j) {
                        it_2d->second(j, 0) = it_2d->second(j, nx_ - 2);
                        it_2d->second(j, nx_ - 1) = it_2d->second(j, 1);
                    }
                } else if (axis == 1) { // y方向周期
                    for (int i = 0; i < nx_; ++i) {
                        it_2d->second(0, i) = it_2d->second(ny_ - 2, i);
                        it_2d->second(ny_ - 1, i) = it_2d->second(1, i);
                    }
                }
            }

            if (it_3d != scalar_fields_3d_.end()) {
                for (auto& layer : it_3d->second) {
                    if (axis == 0) { // x方向周期
                        for (int j = 0; j < ny_; ++j) {
                            layer(j, 0) = layer(j, nx_ - 2);
                            layer(j, nx_ - 1) = layer(j, 1);
                        }
                    } else if (axis == 1) { // y方向周期
                        for (int i = 0; i < nx_; ++i) {
                            layer(0, i) = layer(ny_ - 2, i);
                            layer(ny_ - 1, i) = layer(1, i);
                        }
                    }
                }
            }
        }

        void GridDataStructure::applyNeumannBoundary(const std::string& field) {
            // 实现Neumann边界条件（零梯度）
            auto it_2d = scalar_fields_2d_.find(field);
            auto it_3d = scalar_fields_3d_.find(field);

            if (it_2d != scalar_fields_2d_.end()) {
                // 边界处理
                for (int j = 0; j < ny_; ++j) {
                    it_2d->second(j, 0) = it_2d->second(j, 1);
                    it_2d->second(j, nx_ - 1) = it_2d->second(j, nx_ - 2);
                }
                for (int i = 0; i < nx_; ++i) {
                    it_2d->second(0, i) = it_2d->second(1, i);
                    it_2d->second(ny_ - 1, i) = it_2d->second(ny_ - 2, i);
                }
            }

            if (it_3d != scalar_fields_3d_.end()) {
                for (auto& layer : it_3d->second) {
                    for (int j = 0; j < ny_; ++j) {
                        layer(j, 0) = layer(j, 1);
                        layer(j, nx_ - 1) = layer(j, nx_ - 2);
                    }
                    for (int i = 0; i < nx_; ++i) {
                        layer(0, i) = layer(1, i);
                        layer(ny_ - 1, i) = layer(ny_ - 2, i);
                    }
                }
            }
        }

        void GridDataStructure::applyDirichletBoundary(const std::string& field, double value) {
            // 实现Dirichlet边界条件（固定值）
            auto it_2d = scalar_fields_2d_.find(field);
            auto it_3d = scalar_fields_3d_.find(field);

            if (it_2d != scalar_fields_2d_.end()) {
                // 设置边界值
                for (int j = 0; j < ny_; ++j) {
                    it_2d->second(j, 0) = value;
                    it_2d->second(j, nx_ - 1) = value;
                }
                for (int i = 0; i < nx_; ++i) {
                    it_2d->second(0, i) = value;
                    it_2d->second(ny_ - 1, i) = value;
                }
            }

            if (it_3d != scalar_fields_3d_.end()) {
                for (auto& layer : it_3d->second) {
                    for (int j = 0; j < ny_; ++j) {
                        layer(j, 0) = value;
                        layer(j, nx_ - 1) = value;
                    }
                    for (int i = 0; i < nx_; ++i) {
                        layer(0, i) = value;
                        layer(ny_ - 1, i) = value;
                    }
                }
            }
        }

        // 网格查询方法
        std::vector<double> GridDataStructure::getSpacing() const {
            std::vector<double> spacing;
            spacing.reserve(dx_.size() + dy_.size() + dz_.size());
            spacing.insert(spacing.end(), dx_.begin(), dx_.end());
            spacing.insert(spacing.end(), dy_.begin(), dy_.end());
            spacing.insert(spacing.end(), dz_.begin(), dz_.end());
            return spacing;
        }

        std::pair<Eigen::Vector3d, Eigen::Vector3d> GridDataStructure::getBounds() const {
            return std::make_pair(min_bounds_, max_bounds_);
        }

        // 索引转换
        bool GridDataStructure::isValidIndex(int i, int j, int k) const {
            return i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_;
        }

        int GridDataStructure::flattenIndex(int i, int j, int k) const {
            return k * nx_ * ny_ + j * nx_ + i;
        }

        std::tuple<int, int, int> GridDataStructure::unflattenIndex(int flat_index) const {
            int k = flat_index / (nx_ * ny_);
            int remainder = flat_index % (nx_ * ny_);
            int j = remainder / nx_;
            int i = remainder % nx_;
            return std::make_tuple(i, j, k);
        }

        // 内存管理
        void GridDataStructure::clearField(const std::string& name) {
            scalar_fields_2d_.erase(name);
            scalar_fields_3d_.erase(name);
            vector_fields_.erase(name);
            boundary_conditions_.erase(name);
            if (thread_safe_) {
                field_mutexes_.erase(name);
            }
        }

        void GridDataStructure::clearAllFields() {
            scalar_fields_2d_.clear();
            scalar_fields_3d_.clear();
            vector_fields_.clear();
            boundary_conditions_.clear();
            if (thread_safe_) {
                field_mutexes_.clear();
            }
        }

        size_t GridDataStructure::getMemoryUsage() const {
            size_t total = 0;

            // 计算2D字段内存
            for (const auto& field : scalar_fields_2d_) {
                total += field.second.size() * sizeof(double);
            }

            // 计算3D字段内存
            for (const auto& field : scalar_fields_3d_) {
                for (const auto& layer : field.second) {
                    total += layer.size() * sizeof(double);
                }
            }

            // 计算矢量字段内存
            for (const auto& field : vector_fields_) {
                for (const auto& component : field.second) {
                    total += component.size() * sizeof(double);
                }
            }

            return total;
        }

        void GridDataStructure::optimizeMemoryLayout() {
            // 内存布局优化（预留接口）
            compactFieldStorage();
            alignMemoryAccess();
        }

        // 并行访问支持
        void GridDataStructure::enableThreadSafety(bool enable_thread_safety) {
            thread_safe_ = enable_thread_safety;
            if (enable_thread_safety) {
                // 为现有字段创建互斥锁
                for (const auto& field : scalar_fields_2d_) {
                    if (field_mutexes_.find(field.first) == field_mutexes_.end()) {
                        field_mutexes_[field.first] = std::make_unique<std::mutex>();
                    }
                }
                for (const auto& field : scalar_fields_3d_) {
                    if (field_mutexes_.find(field.first) == field_mutexes_.end()) {
                        field_mutexes_[field.first] = std::make_unique<std::mutex>();
                    }
                }
                for (const auto& field : vector_fields_) {
                    if (field_mutexes_.find(field.first) == field_mutexes_.end()) {
                        field_mutexes_[field.first] = std::make_unique<std::mutex>();
                    }
                }
            }
        }

        void GridDataStructure::lockField(const std::string& field) const {
            if (thread_safe_) {
                auto it = field_mutexes_.find(field);
                if (it != field_mutexes_.end()) {
                    it->second->lock();
                }
            }
        }

        void GridDataStructure::unlockField(const std::string& field) const {
            if (thread_safe_) {
                auto it = field_mutexes_.find(field);
                if (it != field_mutexes_.end()) {
                    it->second->unlock();
                }
            }
        }

        // 坐标变换
        void GridDataStructure::setCoordinateTransform(const Eigen::Matrix3d& transform_matrix) {
            transform_matrix_ = transform_matrix;
            has_transform_ = true;
        }

        Eigen::Vector3d GridDataStructure::transformToGrid(const Eigen::Vector3d& world_coords) const {
            if (has_transform_) {
                return transform_matrix_ * (world_coords - origin_);
            }
            return world_coords - origin_;
        }

        Eigen::Vector3d GridDataStructure::transformToWorld(const Eigen::Vector3d& grid_coords) const {
            if (has_transform_) {
                return transform_matrix_.inverse() * grid_coords + origin_;
            }
            return grid_coords + origin_;
        }

        // 统计信息
        double GridDataStructure::getMinValue(const std::string& field) const {
            double min_val = std::numeric_limits<double>::max();

            auto it_2d = scalar_fields_2d_.find(field);
            if (it_2d != scalar_fields_2d_.end()) {
                min_val = std::min(min_val, it_2d->second.minCoeff());
            }

            auto it_3d = scalar_fields_3d_.find(field);
            if (it_3d != scalar_fields_3d_.end()) {
                for (const auto& layer : it_3d->second) {
                    min_val = std::min(min_val, layer.minCoeff());
                }
            }

            return min_val;
        }

        double GridDataStructure::getMaxValue(const std::string& field) const {
            double max_val = std::numeric_limits<double>::lowest();

            auto it_2d = scalar_fields_2d_.find(field);
            if (it_2d != scalar_fields_2d_.end()) {
                max_val = std::max(max_val, it_2d->second.maxCoeff());
            }

            auto it_3d = scalar_fields_3d_.find(field);
            if (it_3d != scalar_fields_3d_.end()) {
                for (const auto& layer : it_3d->second) {
                    max_val = std::max(max_val, layer.maxCoeff());
                }
            }

            return max_val;
        }

        double GridDataStructure::getMeanValue(const std::string& field) const {
            double sum = 0.0;
            int count = 0;

            auto it_2d = scalar_fields_2d_.find(field);
            if (it_2d != scalar_fields_2d_.end()) {
                sum += it_2d->second.sum();
                count += it_2d->second.size();
            }

            auto it_3d = scalar_fields_3d_.find(field);
            if (it_3d != scalar_fields_3d_.end()) {
                for (const auto& layer : it_3d->second) {
                    sum += layer.sum();
                    count += layer.size();
                }
            }

            return count > 0 ? sum / count : 0.0;
        }

        double GridDataStructure::getStandardDeviation(const std::string& field) const {
            double mean = getMeanValue(field);
            double variance = 0.0;
            int count = 0;

            auto it_2d = scalar_fields_2d_.find(field);
            if (it_2d != scalar_fields_2d_.end()) {
                for (int i = 0; i < it_2d->second.rows(); ++i) {
                    for (int j = 0; j < it_2d->second.cols(); ++j) {
                        double diff = it_2d->second(i, j) - mean;
                        variance += diff * diff;
                        count++;
                    }
                }
            }

            auto it_3d = scalar_fields_3d_.find(field);
            if (it_3d != scalar_fields_3d_.end()) {
                for (const auto& layer : it_3d->second) {
                    for (int i = 0; i < layer.rows(); ++i) {
                        for (int j = 0; j < layer.cols(); ++j) {
                            double diff = layer(i, j) - mean;
                            variance += diff * diff;
                            count++;
                        }
                    }
                }
            }

            return count > 1 ? std::sqrt(variance / (count - 1)) : 0.0;
        }

        // 质量检查
        bool GridDataStructure::validateField(const std::string& field) const {
            auto it_2d = scalar_fields_2d_.find(field);
            auto it_3d = scalar_fields_3d_.find(field);
            auto it_vec = vector_fields_.find(field);

            if (it_2d != scalar_fields_2d_.end()) {
                // 检查2D字段是否有有效数据
                return !it_2d->second.hasNaN() && it_2d->second.allFinite();
            }

            if (it_3d != scalar_fields_3d_.end()) {
                // 检查3D字段是否有有效数据
                for (const auto& layer : it_3d->second) {
                    if (layer.hasNaN() || !layer.allFinite()) {
                        return false;
                    }
                }
                return true;
            }

            if (it_vec != vector_fields_.end()) {
                // 检查矢量字段是否有有效数据
                for (const auto& component : it_vec->second) {
                    if (component.hasNaN() || !component.allFinite()) {
                        return false;
                    }
                }
                return true;
            }

            return false;
        }

        std::vector<std::string> GridDataStructure::getFieldNames() const {
            std::vector<std::string> names;

            for (const auto& field : scalar_fields_2d_) {
                names.push_back(field.first);
            }

            for (const auto& field : scalar_fields_3d_) {
                names.push_back(field.first);
            }

            for (const auto& field : vector_fields_) {
                names.push_back(field.first);
            }

            return names;
        }

        bool GridDataStructure::hasField(const std::string& field) const {
            return scalar_fields_2d_.find(field) != scalar_fields_2d_.end() ||
                   scalar_fields_3d_.find(field) != scalar_fields_3d_.end() ||
                   vector_fields_.find(field) != vector_fields_.end();
        }

        // 数据导入导出
        bool GridDataStructure::loadFromNetCDF(const std::string& filename) {
            // NetCDF文件读取的简化实现
            // 在实际应用中需要链接NetCDF库
            try {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // 这里应该实现NetCDF格式解析
                // 由于没有NetCDF库依赖，暂时返回true表示接口存在
                file.close();
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        bool GridDataStructure::saveToNetCDF(const std::string& filename) const {
            // NetCDF文件写入的简化实现
            try {
                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // 这里应该实现NetCDF格式写入
                file.close();
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        bool GridDataStructure::loadFromHDF5(const std::string& filename) {
            // HDF5文件读取的简化实现
            try {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // 这里应该实现HDF5格式解析
                file.close();
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        bool GridDataStructure::saveToHDF5(const std::string& filename) const {
            // HDF5文件写入的简化实现
            try {
                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // 这里应该实现HDF5格式写入
                file.close();
                return true;
            } catch (const std::exception&) {
                return false;
            }
        }

        // 内部方法实现
        double GridDataStructure::trilinearInterpolation(const std::vector<Eigen::MatrixXd>& field,
                                                         double x, double y, double z) const {
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            int z0 = static_cast<int>(std::floor(z));

            int x1 = std::min(x0 + 1, nx_ - 1);
            int y1 = std::min(y0 + 1, ny_ - 1);
            int z1 = std::min(z0 + 1, nz_ - 1);

            double xd = x - x0;
            double yd = y - y0;
            double zd = z - z0;

            // 三线性插值
            double c000 = field[z0](y0, x0);
            double c001 = field[z0](y0, x1);
            double c010 = field[z0](y1, x0);
            double c011 = field[z0](y1, x1);
            double c100 = field[z1](y0, x0);
            double c101 = field[z1](y0, x1);
            double c110 = field[z1](y1, x0);
            double c111 = field[z1](y1, x1);

            double c00 = c000 * (1 - xd) + c001 * xd;
            double c01 = c010 * (1 - xd) + c011 * xd;
            double c10 = c100 * (1 - xd) + c101 * xd;
            double c11 = c110 * (1 - xd) + c111 * xd;

            double c0 = c00 * (1 - yd) + c01 * yd;
            double c1 = c10 * (1 - yd) + c11 * yd;

            return c0 * (1 - zd) + c1 * zd;
        }

        double GridDataStructure::bilinearInterpolation(const Eigen::MatrixXd& field,
                                                        double x, double y) const {
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));

            int x1 = std::min(x0 + 1, nx_ - 1);
            int y1 = std::min(y0 + 1, ny_ - 1);

            double xd = x - x0;
            double yd = y - y0;

            // 双线性插值
            double c00 = field(y0, x0);
            double c01 = field(y0, x1);
            double c10 = field(y1, x0);
            double c11 = field(y1, x1);

            double c0 = c00 * (1 - xd) + c01 * xd;
            double c1 = c10 * (1 - xd) + c11 * xd;

            return c0 * (1 - yd) + c1 * yd;
        }

        double GridDataStructure::cubicInterpolation(const std::vector<double>& values,
                                                     const std::vector<double>& positions,
                                                     double target_pos) const {
            // 三次样条插值的简化实现
            if (values.size() != positions.size() || values.size() < 2) {
                return 0.0;
            }

            // 找到插值区间
            int i = 0;
            for (size_t j = 1; j < positions.size(); ++j) {
                if (target_pos <= positions[j]) {
                    i = j - 1;
                    break;
                }
            }

            if (i >= static_cast<int>(values.size()) - 1) {
                return values.back();
            }

            // 线性插值作为简化版本
            double t = (target_pos - positions[i]) / (positions[i + 1] - positions[i]);
            return values[i] * (1.0 - t) + values[i + 1] * t;
        }

        // 网格坐标转换
        std::tuple<int, int, int> GridDataStructure::worldToGridIndices(const Eigen::Vector3d& world_pos) const {
            Eigen::Vector3d grid_pos = transformToGrid(world_pos);

            int i = static_cast<int>(std::round(grid_pos.x()));
            int j = static_cast<int>(std::round(grid_pos.y()));
            int k = static_cast<int>(std::round(grid_pos.z()));

            return std::make_tuple(
                    std::max(0, std::min(i, nx_ - 1)),
                    std::max(0, std::min(j, ny_ - 1)),
                    std::max(0, std::min(k, nz_ - 1))
            );
        }

        std::tuple<double, double, double> GridDataStructure::worldToGridCoords(const Eigen::Vector3d& world_pos) const {
            Eigen::Vector3d grid_pos = transformToGrid(world_pos);
            return std::make_tuple(grid_pos.x(), grid_pos.y(), grid_pos.z());
        }

        // 边界处理辅助
        void GridDataStructure::applyBoundaryToMatrix(Eigen::MatrixXd& matrix, const std::string& boundary_type) const {
            if (boundary_type == "periodic") {
                // 周期边界
                for (int j = 0; j < matrix.rows(); ++j) {
                    matrix(j, 0) = matrix(j, matrix.cols() - 2);
                    matrix(j, matrix.cols() - 1) = matrix(j, 1);
                }
                for (int i = 0; i < matrix.cols(); ++i) {
                    matrix(0, i) = matrix(matrix.rows() - 2, i);
                    matrix(matrix.rows() - 1, i) = matrix(1, i);
                }
            } else if (boundary_type == "neumann") {
                // Neumann边界
                for (int j = 0; j < matrix.rows(); ++j) {
                    matrix(j, 0) = matrix(j, 1);
                    matrix(j, matrix.cols() - 1) = matrix(j, matrix.cols() - 2);
                }
                for (int i = 0; i < matrix.cols(); ++i) {
                    matrix(0, i) = matrix(1, i);
                    matrix(matrix.rows() - 1, i) = matrix(matrix.rows() - 2, i);
                }
            }
        }

        void GridDataStructure::extrapolateToGhostCells(Eigen::MatrixXd& matrix) const {
            // 外推到虚拟单元
            int rows = matrix.rows();
            int cols = matrix.cols();

            // 边界外推
            for (int j = 1; j < rows - 1; ++j) {
                matrix(j, 0) = 2.0 * matrix(j, 1) - matrix(j, 2);
                matrix(j, cols - 1) = 2.0 * matrix(j, cols - 2) - matrix(j, cols - 3);
            }

            for (int i = 1; i < cols - 1; ++i) {
                matrix(0, i) = 2.0 * matrix(1, i) - matrix(2, i);
                matrix(rows - 1, i) = 2.0 * matrix(rows - 2, i) - matrix(rows - 3, i);
            }

            // 角落处理
            matrix(0, 0) = 0.5 * (matrix(0, 1) + matrix(1, 0));
            matrix(0, cols - 1) = 0.5 * (matrix(0, cols - 2) + matrix(1, cols - 1));
            matrix(rows - 1, 0) = 0.5 * (matrix(rows - 2, 0) + matrix(rows - 1, 1));
            matrix(rows - 1, cols - 1) = 0.5 * (matrix(rows - 2, cols - 1) + matrix(rows - 1, cols - 2));
        }

        // 数值微分
        double GridDataStructure::computeDerivative(const std::vector<Eigen::MatrixXd>& field,
                                                    int i, int j, int k, int direction) const {
            if (!isValidIndex(i, j, k)) {
                return 0.0;
            }

            double derivative = 0.0;

            switch (direction) {
                case 0: // x方向
                    if (i > 0 && i < nx_ - 1) {
                        derivative = (field[k](j, i + 1) - field[k](j, i - 1)) / (2.0 * dx_[i]);
                    } else if (i == 0) {
                        derivative = (field[k](j, i + 1) - field[k](j, i)) / dx_[i];
                    } else {
                        derivative = (field[k](j, i) - field[k](j, i - 1)) / dx_[i - 1];
                    }
                    break;

                case 1: // y方向
                    if (j > 0 && j < ny_ - 1) {
                        derivative = (field[k](j + 1, i) - field[k](j - 1, i)) / (2.0 * dy_[j]);
                    } else if (j == 0) {
                        derivative = (field[k](j + 1, i) - field[k](j, i)) / dy_[j];
                    } else {
                        derivative = (field[k](j, i) - field[k](j - 1, i)) / dy_[j - 1];
                    }
                    break;

                case 2: // z方向
                    if (k > 0 && k < nz_ - 1) {
                        derivative = (field[k + 1](j, i) - field[k - 1](j, i)) / (dz_[k - 1] + dz_[k]);
                    } else if (k == 0) {
                        derivative = (field[k + 1](j, i) - field[k](j, i)) / dz_[k];
                    } else {
                        derivative = (field[k](j, i) - field[k - 1](j, i)) / dz_[k - 1];
                    }
                    break;
            }

            return derivative;
        }

        // 内存优化
        void GridDataStructure::compactFieldStorage() {
            // 压缩字段存储（预留接口）
            // 可以实现数据压缩、稀疏存储等优化
        }

        void GridDataStructure::alignMemoryAccess() {
            // 内存访问对齐优化（预留接口）
            // 可以实现内存布局重排等优化
        }

        // ===== HybridCoordinateGrid 实现 =====

        HybridCoordinateGrid::HybridCoordinateGrid(int nx, int ny, int nz)
                : GridDataStructure(nx, ny, nz, CoordinateSystem::HYBRID_SIGMA, GridType::REGULAR),
                  has_topography_(false), has_surface_(false) {
        }

        void HybridCoordinateGrid::setSigmaLevels(const std::vector<double>& sigma_levels) {
            if (sigma_levels.size() != static_cast<size_t>(nz_)) {
                throw std::invalid_argument("Sigma levels size must match nz");
            }

            sigma_levels_ = sigma_levels;
            updateVerticalCoordinates();
        }

        void HybridCoordinateGrid::setBottomTopography(const Eigen::MatrixXd& bottom_depth) {
            if (bottom_depth.rows() != ny_ || bottom_depth.cols() != nx_) {
                throw std::invalid_argument("Bottom topography dimensions must match grid");
            }

            bottom_depth_ = bottom_depth;
            has_topography_ = true;
            updateVerticalCoordinates();
        }

        void HybridCoordinateGrid::setSurfaceElevation(const Eigen::MatrixXd& surface_elevation) {
            if (surface_elevation.rows() != ny_ || surface_elevation.cols() != nx_) {
                throw std::invalid_argument("Surface elevation dimensions must match grid");
            }

            surface_elevation_ = surface_elevation;
            has_surface_ = true;
            updateVerticalCoordinates();
        }

        double HybridCoordinateGrid::sigmaToZ(double sigma, int i, int j) const {
            if (!has_topography_ || !has_surface_) {
                return sigma;
            }

            if (!isValidIndex(i, j, 0)) {
                return 0.0;
            }

            double bottom = -bottom_depth_(j, i);  // 负值表示深度
            double surface = surface_elevation_(j, i);

            return bottom + sigma * (surface - bottom);
        }

        double HybridCoordinateGrid::zToSigma(double z, int i, int j) const {
            if (!has_topography_ || !has_surface_) {
                return z;
            }

            if (!isValidIndex(i, j, 0)) {
                return 0.0;
            }

            double bottom = -bottom_depth_(j, i);
            double surface = surface_elevation_(j, i);

            if (std::abs(surface - bottom) < 1e-10) {
                return 0.0;
            }

            return (z - bottom) / (surface - bottom);
        }

        double HybridCoordinateGrid::interpolateVertical(const std::string& field,
                                                         int i, int j, double z_target) const {
            if (!has_topography_ || !has_surface_) {
                return getValue(field, i, j, 0);
            }

            // 转换目标深度到sigma坐标
            double sigma_target = zToSigma(z_target, i, j);

            // 在sigma层之间插值
            for (int k = 0; k < nz_ - 1; ++k) {
                if (sigma_target >= sigma_levels_[k] && sigma_target <= sigma_levels_[k + 1]) {
                    double sigma0 = sigma_levels_[k];
                    double sigma1 = sigma_levels_[k + 1];
                    double value0 = getValue(field, i, j, k);
                    double value1 = getValue(field, i, j, k + 1);

                    double weight = (sigma_target - sigma0) / (sigma1 - sigma0);
                    return value0 * (1.0 - weight) + value1 * weight;
                }
            }

            return 0.0;
        }

        Eigen::MatrixXd HybridCoordinateGrid::computeLayerThickness(int k) const {
            if (!has_topography_ || !has_surface_ || k >= nz_ - 1) {
                return Eigen::MatrixXd::Ones(ny_, nx_);
            }

            Eigen::MatrixXd thickness(ny_, nx_);

            for (int i = 0; i < nx_; ++i) {
                for (int j = 0; j < ny_; ++j) {
                    double bottom = -bottom_depth_(j, i);
                    double surface = surface_elevation_(j, i);
                    double total_depth = surface - bottom;

                    double sigma_thickness = sigma_levels_[k + 1] - sigma_levels_[k];
                    thickness(j, i) = sigma_thickness * total_depth;
                }
            }

            return thickness;
        }

        double HybridCoordinateGrid::getLayerThickness(int i, int j, int k) const {
            if (!isValidIndex(i, j, k) || k >= nz_ - 1) {
                return 0.0;
            }

            if (!has_topography_ || !has_surface_) {
                return dz_[k];
            }

            double bottom = -bottom_depth_(j, i);
            double surface = surface_elevation_(j, i);
            double total_depth = surface - bottom;

            double sigma_thickness = sigma_levels_[k + 1] - sigma_levels_[k];
            return sigma_thickness * total_depth;
        }

        void HybridCoordinateGrid::updateVerticalCoordinates() {
            if (!has_topography_ || !has_surface_ || sigma_levels_.empty()) {
                return;
            }

            // 更新垂直坐标系统
            // 可以在这里实现更复杂的混合坐标变换
        }

        // ===== AdaptiveGrid 实现 =====

        AdaptiveGrid::AdaptiveGrid(int base_nx, int base_ny, int base_nz, int max_refinement_level)
                : GridDataStructure(base_nx, base_ny, base_nz), max_refinement_level_(max_refinement_level) {

            // 初始化细化级别数组
            refinement_levels_.resize(nz_);
            for (int k = 0; k < nz_; ++k) {
                refinement_levels_[k].resize(ny_);
                for (int j = 0; j < ny_; ++j) {
                    refinement_levels_[k][j].resize(nx_, 0);
                }
            }
        }

        void AdaptiveGrid::refineRegion(const Eigen::Vector3d& center, double radius, int levels) {
            levels = std::min(levels, max_refinement_level_);

            for (int k = 0; k < nz_; ++k) {
                for (int j = 0; j < ny_; ++j) {
                    for (int i = 0; i < nx_; ++i) {
                        Eigen::Vector3d grid_point(i, j, k);
                        Eigen::Vector3d world_point = transformToWorld(grid_point);

                        double distance = (world_point - center).norm();
                        if (distance <= radius) {
                            refinement_levels_[k][j][i] = std::min(
                                    refinement_levels_[k][j][i] + levels,
                                    max_refinement_level_
                            );
                        }
                    }
                }
            }
        }

        void AdaptiveGrid::refineBasedOnGradient(const std::string& field, double threshold) {
            for (int k = 1; k < nz_ - 1; ++k) {
                for (int j = 1; j < ny_ - 1; ++j) {
                    for (int i = 1; i < nx_ - 1; ++i) {
                        Eigen::Vector3d gradient = computeGradient(field, i, j, k);
                        double gradient_magnitude = gradient.norm();

                        if (gradient_magnitude > threshold) {
                            refinement_levels_[k][j][i] = std::min(
                                    refinement_levels_[k][j][i] + 1,
                                    max_refinement_level_
                            );
                        }
                    }
                }
            }
        }

        void AdaptiveGrid::coarsenRegion(const Eigen::Vector3d& center, double radius) {
            for (int k = 0; k < nz_; ++k) {
                for (int j = 0; j < ny_; ++j) {
                    for (int i = 0; i < nx_; ++i) {
                        Eigen::Vector3d grid_point(i, j, k);
                        Eigen::Vector3d world_point = transformToWorld(grid_point);

                        double distance = (world_point - center).norm();
                        if (distance <= radius) {
                            refinement_levels_[k][j][i] = std::max(
                                    refinement_levels_[k][j][i] - 1,
                                    0
                            );
                        }
                    }
                }
            }
        }

        void AdaptiveGrid::updateRefinement() {
            // 更新细化结构
            // 可以实现网格重新生成、数据重新分布等
        }

        void AdaptiveGrid::balanceLoad() {
            // 负载平衡
            // 可以实现并行计算时的负载分配优化
        }

        int AdaptiveGrid::getRefinementLevel(int i, int j, int k) const {
            if (!isValidIndex(i, j, k)) {
                return 0;
            }
            return refinement_levels_[k][j][i];
        }

        void AdaptiveGrid::subdivideCell(int i, int j, int k) {
            // 细分单元
            if (isValidIndex(i, j, k)) {
                refinement_levels_[k][j][i] = std::min(
                        refinement_levels_[k][j][i] + 1,
                        max_refinement_level_
                );
            }
        }

        void AdaptiveGrid::mergeCell(int i, int j, int k) {
            // 合并单元
            if (isValidIndex(i, j, k)) {
                refinement_levels_[k][j][i] = std::max(
                        refinement_levels_[k][j][i] - 1,
                        0
                );
            }
        }

        double AdaptiveGrid::estimateError(const std::string& field, int i, int j, int k) const {
            // 误差估计
            if (!isValidIndex(i, j, k)) {
                return 0.0;
            }

            // 简化的误差估计：基于梯度
            Eigen::Vector3d gradient = computeGradient(field, i, j, k);
            return gradient.norm();
        }

    } // namespace Data
} // namespace OceanSim