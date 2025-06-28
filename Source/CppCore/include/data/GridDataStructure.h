// include/data/GridDataStructure.h
#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace OceanSim {
    namespace Data {

/**
 * @brief 高性能海洋网格数据结构
 * 支持混合坐标系统、多层数据存储和高效插值
 */
        class GridDataStructure {
        public:
            // 坐标系统类型
            enum class CoordinateSystem {
                CARTESIAN,      // 直角坐标系
                SPHERICAL,      // 球坐标系
                HYBRID_SIGMA,   // 混合σ坐标系
                ISOPYCNAL      // 等密度坐标系
            };

            // 网格类型
            enum class GridType {
                REGULAR,        // 规则网格
                CURVILINEAR,    // 曲线坐标网格
                UNSTRUCTURED    // 非结构化网格
            };

            // 插值方法
            enum class InterpolationMethod {
                LINEAR,         // 线性插值
                CUBIC,          // 三次样条插值
                BILINEAR,       // 双线性插值
                TRILINEAR,      // 三线性插值
                CONSERVATIVE    // 保守插值
            };

            // 构造函数
            GridDataStructure(int nx, int ny, int nz,
                              CoordinateSystem coord_sys = CoordinateSystem::CARTESIAN,
                              GridType grid_type = GridType::REGULAR);

            ~GridDataStructure() = default;

            // 网格设置
            void setSpacing(double dx, double dy, const std::vector<double>& dz);
            void setSpacing(const std::vector<double>& dx,
                            const std::vector<double>& dy,
                            const std::vector<double>& dz);

            void setOrigin(const Eigen::Vector3d& origin);
            void setBounds(const Eigen::Vector3d& min_bounds,
                           const Eigen::Vector3d& max_bounds);

            // 坐标变换
            void setCoordinateTransform(const Eigen::Matrix3d& transform_matrix);
            Eigen::Vector3d transformToGrid(const Eigen::Vector3d& world_coords) const;
            Eigen::Vector3d transformToWorld(const Eigen::Vector3d& grid_coords) const;

            // 数据存储和访问
            void addField(const std::string& name, const Eigen::MatrixXd& data_2d);
            void addField(const std::string& name, const std::vector<Eigen::MatrixXd>& data_3d);
            void addVectorField(const std::string& name,
                                const std::vector<Eigen::MatrixXd>& components);

            const Eigen::MatrixXd& getField2D(const std::string& name) const;
            const std::vector<Eigen::MatrixXd>& getField3D(const std::string& name) const;
            std::vector<Eigen::MatrixXd>& getField3D(const std::string& name);

            // 高性能数据访问
            double getValue(const std::string& field, int i, int j, int k = 0) const;
            void setValue(const std::string& field, int i, int j, int k, double value);
            Eigen::Vector3d getVector(const std::string& field, int i, int j, int k = 0) const;

            // 插值方法
            double interpolateScalar(const Eigen::Vector3d& position,
                                     const std::string& field,
                                     InterpolationMethod method = InterpolationMethod::TRILINEAR) const;

            Eigen::Vector3d interpolateVector(const Eigen::Vector3d& position,
                                              const std::string& field,
                                              InterpolationMethod method = InterpolationMethod::TRILINEAR) const;

            // 梯度计算
            Eigen::Vector3d computeGradient(const std::string& field,
                                            const Eigen::Vector3d& position) const;
            Eigen::Vector3d computeGradient(const std::string& field,
                                            int i, int j, int k) const;

            // 散度和旋度
            double computeDivergence(const std::string& vector_field,
                                     const Eigen::Vector3d& position) const;
            Eigen::Vector3d computeCurl(const std::string& vector_field,
                                        const Eigen::Vector3d& position) const;

            // 边界处理
            void setBoundaryConditions(const std::string& field,
                                       const std::string& boundary_type);
            void applyPeriodicBoundary(const std::string& field, int axis);
            void applyNeumannBoundary(const std::string& field);
            void applyDirichletBoundary(const std::string& field, double value);

            // 网格查询
            std::vector<int> getDimensions() const { return {nx_, ny_, nz_}; }
            std::vector<double> getSpacing() const;
            Eigen::Vector3d getOrigin() const { return origin_; }
            std::pair<Eigen::Vector3d, Eigen::Vector3d> getBounds() const;

            // 索引转换
            bool isValidIndex(int i, int j, int k) const;
            int flattenIndex(int i, int j, int k) const;
            std::tuple<int, int, int> unflattenIndex(int flat_index) const;

            // 内存管理
            void clearField(const std::string& name);
            void clearAllFields();
            size_t getMemoryUsage() const;
            void optimizeMemoryLayout();

            // 并行访问支持
            void enableThreadSafety(bool enable_thread_safety);
            void lockField(const std::string& field) const;
            void unlockField(const std::string& field) const;

            // 数据导入导出
            bool loadFromNetCDF(const std::string& filename);
            bool saveToNetCDF(const std::string& filename) const;
            bool loadFromHDF5(const std::string& filename);
            bool saveToHDF5(const std::string& filename) const;

            // 统计信息
            double getMinValue(const std::string& field) const;
            double getMaxValue(const std::string& field) const;
            double getMeanValue(const std::string& field) const;
            double getStandardDeviation(const std::string& field) const;

            // 质量检查
            bool validateField(const std::string& field) const;
            std::vector<std::string> getFieldNames() const;
            bool hasField(const std::string& field) const;

        private:
            // 网格参数
            int nx_, ny_, nz_;
            CoordinateSystem coord_system_;
            GridType grid_type_;

            // 空间参数
            std::vector<double> dx_, dy_, dz_;
            Eigen::Vector3d origin_;
            Eigen::Vector3d min_bounds_, max_bounds_;
            Eigen::Matrix3d transform_matrix_;
            bool has_transform_ = false;

            // 数据存储
            std::unordered_map<std::string, Eigen::MatrixXd> scalar_fields_2d_;
            std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> scalar_fields_3d_;
            std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> vector_fields_;

            // 边界条件
            std::unordered_map<std::string, std::string> boundary_conditions_;

            // 线程安全
            bool thread_safe_ = false;
            mutable std::unordered_map<std::string, std::unique_ptr<std::mutex>> field_mutexes_;
            mutable std::mutex global_mutex_;

            // 内部方法
            void initializeSpacing();
            void updateBounds();

            // 插值实现
            double trilinearInterpolation(const std::vector<Eigen::MatrixXd>& field,
                                          double x, double y, double z) const;
            double bilinearInterpolation(const Eigen::MatrixXd& field,
                                         double x, double y) const;
            double cubicInterpolation(const std::vector<double>& values,
                                      const std::vector<double>& positions,
                                      double target_pos) const;

            // 网格坐标转换
            std::tuple<int, int, int> worldToGridIndices(const Eigen::Vector3d& world_pos) const;
            std::tuple<double, double, double> worldToGridCoords(const Eigen::Vector3d& world_pos) const;

            // 边界处理辅助
            void applyBoundaryToMatrix(Eigen::MatrixXd& matrix, const std::string& boundary_type) const;
            void extrapolateToGhostCells(Eigen::MatrixXd& matrix) const;

            // 数值微分
            double computeDerivative(const std::vector<Eigen::MatrixXd>& field,
                                     int i, int j, int k, int direction) const;

            // 内存优化
            void compactFieldStorage();
            void alignMemoryAccess();
        };

/**
 * @brief 混合坐标网格扩展
 * 专门用于海洋模式的σ坐标系统
 */
        class HybridCoordinateGrid : public GridDataStructure {
        public:
            HybridCoordinateGrid(int nx, int ny, int nz);

            // σ坐标设置
            void setSigmaLevels(const std::vector<double>& sigma_levels);
            void setBottomTopography(const Eigen::MatrixXd& bottom_depth);
            void setSurfaceElevation(const Eigen::MatrixXd& surface_elevation);

            // 坐标变换
            double sigmaToZ(double sigma, int i, int j) const;
            double zToSigma(double z, int i, int j) const;

            // 垂直插值
            double interpolateVertical(const std::string& field,
                                       int i, int j, double z_target) const;

            // 层厚度计算
            Eigen::MatrixXd computeLayerThickness(int k) const;
            double getLayerThickness(int i, int j, int k) const;

        private:
            std::vector<double> sigma_levels_;
            Eigen::MatrixXd bottom_depth_;
            Eigen::MatrixXd surface_elevation_;
            bool has_topography_ = false;
            bool has_surface_ = false;

            void updateVerticalCoordinates();
        };

/**
 * @brief 自适应网格细化
 * 基于误差估计的动态网格调整
 */
        class AdaptiveGrid : public GridDataStructure {
        public:
            AdaptiveGrid(int base_nx, int base_ny, int base_nz, int max_refinement_level = 3);

            // 细化控制
            void refineRegion(const Eigen::Vector3d& center, double radius, int levels = 1);
            void refineBasedOnGradient(const std::string& field, double threshold);
            void coarsenRegion(const Eigen::Vector3d& center, double radius);

            // 网格管理
            void updateRefinement();
            void balanceLoad();
            int getRefinementLevel(int i, int j, int k) const;

        private:
            int max_refinement_level_;
            std::vector<std::vector<std::vector<int>>> refinement_levels_;

            void subdivideCell(int i, int j, int k);
            void mergeCell(int i, int j, int k);
            double estimateError(const std::string& field, int i, int j, int k) const;
        };

    } // namespace Data
} // namespace OceanSim