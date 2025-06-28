// include/utils/MathUtils.h
#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

namespace OceanSim {
    namespace Utils {

/**
 * @brief 海洋数值计算数学工具库
 * 提供高效的数值计算、插值、积分和优化算法
 */
        class MathUtils {
        public:
            // ========================= 常量定义 =========================
            static constexpr double PI = 3.14159265358979323846;
            static constexpr double EARTH_RADIUS = 6371000.0;           // 地球半径（米）
            static constexpr double GRAVITY = 9.81;                     // 重力加速度
            static constexpr double OMEGA_EARTH = 7.2921159e-5;         // 地球自转角速度
            static constexpr double SEAWATER_DENSITY = 1025.0;          // 标准海水密度
            static constexpr double ATMOSPHERIC_PRESSURE = 101325.0;     // 标准大气压

            // ========================= 坐标变换 =========================

            /**
             * @brief 地理坐标转换为投影坐标
             */
            static Eigen::Vector2d geographicToProjected(double latitude, double longitude,
                                                         const std::string& projection = "mercator");

            static Eigen::Vector2d projectedToGeographic(double x, double y,
                                                         const std::string& projection = "mercator");

            /**
             * @brief 球面坐标系变换
             */
            static Eigen::Vector3d sphericalToCartesian(double r, double theta, double phi);
            static Eigen::Vector3d cartesianToSpherical(const Eigen::Vector3d& cartesian);

            /**
             * @brief 地球坐标系变换
             */
            static double haversineDistance(double lat1, double lon1, double lat2, double lon2);
            static double azimuthAngle(double lat1, double lon1, double lat2, double lon2);
            static Eigen::Vector2d destinationPoint(double lat, double lon, double distance, double bearing);

            // ========================= 插值算法 =========================

            /**
             * @brief 一维插值
             */
            static double linearInterpolation(const std::vector<double>& x,
                                              const std::vector<double>& y, double xi);

            static double cubicSplineInterpolation(const std::vector<double>& x,
                                                   const std::vector<double>& y, double xi);

            static double lagrangeInterpolation(const std::vector<double>& x,
                                                const std::vector<double>& y, double xi);

            /**
             * @brief 多维插值
             */
            static double bilinearInterpolation(const Eigen::MatrixXd& grid,
                                                double x, double y,
                                                double x_min, double x_max,
                                                double y_min, double y_max);

            static double trilinearInterpolation(const std::vector<Eigen::MatrixXd>& grid,
                                                 double x, double y, double z,
                                                 const Eigen::Vector3d& min_coords,
                                                 const Eigen::Vector3d& max_coords);

            static Eigen::Vector3d vectorTrilinearInterpolation(
                    const std::vector<Eigen::MatrixXd>& u_grid,
                    const std::vector<Eigen::MatrixXd>& v_grid,
                    const std::vector<Eigen::MatrixXd>& w_grid,
                    double x, double y, double z,
                    const Eigen::Vector3d& min_coords,
                    const Eigen::Vector3d& max_coords);

            // ========================= 数值微分 =========================

            /**
             * @brief 有限差分近似
             */
            static double centralDifference(const std::vector<double>& y, int index, double h);
            static double forwardDifference(const std::vector<double>& y, int index, double h);
            static double backwardDifference(const std::vector<double>& y, int index, double h);

            /**
             * @brief 梯度计算
             */
            static Eigen::Vector2d gradient2D(const Eigen::MatrixXd& field, int i, int j,
                                              double dx, double dy);
            static Eigen::Vector3d gradient3D(const std::vector<Eigen::MatrixXd>& field,
                                              int i, int j, int k, double dx, double dy, double dz);

            /**
             * @brief 散度和旋度
             */
            static double divergence2D(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                       int i, int j, double dx, double dy);
            static double curl2D(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                 int i, int j, double dx, double dy);
            static Eigen::Vector3d curl3D(const std::vector<Eigen::MatrixXd>& u,
                                          const std::vector<Eigen::MatrixXd>& v,
                                          const std::vector<Eigen::MatrixXd>& w,
                                          int i, int j, int k, double dx, double dy, double dz);

            // ========================= 数值积分 =========================

            /**
             * @brief 一维积分
             */
            static double trapezoidalRule(const std::vector<double>& x, const std::vector<double>& y);
            static double simpsonsRule(const std::vector<double>& x, const std::vector<double>& y);
            static double gaussianQuadrature(const std::function<double(double)>& f,
                                             double a, double b, int n_points = 5);

            /**
             * @brief 多维积分
             */
            static double integrate2D(const Eigen::MatrixXd& field, double dx, double dy);
            static double integrate3D(const std::vector<Eigen::MatrixXd>& field,
                                      double dx, double dy, const std::vector<double>& dz);

            // ========================= 统计分析 =========================

            /**
             * @brief 基本统计量
             */
            static double mean(const std::vector<double>& data);
            static double variance(const std::vector<double>& data);
            static double standardDeviation(const std::vector<double>& data);
            static double median(std::vector<double> data);
            static std::pair<double, double> minMax(const std::vector<double>& data);

            /**
             * @brief 相关性分析
             */
            static double pearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y);
            static double spearmanCorrelation(const std::vector<double>& x, const std::vector<double>& y);
            static Eigen::MatrixXd covarianceMatrix(const std::vector<std::vector<double>>& data);

            /**
             * @brief 时间序列分析
             */
            static std::vector<double> movingAverage(const std::vector<double>& data, int window_size);
            static std::vector<double> exponentialSmoothing(const std::vector<double>& data, double alpha);
            static std::pair<std::vector<double>, std::vector<double>> linearRegression(
                    const std::vector<double>& x, const std::vector<double>& y);

            // ========================= 随机数生成 =========================

            /**
             * @brief 随机数生成器
             */
            class RandomGenerator {
            public:
                RandomGenerator(unsigned int seed = std::random_device{}());

                double uniform(double min = 0.0, double max = 1.0);
                double normal(double mean = 0.0, double stddev = 1.0);
                double exponential(double lambda = 1.0);
                double gamma(double alpha, double beta);

                Eigen::Vector3d uniformVector3D(const Eigen::Vector3d& min, const Eigen::Vector3d& max);
                Eigen::Vector3d normalVector3D(const Eigen::Vector3d& mean, const Eigen::Vector3d& stddev);

                std::vector<double> uniformArray(size_t size, double min = 0.0, double max = 1.0);
                std::vector<double> normalArray(size_t size, double mean = 0.0, double stddev = 1.0);

            private:
                std::mt19937 generator_;
            };

            // ========================= 优化算法 =========================

            /**
             * @brief 一维优化
             */
            static double goldenSectionSearch(const std::function<double(double)>& f,
                                              double a, double b, double tolerance = 1e-6);
            static double brentMethod(const std::function<double(double)>& f,
                                      double a, double b, double tolerance = 1e-6);

            /**
             * @brief 多维优化
             */
            static Eigen::VectorXd gradientDescent(const std::function<double(const Eigen::VectorXd&)>& f,
                                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& df,
                                                   const Eigen::VectorXd& x0,
                                                   double learning_rate = 0.01,
                                                   double tolerance = 1e-6,
                                                   int max_iterations = 1000);

            static Eigen::VectorXd newtonMethod(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f,
                                                const std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>& df,
                                                const Eigen::VectorXd& x0,
                                                double tolerance = 1e-6,
                                                int max_iterations = 100);

            // ========================= 海洋学专用函数 =========================

            /**
             * @brief 科里奥利参数
             */
            static double coriolisParameter(double latitude);
            static double betaParameter(double latitude);

            /**
             * @brief 海水物性参数
             */
            static double seawaterDensity(double temperature, double salinity, double pressure);
            static double seawaterSoundSpeed(double temperature, double salinity, double depth);
            static double seawaterViscosity(double temperature, double salinity);

            /**
             * @brief 海面高度异常计算
             */
            static double geostrophicVelocity(double ssh_gradient, double latitude);
            static Eigen::Vector2d geostrophicCurrents(const Eigen::MatrixXd& ssh,
                                                       int i, int j, double dx, double dy, double latitude);

            /**
             * @brief 潮汐调和分析
             */
            static std::vector<double> harmonicAnalysis(const std::vector<double>& time_series,
                                                        const std::vector<double>& frequencies);
            static double tidalConstituent(const std::vector<double>& time_series,
                                           const std::vector<double>& time,
                                           double frequency);

            /**
             * @brief 海洋涡度和应变率
             */
            static double relativeVorticity(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                            int i, int j, double dx, double dy);
            static double strainRate(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                     int i, int j, double dx, double dy);
            static double okuboWeissParameter(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                              int i, int j, double dx, double dy);

            // ========================= 矩阵运算扩展 =========================

            /**
             * @brief 矩阵工具
             */
            static Eigen::MatrixXd smoothMatrix(const Eigen::MatrixXd& matrix, int kernel_size = 3);
            static Eigen::MatrixXd gaussianFilter(const Eigen::MatrixXd& matrix, double sigma);
            static Eigen::MatrixXd sobelFilter(const Eigen::MatrixXd& matrix);

            /**
             * @brief 稀疏矩阵操作
             */
            static Eigen::SparseMatrix<double> createLaplacianMatrix(int nx, int ny, double dx, double dy);
            static Eigen::SparseMatrix<double> createAdvectionMatrix(const Eigen::MatrixXd& u,
                                                                     const Eigen::MatrixXd& v,
                                                                     double dx, double dy, double dt);

            // ========================= 频谱分析 =========================

            /**
             * @brief 傅里叶变换
             */
            static std::vector<std::complex<double>> fft(const std::vector<double>& signal);
            static std::vector<double> ifft(const std::vector<std::complex<double>>& spectrum);
            static std::vector<double> powerSpectrum(const std::vector<double>& signal);

            /**
             * @brief 小波变换
             */
            static std::vector<double> morletWavelet(const std::vector<double>& signal, double frequency);
            static Eigen::MatrixXd continuousWaveletTransform(const std::vector<double>& signal,
                                                              const std::vector<double>& scales);

            // ========================= 数值求解器辅助 =========================

            /**
             * @brief 线性系统求解
             */
            static Eigen::VectorXd solvePoissonEquation(const Eigen::MatrixXd& rhs,
                                                        double dx, double dy,
                                                        const std::string& boundary_condition = "dirichlet");

            static Eigen::VectorXd solveTridiagonal(const std::vector<double>& a,
                                                    const std::vector<double>& b,
                                                    const std::vector<double>& c,
                                                    const std::vector<double>& d);

            /**
             * @brief 迭代求解器
             */
            static bool jacobiIteration(Eigen::MatrixXd& x,
                                        const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& b,
                                        double tolerance = 1e-6,
                                        int max_iterations = 1000);

            static bool gaussSeidelIteration(Eigen::MatrixXd& x,
                                             const Eigen::MatrixXd& A,
                                             const Eigen::VectorXd& b,
                                             double tolerance = 1e-6,
                                             int max_iterations = 1000);

            static bool conjugateGradient(Eigen::VectorXd& x,
                                          const Eigen::SparseMatrix<double>& A,
                                          const Eigen::VectorXd& b,
                                          double tolerance = 1e-6,
                                          int max_iterations = 1000);

            // ========================= 几何计算 =========================

            /**
             * @brief 几何形状
             */
            static bool pointInPolygon(const Eigen::Vector2d& point,
                                       const std::vector<Eigen::Vector2d>& polygon);
            static double polygonArea(const std::vector<Eigen::Vector2d>& polygon);
            static Eigen::Vector2d polygonCentroid(const std::vector<Eigen::Vector2d>& polygon);

            /**
             * @brief 线段和射线
             */
            static bool lineIntersection(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
                                         const Eigen::Vector2d& p3, const Eigen::Vector2d& p4,
                                         Eigen::Vector2d& intersection);
            static double pointToLineDistance(const Eigen::Vector2d& point,
                                              const Eigen::Vector2d& line_start,
                                              const Eigen::Vector2d& line_end);

            // ========================= 数值稳定性检查 =========================

            /**
             * @brief 稳定性分析
             */
            static double computeCFLNumber(double velocity, double grid_spacing, double time_step);
            static double computeDiffusionNumber(double diffusivity, double grid_spacing, double time_step);
            static bool checkNumericalStability(double cfl, double diffusion_number);

            /**
             * @brief 误差分析
             */
            static double l2Norm(const Eigen::VectorXd& vector);
            static double l2Norm(const Eigen::MatrixXd& matrix);
            static double maxNorm(const Eigen::VectorXd& vector);
            static double maxNorm(const Eigen::MatrixXd& matrix);
            static double relativeError(const Eigen::VectorXd& exact, const Eigen::VectorXd& approximate);

            // ========================= 工具函数 =========================

            /**
             * @brief 角度转换
             */
            static double degreesToRadians(double degrees) { return degrees * PI / 180.0; }
            static double radiansToDegrees(double radians) { return radians * 180.0 / PI; }

            /**
             * @brief 数值比较
             */
            static bool isEqual(double a, double b, double tolerance = 1e-10);
            static bool isZero(double value, double tolerance = 1e-10);

            /**
             * @brief 限制和截断
             */
            static double clamp(double value, double min_val, double max_val);
            static Eigen::Vector3d clamp(const Eigen::Vector3d& vector,
                                         const Eigen::Vector3d& min_vals,
                                         const Eigen::Vector3d& max_vals);

            /**
             * @brief 安全数学运算
             */
            static double safeDivision(double numerator, double denominator, double default_value = 0.0);
            static double safeSqrt(double value);
            static double safeLog(double value, double min_value = 1e-10);

        private:
            // 私有辅助函数
            static void computeCubicSplineCoefficients(const std::vector<double>& x,
                                                       const std::vector<double>& y,
                                                       std::vector<double>& a,
                                                       std::vector<double>& b,
                                                       std::vector<double>& c,
                                                       std::vector<double>& d);

            static std::vector<double> gaussianQuadratureWeights(int n);
            static std::vector<double> gaussianQuadratureNodes(int n);
        };

/**
 * @brief 向量场分析工具
 */
        class VectorFieldAnalysis {
        public:
            /**
             * @brief 流线追踪
             */
            static std::vector<Eigen::Vector2d> traceStreamline(
                    const Eigen::Vector2d& start_point,
                    const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& velocity_field,
                    double step_size, int max_steps);

            /**
             * @brief 向量场拓扑分析
             */
            static std::vector<Eigen::Vector2d> findCriticalPoints(
                    const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& velocity_field,
                    const Eigen::Vector2d& search_min, const Eigen::Vector2d& search_max,
                    double tolerance = 1e-6);

            /**
             * @brief 路径积分
             */
            static double lineIntegral(const std::vector<Eigen::Vector2d>& path,
                                       const std::function<double(const Eigen::Vector2d&)>& scalar_field);

            static double circularIntegral(const std::vector<Eigen::Vector2d>& closed_path,
                                           const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& vector_field);
        };

    } // namespace Utils
} // namespace OceanSim