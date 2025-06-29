// src/utils/MathUtils.cpp
#include "utils/MathUtils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <complex>
#include <random>
#include <functional>

namespace OceanSim {
    namespace Utils {

        // ========================= 坐标变换 =========================

        Eigen::Vector2d MathUtils::geographicToProjected(double latitude, double longitude,
                                                         const std::string& projection) {
            if (projection == "mercator") {
                double x = EARTH_RADIUS * degreesToRadians(longitude);
                double lat_rad = degreesToRadians(latitude);
                double y = EARTH_RADIUS * std::log(std::tan(PI/4.0 + lat_rad/2.0));
                return Eigen::Vector2d(x, y);
            }
            return Eigen::Vector2d::Zero();
        }

        Eigen::Vector2d MathUtils::projectedToGeographic(double x, double y,
                                                         const std::string& projection) {
            if (projection == "mercator") {
                double longitude = radiansToDegrees(x / EARTH_RADIUS);
                double lat_rad = 2.0 * std::atan(std::exp(y / EARTH_RADIUS)) - PI/2.0;
                double latitude = radiansToDegrees(lat_rad);
                return Eigen::Vector2d(latitude, longitude);
            }
            return Eigen::Vector2d::Zero();
        }

        Eigen::Vector3d MathUtils::sphericalToCartesian(double r, double theta, double phi) {
            double x = r * std::sin(phi) * std::cos(theta);
            double y = r * std::sin(phi) * std::sin(theta);
            double z = r * std::cos(phi);
            return Eigen::Vector3d(x, y, z);
        }

        Eigen::Vector3d MathUtils::cartesianToSpherical(const Eigen::Vector3d& cartesian) {
            double r = cartesian.norm();
            double theta = std::atan2(cartesian.y(), cartesian.x());
            double phi = std::acos(cartesian.z() / r);
            return Eigen::Vector3d(r, theta, phi);
        }

        double MathUtils::haversineDistance(double lat1, double lon1, double lat2, double lon2) {
            double dlat = degreesToRadians(lat2 - lat1);
            double dlon = degreesToRadians(lon2 - lon1);
            double a = std::sin(dlat/2) * std::sin(dlat/2) +
                       std::cos(degreesToRadians(lat1)) * std::cos(degreesToRadians(lat2)) *
                       std::sin(dlon/2) * std::sin(dlon/2);
            double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
            return EARTH_RADIUS * c;
        }

        // ========================= 插值算法 =========================

        double MathUtils::linearInterpolation(const std::vector<double>& x,
                                              const std::vector<double>& y, double xi) {
            if (x.size() != y.size() || x.size() < 2) return 0.0;

            auto it = std::lower_bound(x.begin(), x.end(), xi);
            if (it == x.begin()) return y[0];
            if (it == x.end()) return y.back();

            size_t i = std::distance(x.begin(), it) - 1;
            double t = (xi - x[i]) / (x[i+1] - x[i]);
            return y[i] + t * (y[i+1] - y[i]);
        }

        double MathUtils::bilinearInterpolation(const Eigen::MatrixXd& grid,
                                                double x, double y,
                                                double x_min, double x_max,
                                                double y_min, double y_max) {
            int rows = grid.rows();
            int cols = grid.cols();

            double dx = (x_max - x_min) / (cols - 1);
            double dy = (y_max - y_min) / (rows - 1);

            double fx = (x - x_min) / dx;
            double fy = (y - y_min) / dy;

            int i = static_cast<int>(fx);
            int j = static_cast<int>(fy);

            i = std::max(0, std::min(i, cols - 2));
            j = std::max(0, std::min(j, rows - 2));

            double tx = fx - i;
            double ty = fy - j;

            double v00 = grid(j, i);
            double v01 = grid(j, i+1);
            double v10 = grid(j+1, i);
            double v11 = grid(j+1, i+1);

            double v0 = v00 * (1 - tx) + v01 * tx;
            double v1 = v10 * (1 - tx) + v11 * tx;

            return v0 * (1 - ty) + v1 * ty;
        }

        // ========================= 数值微分 =========================

        double MathUtils::centralDifference(const std::vector<double>& y, int index, double h) {
            if (index <= 0 || index >= static_cast<int>(y.size()) - 1) return 0.0;
            return (y[index + 1] - y[index - 1]) / (2.0 * h);
        }

        Eigen::Vector2d MathUtils::gradient2D(const Eigen::MatrixXd& field, int i, int j,
                                              double dx, double dy) {
            int rows = field.rows();
            int cols = field.cols();

            double grad_x = 0.0, grad_y = 0.0;

            if (i > 0 && i < cols - 1) {
                grad_x = (field(j, i+1) - field(j, i-1)) / (2.0 * dx);
            }

            if (j > 0 && j < rows - 1) {
                grad_y = (field(j+1, i) - field(j-1, i)) / (2.0 * dy);
            }

            return Eigen::Vector2d(grad_x, grad_y);
        }

        double MathUtils::divergence2D(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                       int i, int j, double dx, double dy) {
            double dudx = 0.0, dvdy = 0.0;

            if (i > 0 && i < u.cols() - 1) {
                dudx = (u(j, i+1) - u(j, i-1)) / (2.0 * dx);
            }

            if (j > 0 && j < v.rows() - 1) {
                dvdy = (v(j+1, i) - v(j-1, i)) / (2.0 * dy);
            }

            return dudx + dvdy;
        }

        double MathUtils::curl2D(const Eigen::MatrixXd& u, const Eigen::MatrixXd& v,
                                 int i, int j, double dx, double dy) {
            double dvdx = 0.0, dudy = 0.0;

            if (i > 0 && i < v.cols() - 1) {
                dvdx = (v(j, i+1) - v(j, i-1)) / (2.0 * dx);
            }

            if (j > 0 && j < u.rows() - 1) {
                dudy = (u(j+1, i) - u(j-1, i)) / (2.0 * dy);
            }

            return dvdx - dudy;
        }

        // ========================= 数值积分 =========================

        double MathUtils::trapezoidalRule(const std::vector<double>& x, const std::vector<double>& y) {
            if (x.size() != y.size() || x.size() < 2) return 0.0;

            double integral = 0.0;
            for (size_t i = 1; i < x.size(); ++i) {
                integral += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
            }
            return integral;
        }

        double MathUtils::simpsonsRule(const std::vector<double>& x, const std::vector<double>& y) {
            if (x.size() != y.size() || x.size() < 3) return 0.0;

            double integral = 0.0;
            size_t n = x.size() - 1;

            if (n % 2 == 1) {  // 偶数个区间
                for (size_t i = 0; i < n; i += 2) {
                    double h = (x[i+2] - x[i]) / 2.0;
                    integral += h * (y[i] + 4*y[i+1] + y[i+2]) / 3.0;
                }
            }
            return integral;
        }

        // ========================= 统计分析 =========================

        double MathUtils::mean(const std::vector<double>& data) {
            if (data.empty()) return 0.0;
            return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        }

        double MathUtils::variance(const std::vector<double>& data) {
            if (data.size() < 2) return 0.0;

            double m = mean(data);
            double sum_sq_diff = 0.0;
            for (double x : data) {
                sum_sq_diff += (x - m) * (x - m);
            }
            return sum_sq_diff / (data.size() - 1);
        }

        double MathUtils::standardDeviation(const std::vector<double>& data) {
            return std::sqrt(variance(data));
        }

        double MathUtils::median(std::vector<double> data) {
            if (data.empty()) return 0.0;

            std::sort(data.begin(), data.end());
            size_t n = data.size();

            if (n % 2 == 0) {
                return (data[n/2 - 1] + data[n/2]) / 2.0;
            } else {
                return data[n/2];
            }
        }

        // ========================= 海洋学专用函数 =========================

        double MathUtils::coriolisParameter(double latitude) {
            return 2.0 * OMEGA_EARTH * std::sin(degreesToRadians(latitude));
        }

        double MathUtils::seawaterDensity(double temperature, double salinity, double pressure) {
            // UNESCO国际海水状态方程 (简化版)
            double t = temperature;
            double s = salinity;
            double p = pressure / 10000.0; // 转换为bar

            // 纯水密度
            double rho_w = 999.842594 + 6.793952e-2*t - 9.095290e-3*t*t +
                           1.001685e-4*t*t*t - 1.120083e-6*t*t*t*t + 6.536332e-9*t*t*t*t*t;

            // 盐度修正
            double A = 8.24493e-1 - 4.0899e-3*t + 7.6438e-5*t*t - 8.2467e-7*t*t*t + 5.3875e-9*t*t*t*t;
            double B = -5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t*t;
            double C = 4.8314e-4;

            double rho = rho_w + A*s + B*s*std::sqrt(s) + C*s*s;

            // 压力修正 (简化)
            double K = 19652.21 + 148.4206*t - 2.327105*t*t + 1.360477e-2*t*t*t - 5.155288e-5*t*t*t*t;
            K += s * (54.6746 - 0.603459*t + 1.09987e-2*t*t - 6.1670e-5*t*t*t);
            K += s * std::sqrt(s) * (7.944e-2 + 1.6483e-2*t - 5.3009e-4*t*t);

            rho = rho / (1.0 - p/K);

            return rho;
        }

        double MathUtils::geostrophicVelocity(double ssh_gradient, double latitude) {
            double f = coriolisParameter(latitude);
            if (std::abs(f) < 1e-10) return 0.0;
            return GRAVITY * ssh_gradient / f;
        }

        // ========================= 随机数生成器实现 =========================

        MathUtils::RandomGenerator::RandomGenerator(unsigned int seed) : generator_(seed) {}

        double MathUtils::RandomGenerator::uniform(double min, double max) {
            std::uniform_real_distribution<double> dist(min, max);
            return dist(generator_);
        }

        double MathUtils::RandomGenerator::normal(double mean, double stddev) {
            std::normal_distribution<double> dist(mean, stddev);
            return dist(generator_);
        }

        Eigen::Vector3d MathUtils::RandomGenerator::uniformVector3D(const Eigen::Vector3d& min,
                                                                    const Eigen::Vector3d& max) {
            return Eigen::Vector3d(
                    uniform(min.x(), max.x()),
                    uniform(min.y(), max.y()),
                    uniform(min.z(), max.z())
            );
        }

        // ========================= 优化算法 =========================

        double MathUtils::goldenSectionSearch(const std::function<double(double)>& f,
                                              double a, double b, double tolerance) {
            const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
            const double resphi = 2.0 - phi;

            double x1 = a + resphi * (b - a);
            double x2 = b - resphi * (b - a);
            double f1 = f(x1);
            double f2 = f(x2);

            while (std::abs(b - a) > tolerance) {
                if (f1 > f2) {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = b - resphi * (b - a);
                    f2 = f(x2);
                } else {
                    b = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = a + resphi * (b - a);
                    f1 = f(x1);
                }
            }

            return (a + b) / 2.0;
        }

        // ========================= 数值稳定性检查 =========================

        double MathUtils::computeCFLNumber(double velocity, double grid_spacing, double time_step) {
            return velocity * time_step / grid_spacing;
        }

        double MathUtils::computeDiffusionNumber(double diffusivity, double grid_spacing, double time_step) {
            return diffusivity * time_step / (grid_spacing * grid_spacing);
        }

        bool MathUtils::checkNumericalStability(double cfl, double diffusion_number) {
            return cfl <= 1.0 && diffusion_number <= 0.5;
        }

        // ========================= 工具函数 =========================

        bool MathUtils::isEqual(double a, double b, double tolerance) {
            return std::abs(a - b) < tolerance;
        }

        bool MathUtils::isZero(double value, double tolerance) {
            return std::abs(value) < tolerance;
        }

        double MathUtils::clamp(double value, double min_val, double max_val) {
            return std::max(min_val, std::min(value, max_val));
        }

        double MathUtils::safeDivision(double numerator, double denominator, double default_value) {
            if (std::abs(denominator) < 1e-15) {
                return default_value;
            }
            return numerator / denominator;
        }

        double MathUtils::safeSqrt(double value) {
            return std::sqrt(std::max(0.0, value));
        }

        double MathUtils::l2Norm(const Eigen::VectorXd& vector) {
            return vector.norm();
        }

        double MathUtils::l2Norm(const Eigen::MatrixXd& matrix) {
            return matrix.norm();
        }

        // ========================= 向量场分析工具实现 =========================

        std::vector<Eigen::Vector2d> VectorFieldAnalysis::traceStreamline(
                const Eigen::Vector2d& start_point,
                const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& velocity_field,
                double step_size, int max_steps) {

            std::vector<Eigen::Vector2d> streamline;
            streamline.push_back(start_point);

            Eigen::Vector2d current_point = start_point;

            for (int i = 0; i < max_steps; ++i) {
                Eigen::Vector2d velocity = velocity_field(current_point);

                if (velocity.norm() < 1e-10) break;

                // 使用四阶Runge-Kutta方法
                Eigen::Vector2d k1 = step_size * velocity_field(current_point);
                Eigen::Vector2d k2 = step_size * velocity_field(current_point + k1/2.0);
                Eigen::Vector2d k3 = step_size * velocity_field(current_point + k2/2.0);
                Eigen::Vector2d k4 = step_size * velocity_field(current_point + k3);

                current_point += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
                streamline.push_back(current_point);
            }

            return streamline;
        }

    } // namespace Utils
} // namespace OceanSim