#ifndef VECTORIZED_OPERATIONS_H
#define VECTORIZED_OPERATIONS_H

#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

// 仅在 x86 平台包含SIMD头文件，避免在arm64(M1/M2/M3)上编译失败
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#  include <immintrin.h>  // AVX/SSE 指令集
#endif

#ifdef __ARM_NEON
#  include <arm_neon.h>  // ARM NEON指令集
#endif

namespace OceanSimulation {
    namespace Core {

/**
 * @brief 向量化运算模块
 * 
 * 提供高性能的SIMD向量化运算，专门针对海洋模拟中的大规模数值计算进行优化。
 * 支持AVX2、AVX-512、SSE和ARM NEON指令集。
 */
        class VectorizedOperations {
        public:
            /**
             * @brief SIMD指令集类型
             */
            enum class SimdType {
                None,        // 无SIMD支持
                SSE,         // SSE 4.1
                AVX,         // AVX
                AVX2,        // AVX2
                AVX512,      // AVX-512
                NEON         // ARM NEON
            };

            /**
             * @brief 向量运算配置
             */
            struct Config {
                SimdType preferredSimd = SimdType::AVX2;
                bool enableAutoAlignment = true;
                size_t alignmentBytes = 32;  // AVX2对齐要求
                bool enableBoundsCheck = true;
                bool enablePrefetch = true;
            };

            /**
             * @brief 性能计数器
             */
            struct PerformanceCounters {
                uint64_t vectorOperations = 0;
                uint64_t scalarOperations = 0;
                uint64_t alignmentMisses = 0;
                double simdEfficiency = 0.0;
            };

        private:
            Config config_;
            SimdType activeSimd_;
            PerformanceCounters counters_;

            // 检测可用的SIMD指令集
            SimdType detectSimdCapabilities() const;

            // 内存对齐检查
            bool isAligned(const void* ptr, size_t alignment) const;

            // 数据预取
            void prefetchData(const void* ptr, size_t size) const;

        public:
            /**
             * @brief 构造函数
             * @param config 向量化配置
             */
            explicit VectorizedOperations(const Config& config);

            /**
             * @brief 析构函数
             */
            ~VectorizedOperations() = default;

            // ===========================================
            // 基础向量运算
            // ===========================================

            /**
             * @brief 向量加法: result = a + b
             * @param a 输入向量A
             * @param b 输入向量B
             * @param result 输出向量
             * @param size 向量长度
             */
            void vectorAdd(const float* a, const float* b, float* result, size_t size);
            void vectorAdd(const double* a, const double* b, double* result, size_t size);

            /**
             * @brief 向量减法: result = a - b
             */
            void vectorSub(const float* a, const float* b, float* result, size_t size);
            void vectorSub(const double* a, const double* b, double* result, size_t size);

            /**
             * @brief 向量乘法: result = a * b
             */
            void vectorMul(const float* a, const float* b, float* result, size_t size);
            void vectorMul(const double* a, const double* b, double* result, size_t size);

            /**
             * @brief 向量除法: result = a / b
             */
            void vectorDiv(const float* a, const float* b, float* result, size_t size);
            void vectorDiv(const double* a, const double* b, double* result, size_t size);

            /**
             * @brief 标量乘法: result = a * scalar
             */
            void scalarMul(const float* a, float scalar, float* result, size_t size);
            void scalarMul(const double* a, double scalar, double* result, size_t size);

            // ===========================================
            // 高级数学运算
            // ===========================================

            /**
             * @brief 点积运算
             */
            float dotProduct(const float* a, const float* b, size_t size);
            double dotProduct(const double* a, const double* b, size_t size);

            /**
             * @brief 向量范数计算
             */
            float vectorNorm(const float* a, size_t size);
            double vectorNorm(const double* a, size_t size);

            /**
             * @brief 快速平方根倒数
             */
            void fastInvSqrt(const float* input, float* output, size_t size);

            /**
             * @brief 三角函数向量化
             */
            void vectorSin(const float* input, float* output, size_t size);
            void vectorCos(const float* input, float* output, size_t size);
            void vectorExp(const float* input, float* output, size_t size);
            void vectorLog(const float* input, float* output, size_t size);

            // ===========================================
            // 海洋模拟专用运算
            // ===========================================

            /**
             * @brief 流体速度场插值
             * @param u X方向速度分量
             * @param v Y方向速度分量
             * @param x 插值位置X坐标
             * @param y 插值位置Y坐标
             * @param gridWidth 网格宽度
             * @param gridHeight 网格高度
             * @param numParticles 粒子数量
             * @param resultU 插值后的U分量
             * @param resultV 插值后的V分量
             */
            void bilinearInterpolation(const float* u, const float* v,
                                       const float* x, const float* y,
                                       size_t gridWidth, size_t gridHeight,
                                       size_t numParticles,
                                       float* resultU, float* resultV);

            /**
             * @brief 龙格-库塔积分器（向量化版本）
             * @param positions 粒子位置数组
             * @param velocities 速度场
             * @param dt 时间步长
             * @param numParticles 粒子数量
             * @param newPositions 更新后的位置
             */
            void rungeKutta4Integration(const float* positions, const float* velocities,
                                        float dt, size_t numParticles, float* newPositions);

            /**
             * @brief 扩散方程求解器
             * @param concentration 浓度场
             * @param diffusivity 扩散系数
             * @param dx 空间步长X
             * @param dy 空间步长Y
             * @param dt 时间步长
             * @param width 网格宽度
             * @param height 网格高度
             * @param newConcentration 更新后的浓度场
             */
            void diffusionSolver(const float* concentration, float diffusivity,
                                 float dx, float dy, float dt,
                                 size_t width, size_t height,
                                 float* newConcentration);

            /**
             * @brief 梯度计算（中心差分）
             * @param field 标量场
             * @param width 网格宽度
             * @param height 网格高度
             * @param dx 空间步长X
             * @param dy 空间步长Y
             * @param gradX X方向梯度
             * @param gradY Y方向梯度
             */
            void computeGradient(const float* field, size_t width, size_t height,
                                 float dx, float dy, float* gradX, float* gradY);

            /**
             * @brief 散度计算
             * @param u X方向速度分量
             * @param v Y方向速度分量
             * @param width 网格宽度
             * @param height 网格高度
             * @param dx 空间步长X
             * @param dy 空间步长Y
             * @param divergence 散度结果
             */
            void computeDivergence(const float* u, const float* v,
                                   size_t width, size_t height,
                                   float dx, float dy, float* divergence);

            /**
             * @brief 涡度计算
             * @param u X方向速度分量
             * @param v Y方向速度分量
             * @param width 网格宽度
             * @param height 网格高度
             * @param dx 空间步长X
             * @param dy 空间步长Y
             * @param vorticity 涡度结果
             */
            void computeVorticity(const float* u, const float* v,
                                  size_t width, size_t height,
                                  float dx, float dy, float* vorticity);

            // ===========================================
            // 矩阵运算
            // ===========================================

            /**
             * @brief 矩阵乘法（向量化优化）
             * @param A 矩阵A
             * @param B 矩阵B
             * @param C 结果矩阵C
             * @param M 矩阵A的行数
             * @param N 矩阵A的列数（矩阵B的行数）
             * @param K 矩阵B的列数
             */
            void matrixMultiply(const float* A, const float* B, float* C,
                                size_t M, size_t N, size_t K);

            /**
             * @brief 矩阵转置
             */
            void matrixTranspose(const float* input, float* output,
                                 size_t rows, size_t cols);

            // ===========================================
            // 内存和性能管理
            // ===========================================

            /**
             * @brief 分配对齐内存
             * @param size 字节数
             * @param alignment 对齐字节数
             * @return 对齐的内存指针
             */
            void* alignedAlloc(size_t size, size_t alignment);

            /**
             * @brief 释放对齐内存
             * @param ptr 内存指针
             */
            void alignedFree(void* ptr);

            /**
             * @brief 获取当前SIMD类型
             */
            SimdType getCurrentSimdType() const { return activeSimd_; }

            /**
             * @brief 获取性能计数器
             */
            const PerformanceCounters& getPerformanceCounters() const { return counters_; }

            /**
             * @brief 重置性能计数器
             */
            void resetPerformanceCounters();

            /**
             * @brief 设置配置
             */
            void setConfig(const Config& config);

            /**
             * @brief 预热SIMD单元
             */
            void warmupSimd();

            /**
              * @brief 标量回退实现 - 向量乘法
              * @param a 输入向量A
              * @param b 输入向量B  
              * @param result 输出向量
              * @param size 向量长度
              */
            void vectorAddScalar(const float* a, const float* b, float* result, size_t size);
            void vectorMulScalar(const float* a, const float* b, float* result, size_t size);
            
            
        private:
            // AVX2实现
            void vectorAddAVX2(const float* a, const float* b, float* result, size_t size);
            void vectorMulAVX2(const float* a, const float* b, float* result, size_t size);

            // SSE实现
            void vectorAddSSE(const float* a, const float* b, float* result, size_t size);
            void vectorMulSSE(const float* a, const float* b, float* result, size_t size);

            
           
        };

    } // namespace Core
} // namespace OceanSimulation

#endif // VECTORIZED_OPERATIONS_H