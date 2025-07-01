#include "algorithms/VectorizedOperations.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>


#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace OceanSimulation {
    namespace Core {

        VectorizedOperations::VectorizedOperations(const Config& config)
                : config_(config), counters_{} {
            activeSimd_ = detectSimdCapabilities();

            // 如果检测到的SIMD能力低于配置要求，则降级使用
            if (activeSimd_ < config_.preferredSimd) {
                std::cout << "警告：系统SIMD能力低于配置要求，使用可用的最高级别" << std::endl;
            } else {
                activeSimd_ = config_.preferredSimd;
            }

            warmupSimd();
        }

        VectorizedOperations::SimdType VectorizedOperations::detectSimdCapabilities() const {
#ifdef __AVX512F__
            return SimdType::AVX512;
#elif defined(__AVX2__)
            return SimdType::AVX2;
#elif defined(__AVX__)
            return SimdType::AVX;
#elif defined(__SSE4_1__)
            return SimdType::SSE;
#elif defined(__ARM_NEON)
            return SimdType::NEON;
#else
            return SimdType::None;
#endif
        }

        bool VectorizedOperations::isAligned(const void* ptr, size_t alignment) const {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        }

        void VectorizedOperations::prefetchData(const void* ptr, size_t size) const {
            if (!config_.enablePrefetch) return;

            const char* data = static_cast<const char*>(ptr);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            for (size_t i = 0; i < size; i += 64) { // 按缓存行大小预取
                _mm_prefetch(data + i, _MM_HINT_T0);
            }
#elif defined(__aarch64__) || defined(__arm64__)
            for (size_t i = 0; i < size; i += 64) {
                __builtin_prefetch(data + i);
            }
#else
            (void)data;
            (void)size;
#endif
        }

// ===========================================
// 基础向量运算实现
// ===========================================

        void VectorizedOperations::vectorAdd(const float* a, const float* b, float* result, size_t size) {
            if (config_.enableBoundsCheck && (!a || !b || !result)) {
                throw std::invalid_argument("空指针参数");
            }

            if (config_.enablePrefetch) {
                prefetchData(a, size * sizeof(float));
                prefetchData(b, size * sizeof(float));
            }

            switch (activeSimd_) {
                case SimdType::AVX2:
                case SimdType::AVX512:
                    vectorAddAVX2(a, b, result, size);
                    break;
                case SimdType::AVX:
                case SimdType::SSE:
                    vectorAddSSE(a, b, result, size);
                    break;
                default:
                    vectorAddScalar(a, b, result, size);
                    break;
            }

            ++counters_.vectorOperations;
        }

        void VectorizedOperations::vectorAddAVX2(const float* a, const float* b, float* result, size_t size) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            const size_t simdSize = 8; // AVX2处理8个float
            const size_t simdEnd = (size / simdSize) * simdSize;

            // SIMD处理主体部分
            for (size_t i = 0; i < simdEnd; i += simdSize) {
                __m256 va = _mm256_load_ps(&a[i]);
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vresult = _mm256_add_ps(va, vb);
                _mm256_store_ps(&result[i], vresult);
            }

            // 处理剩余元素
            for (size_t i = simdEnd; i < size; ++i) {
                result[i] = a[i] + b[i];
            }
#else
            vectorAddScalar(a, b, result, size);
#endif
        }

        void VectorizedOperations::vectorAddSSE(const float* a, const float* b, float* result, size_t size) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            const size_t simdSize = 4; // SSE处理4个float
            const size_t simdEnd = (size / simdSize) * simdSize;

            for (size_t i = 0; i < simdEnd; i += simdSize) {
                __m128 va = _mm_load_ps(&a[i]);
                __m128 vb = _mm_load_ps(&b[i]);
                __m128 vresult = _mm_add_ps(va, vb);
                _mm_store_ps(&result[i], vresult);
            }

            for (size_t i = simdEnd; i < size; ++i) {
                result[i] = a[i] + b[i];
            }
#else
            vectorAddScalar(a, b, result, size);
#endif
        }

        void VectorizedOperations::vectorAddScalar(const float* a, const float* b, float* result, size_t size) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] + b[i];
            }
            ++counters_.scalarOperations;
        }

        void VectorizedOperations::vectorAdd(const double* a, const double* b, double* result, size_t size) {
            // 双精度版本实现
            switch (activeSimd_) {
                case SimdType::AVX2:
                case SimdType::AVX512: {
                    const size_t simdSize = 4; // AVX2处理4个double
                    const size_t simdEnd = (size / simdSize) * simdSize;

                    for (size_t i = 0; i < simdEnd; i += simdSize) {
                        __m256d va = _mm256_load_pd(&a[i]);
                        __m256d vb = _mm256_load_pd(&b[i]);
                        __m256d vresult = _mm256_add_pd(va, vb);
                        _mm256_store_pd(&result[i], vresult);
                    }

                    for (size_t i = simdEnd; i < size; ++i) {
                        result[i] = a[i] + b[i];
                    }
                    break;
                }
                default:
                    for (size_t i = 0; i < size; ++i) {
                        result[i] = a[i] + b[i];
                    }
                    break;
            }
#else
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] + b[i];
            }
#endif
            ++counters_.vectorOperations;
        }

        void VectorizedOperations::vectorMulScalar(const float* a, const float* b, float* result, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] * b[i];
            }
        }
        
        void VectorizedOperations::vectorMul(const float* a, const float* b, float* result, size_t size) {
            switch (activeSimd_) {
                case SimdType::AVX2:
                case SimdType::AVX512:
                    vectorMulAVX2(a, b, result, size);
                    break;
                case SimdType::AVX:
                case SimdType::SSE:
                    vectorMulSSE(a, b, result, size);
                    break;
                default:
                    vectorMulScalar(a, b, result, size);
                    break;
            }
            ++counters_.vectorOperations;
        }

        void VectorizedOperations::vectorMulAVX2(const float* a, const float* b, float* result, size_t size) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            const size_t simdSize = 8;
            const size_t simdEnd = (size / simdSize) * simdSize;

            for (size_t i = 0; i < simdEnd; i += simdSize) {
                __m256 va = _mm256_load_ps(&a[i]);
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vresult = _mm256_mul_ps(va, vb);
                _mm256_store_ps(&result[i], vresult);
            }

            for (size_t i = simdEnd; i < size; ++i) {
                result[i] = a[i] * b[i];
            }
#else
            vectorMulScalar(a, b, result, size);
#endif
        }

        void VectorizedOperations::vectorMulSSE(const float* a, const float* b, float* result, size_t size) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            const size_t simdSize = 4;
            const size_t simdEnd = (size / simdSize) * simdSize;

            for (size_t i = 0; i < simdEnd; i += simdSize) {
                __m128 va = _mm_load_ps(&a[i]);
                __m128 vb = _mm_load_ps(&b[i]);
                __m128 vresult = _mm_mul_ps(va, vb);
                _mm_store_ps(&result[i], vresult);
            }

            for (size_t i = simdEnd; i < size; ++i) {
                result[i] = a[i] * b[i];
            }
#else
            vectorMulScalar(a, b, result, size);
#endif
        }

        float VectorizedOperations::dotProduct(const float* a, const float* b, size_t size) {

            float result = 0.0f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
            switch (activeSimd_) {
                case SimdType::AVX2:
                case SimdType::AVX512: {
                    const size_t simdSize = 8;
                    const size_t simdEnd = (size / simdSize) * simdSize;
                    __m256 sum = _mm256_setzero_ps();

                    for (size_t i = 0; i < simdEnd; i += simdSize) {
                        __m256 va = _mm256_load_ps(&a[i]);
                        __m256 vb = _mm256_load_ps(&b[i]);
                        __m256 vmul = _mm256_mul_ps(va, vb);
                        sum = _mm256_add_ps(sum, vmul);
                    }

                    // 水平求和
                    __m128 lo = _mm256_castps256_ps128(sum);
                    __m128 hi = _mm256_extractf128_ps(sum, 1);
                    __m128 sum128 = _mm_add_ps(lo, hi);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    sum128 = _mm_hadd_ps(sum128, sum128);
                    result = _mm_cvtss_f32(sum128);

                    // 处理剩余元素
                    for (size_t i = simdEnd; i < size; ++i) {
                        result += a[i] * b[i];
                    }
                    break;
                }
                default:
                    for (size_t i = 0; i < size; ++i) {
                        result += a[i] * b[i];
                    }
                    break;
            }
#else
            for (size_t i = 0; i < size; ++i) {
                result += a[i] * b[i];
            }
#endif
            ++counters_.vectorOperations;
            return result;
        }

// ===========================================
// 海洋模拟专用运算实现
// ===========================================

        void VectorizedOperations::bilinearInterpolation(const float* u, const float* v,
                                                         const float* x, const float* y,
                                                         size_t gridWidth, size_t gridHeight,
                                                         size_t numParticles,
                                                         float* resultU, float* resultV) {
            const float invGridWidth = 1.0f / static_cast<float>(gridWidth - 1);
            const float invGridHeight = 1.0f / static_cast<float>(gridHeight - 1);

            for (size_t p = 0; p < numParticles; ++p) {
                float fx = x[p] * invGridWidth;
                float fy = y[p] * invGridHeight;

                // 确保在网格范围内
                fx = std::max(0.0f, std::min(fx, static_cast<float>(gridWidth - 1)));
                fy = std::max(0.0f, std::min(fy, static_cast<float>(gridHeight - 1)));

                int i0 = static_cast<int>(fx);
                int j0 = static_cast<int>(fy);
                int i1 = std::min(i0 + 1, static_cast<int>(gridWidth - 1));
                int j1 = std::min(j0 + 1, static_cast<int>(gridHeight - 1));

                float wx = fx - i0;
                float wy = fy - j0;

                // 双线性插值
                size_t idx00 = j0 * gridWidth + i0;
                size_t idx01 = j0 * gridWidth + i1;
                size_t idx10 = j1 * gridWidth + i0;
                size_t idx11 = j1 * gridWidth + i1;

                float u00 = u[idx00], u01 = u[idx01], u10 = u[idx10], u11 = u[idx11];
                float v00 = v[idx00], v01 = v[idx01], v10 = v[idx10], v11 = v[idx11];

                resultU[p] = u00 * (1 - wx) * (1 - wy) + u01 * wx * (1 - wy) +
                             u10 * (1 - wx) * wy + u11 * wx * wy;
                resultV[p] = v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) +
                             v10 * (1 - wx) * wy + v11 * wx * wy;
            }

            ++counters_.vectorOperations;
        }

        void VectorizedOperations::computeGradient(const float* field, size_t width, size_t height,
                                                   float dx, float dy, float* gradX, float* gradY) {
            const float invDx2 = 1.0f / (2.0f * dx);
            const float invDy2 = 1.0f / (2.0f * dy);

            // 处理内部点（使用中心差分）
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    size_t idx = j * width + i;

                    // X方向梯度
                    gradX[idx] = (field[idx + 1] - field[idx - 1]) * invDx2;

                    // Y方向梯度
                    gradY[idx] = (field[idx + width] - field[idx - width]) * invDy2;
                }
            }

            // 处理边界点（使用前向/后向差分）
            for (size_t j = 0; j < height; ++j) {
                for (size_t i = 0; i < width; ++i) {
                    if (i == 0 || i == width - 1 || j == 0 || j == height - 1) {
                        size_t idx = j * width + i;

                        // 边界处理
                        if (i == 0) {
                            gradX[idx] = (field[idx + 1] - field[idx]) / dx;
                        } else if (i == width - 1) {
                            gradX[idx] = (field[idx] - field[idx - 1]) / dx;
                        }

                        if (j == 0) {
                            gradY[idx] = (field[idx + width] - field[idx]) / dy;
                        } else if (j == height - 1) {
                            gradY[idx] = (field[idx] - field[idx - width]) / dy;
                        }
                    }
                }
            }

            ++counters_.vectorOperations;
        }

// ===========================================
// 内存管理实现
// ===========================================

        void* VectorizedOperations::alignedAlloc(size_t size, size_t alignment) {
#ifdef _WIN32
            return _aligned_malloc(size, alignment);
#else
            void* ptr = nullptr;
            if (posix_memalign(&ptr, alignment, size) != 0) {
                return nullptr;
            }
            return ptr;
#endif
        }

        void VectorizedOperations::alignedFree(void* ptr) {
            if (!ptr) return;

#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        void VectorizedOperations::resetPerformanceCounters() {
            counters_ = PerformanceCounters{};
        }

        void VectorizedOperations::warmupSimd() {
            // 预热SIMD单元，避免首次使用时的性能损失
            constexpr size_t warmupSize = 1024;
            std::vector<float> a(warmupSize, 1.0f);
            std::vector<float> b(warmupSize, 2.0f);
            std::vector<float> result(warmupSize);

            vectorAdd(a.data(), b.data(), result.data(), warmupSize);

            std::cout << "SIMD单元预热完成，当前使用: ";
            switch (activeSimd_) {
                case SimdType::AVX512: std::cout << "AVX-512"; break;
                case SimdType::AVX2: std::cout << "AVX2"; break;
                case SimdType::AVX: std::cout << "AVX"; break;
                case SimdType::SSE: std::cout << "SSE"; break;
                case SimdType::NEON: std::cout << "ARM NEON"; break;
                default: std::cout << "标量运算"; break;
            }
            std::cout << std::endl;
        }

// 剩余函数的简化实现（占位符）
        void VectorizedOperations::vectorSub(const float* a, const float* b, float* result, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        void VectorizedOperations::vectorDiv(const float* a, const float* b, float* result, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        void VectorizedOperations::scalarMul(const float* a, float scalar, float* result, size_t size) {
            for (size_t i = 0; i < size; ++i) {
                result[i] = a[i] * scalar;
            }
        }

    } // namespace Core
} // namespace OceanSimulation