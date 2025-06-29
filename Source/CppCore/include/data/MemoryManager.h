#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <string>

namespace OceanSimulation {
    namespace Core {

/**
 * @brief 高性能内存管理器
 * 
 * 专门为海洋模拟系统设计的内存管理器，提供内存池、对象缓存、
 * 内存对齐、垃圾回收和内存泄漏检测功能。
 */
        class MemoryManager {
        public:
            /**
             * @brief 内存分配策略
             */
            enum class AllocationStrategy {
                LinearPool,      // 线性内存池
                FreeListPool,    // 自由列表内存池
                BuddySystem,     // 伙伴系统
                SlabAllocator,   // Slab分配器
                ThreadLocal      // 线程本地分配器
            };

            /**
             * @brief 内存对齐类型
             */
            enum class AlignmentType {
                Byte1 = 1,
                Byte4 = 4,
                Byte8 = 8,
                Byte16 = 16,
                Byte32 = 32,      // AVX对齐
                Byte64 = 64,      // 缓存行对齐
                Byte128 = 128,    // 某些SIMD指令对齐
                PageAlign = 4096  // 页面对齐
            };

            /**
             * @brief 内存管理器配置
             */
            struct Config {
                AllocationStrategy strategy = AllocationStrategy::FreeListPool;
                size_t poolSize = 256 * 1024 * 1024;  // 256MB默认池大小
                size_t maxPoolCount = 16;             // 最大池数量
                AlignmentType defaultAlignment = AlignmentType::Byte32;
                bool enableMemoryTracking = true;    // 启用内存跟踪
                bool enableLeakDetection = true;     // 启用泄漏检测
                bool enableGarbageCollection = true; // 启用垃圾回收
                size_t gcThreshold = 100 * 1024 * 1024; // 100MB GC阈值
                bool enableThreadSafety = true;      // 启用线程安全
                bool enableStatistics = true;       // 启用统计信息
            };

            /**
             * @brief 内存块信息
             */
            struct MemoryBlock {
                void* ptr;
                size_t size;
                size_t alignment;
                std::chrono::system_clock::time_point allocTime;
                std::string allocLocation;
                bool inUse;

                MemoryBlock() : ptr(nullptr), size(0), alignment(0), inUse(false) {}
                MemoryBlock(void* p, size_t s, size_t a, const std::string& loc)
                        : ptr(p), size(s), alignment(a), allocLocation(loc), inUse(true) {
                    allocTime = std::chrono::system_clock::now();
                }
            };

            /**
             * @brief 内存池统计信息
             */
            struct PoolStats {
                size_t totalSize;
                size_t usedSize;
                size_t freeSize;
                size_t blockCount;
                size_t peakUsage;
                double fragmentationRatio;
                std::chrono::system_clock::time_point creationTime;
            };

            /**
             * @brief 全局内存统计信息
             */
            struct GlobalStats {
                std::atomic<size_t> totalAllocated{0};
                std::atomic<size_t> totalFreed{0};
                std::atomic<size_t> currentUsage{0};
                std::atomic<size_t> peakUsage{0};
                std::atomic<size_t> allocationCount{0};
                std::atomic<size_t> freeCount{0};
                std::atomic<size_t> reallocCount{0};
                std::atomic<size_t> leakCount{0};
                std::atomic<double> averageFragmentation{0.0};
            };

        private:
            Config config_;
            mutable std::mutex poolMutex_;
            mutable std::mutex statsMutex_;

            // 内存池管理
            std::vector<std::unique_ptr<uint8_t[]>> memoryPools_;
            std::vector<PoolStats> poolStats_;
            std::unordered_map<void*, size_t> pointerToPool_;

            // 空闲块管理
            struct FreeBlock {
                size_t size;
                FreeBlock* next;
            };
            std::vector<FreeBlock*> freeListHeads_;

            // 内存跟踪
            std::unordered_map<void*, MemoryBlock> allocatedBlocks_;
            GlobalStats globalStats_;

            // 垃圾回收
            std::atomic<bool> gcEnabled_{true};
            std::chrono::system_clock::time_point lastGcTime_;

            // 线程本地存储
            thread_local static MemoryManager* threadLocalManager_;

            // 内部实现方法
            void* allocateFromPool(size_t size, size_t alignment, size_t poolIndex);
            void* allocateNewPool(size_t size, size_t alignment);
            bool deallocateFromPool(void* ptr, size_t poolIndex);
            size_t findBestFitPool(size_t size) const;
            size_t createNewPool(size_t minSize);
            void updatePoolStats(size_t poolIndex);
            void performGarbageCollection();

            // 对齐计算
            size_t alignSize(size_t size, size_t alignment) const;
            void* alignPointer(void* ptr, size_t alignment) const;
            bool isAligned(void* ptr, size_t alignment) const;

        public:
            /**
             * @brief 构造函数
             * @param config 内存管理器配置
             */
            explicit MemoryManager(const Config& config = Config{});

            /**
             * @brief 析构函数
             */
            ~MemoryManager();

            /**
             * @brief 获取单例实例
             * @return 全局内存管理器实例
             */
            static MemoryManager& getInstance();

            // ===========================================
            // 基础内存分配接口
            // ===========================================

            /**
             * @brief 分配内存
             * @param size 要分配的字节数
             * @param alignment 内存对齐要求
             * @param location 分配位置（用于调试）
             * @return 分配的内存指针，失败返回nullptr
             */
            void* allocate(size_t size, AlignmentType alignment = AlignmentType::Byte32,
                           const std::string& location = "");

            /**
             * @brief 释放内存
             * @param ptr 要释放的内存指针
             */
            void deallocate(void* ptr);

            /**
             * @brief 重新分配内存
             * @param ptr 原内存指针
             * @param newSize 新大小
             * @param alignment 内存对齐要求
             * @return 重新分配的内存指针
             */
            void* reallocate(void* ptr, size_t newSize,
                             AlignmentType alignment = AlignmentType::Byte32);

            /**
             * @brief 分配对齐内存
             * @param size 字节数
             * @param alignment 对齐字节数
             * @return 对齐的内存指针
             */
            void* allocateAligned(size_t size, size_t alignment);

            /**
             * @brief 释放对齐内存
             * @param ptr 内存指针
             */
            void deallocateAligned(void* ptr);

            // ===========================================
            // 类型化内存分配接口
            // ===========================================

            /**
             * @brief 分配类型化数组
             * @tparam T 数据类型
             * @param count 元素数量
             * @param alignment 内存对齐要求
             * @return 类型化指针
             */
            template<typename T>
            T* allocateArray(size_t count, AlignmentType alignment = AlignmentType::Byte32);

            /**
             * @brief 释放类型化数组
             * @tparam T 数据类型
             * @param ptr 数组指针
             * @param count 元素数量
             */
            template<typename T>
            void deallocateArray(T* ptr, size_t count);

            /**
             * @brief 创建对象实例
             * @tparam T 对象类型
             * @tparam Args 构造参数类型
             * @param args 构造参数
             * @return 对象指针
             */
            template<typename T, typename... Args>
            T* createObject(Args&&... args);

            /**
             * @brief 销毁对象实例
             * @tparam T 对象类型
             * @param ptr 对象指针
             */
            template<typename T>
            void destroyObject(T* ptr);

            // ===========================================
            // 智能指针接口
            // ===========================================

            /**
             * @brief 创建自定义删除器的unique_ptr
             * @tparam T 数据类型
             * @param size 字节数
             * @param alignment 内存对齐要求
             * @return unique_ptr实例
             */
            template<typename T>
            std::unique_ptr<T, std::function<void(T*)>>
            makeUniquePtr(size_t size = sizeof(T),
                          AlignmentType alignment = AlignmentType::Byte32);

            /**
             * @brief 创建自定义删除器的shared_ptr
             * @tparam T 数据类型
             * @param size 字节数
             * @param alignment 内存对齐要求
             * @return shared_ptr实例
             */
            template<typename T>
            std::shared_ptr<T> makeSharedPtr(size_t size = sizeof(T),
                                             AlignmentType alignment = AlignmentType::Byte32);

            // ===========================================
            // 内存池管理接口
            // ===========================================

            /**
             * @brief 预分配内存池
             * @param poolSize 池大小
             * @param blockSize 块大小
             * @return 池索引
             */
            size_t preallocatePool(size_t poolSize, size_t blockSize = 0);

            /**
             * @brief 释放内存池
             * @param poolIndex 池索引
             */
            void releasePool(size_t poolIndex);

            /**
             * @brief 清理所有内存池
             */
            void clearAllPools();

            /**
             * @brief 压缩内存池（整理碎片）
             * @param poolIndex 池索引，-1表示所有池
             */
            void compactPools(int poolIndex = -1);

            // ===========================================
            // 垃圾回收接口
            // ===========================================

            /**
             * @brief 手动触发垃圾回收
             */
            void runGarbageCollection();

            /**
             * @brief 设置垃圾回收阈值
             * @param threshold 内存使用阈值（字节）
             */
            void setGcThreshold(size_t threshold);

            /**
             * @brief 启用/禁用自动垃圾回收
             * @param enabled 是否启用
             */
            void setGcEnabled(bool enabled);

            // ===========================================
            // 内存检测和调试接口
            // ===========================================

            /**
             * @brief 检测内存泄漏
             * @return 泄漏的内存块信息
             */
            std::vector<MemoryBlock> detectMemoryLeaks() const;

            /**
             * @brief 验证内存完整性
             * @param ptr 内存指针
             * @return 内存是否完整
             */
            bool validateMemory(void* ptr) const;

            /**
             * @brief 获取内存块信息
             * @param ptr 内存指针
             * @return 内存块信息
             */
            MemoryBlock getBlockInfo(void* ptr) const;

            /**
             * @brief 检查内存对齐
             * @param ptr 内存指针
             * @param alignment 对齐要求
             * @return 是否正确对齐
             */
            bool checkAlignment(void* ptr, size_t alignment) const;

            /**
             * @brief 打印内存使用报告
             */
            void printMemoryReport() const;

            /**
             * @brief 导出内存使用统计
             * @param filename 输出文件名
             */
            void exportMemoryStats(const std::string& filename) const;

            // ===========================================
            // 统计信息接口
            // ===========================================

            /**
             * @brief 获取全局统计信息
             * @return 统计信息结构
             */
            GlobalStats getGlobalStats() const;

            /**
             * @brief 获取内存池统计信息
             * @param poolIndex 池索引
             * @return 池统计信息
             */
            PoolStats getPoolStats(size_t poolIndex) const;

            /**
             * @brief 重置统计信息
             */
            void resetStats();

            /**
             * @brief 获取当前内存使用量
             * @return 使用的字节数
             */
            size_t getCurrentUsage() const;

            /**
             * @brief 获取峰值内存使用量
             * @return 峰值字节数
             */
            size_t getPeakUsage() const;

            /**
             * @brief 获取内存碎片化率
             * @return 碎片化百分比
             */
            double getFragmentationRatio() const;

            // ===========================================
            // 配置管理接口
            // ===========================================

            /**
             * @brief 更新配置
             * @param newConfig 新配置
             */
            void updateConfig(const Config& newConfig);

            /**
             * @brief 获取当前配置
             * @return 配置信息
             */
            const Config& getConfig() const;

            /**
             * @brief 设置默认对齐方式
             * @param alignment 对齐类型
             */
            void setDefaultAlignment(AlignmentType alignment);

            // ===========================================
            // 高级功能接口
            // ===========================================

            /**
             * @brief 内存预取提示
             * @param ptr 内存地址
             * @param size 预取大小
             */
            void prefetchMemory(void* ptr, size_t size) const;

            /**
             * @brief 设置内存访问模式提示
             * @param ptr 内存地址
             * @param size 内存大小
             * @param sequential 是否顺序访问
             */
            void setAccessPattern(void* ptr, size_t size, bool sequential) const;

            /**
             * @brief 锁定内存页面（防止交换）
             * @param ptr 内存地址
             * @param size 内存大小
             * @return 锁定是否成功
             */
            bool lockMemory(void* ptr, size_t size) const;

            /**
             * @brief 解锁内存页面
             * @param ptr 内存地址
             * @param size 内存大小
             */
            void unlockMemory(void* ptr, size_t size) const;

        private:
            // 禁用拷贝构造和赋值
            MemoryManager(const MemoryManager&) = delete;
            MemoryManager& operator=(const MemoryManager&) = delete;

            // 内部辅助函数
            void initializePools();
            void cleanupPools();
            void updateGlobalStats(size_t allocated, bool isAllocation);
            void recordAllocation(void* ptr, size_t size, size_t alignment, const std::string& location);
            void recordDeallocation(void* ptr);
            size_t calculateOptimalPoolSize(size_t requestedSize) const;
            void mergeFreeBlocks(size_t poolIndex);
            bool isValidPointer(void* ptr) const;
        };

// ===========================================
// 模板方法实现
// ===========================================

        template<typename T>
        T* MemoryManager::allocateArray(size_t count, AlignmentType alignment) {
            size_t totalSize = count * sizeof(T);
            void* ptr = allocate(totalSize, alignment, typeid(T).name());
            return static_cast<T*>(ptr);
        }

        template<typename T>
        void MemoryManager::deallocateArray(T* ptr, size_t count) {
            if (ptr) {
                // 调用析构函数（如果需要）
                if (!std::is_trivially_destructible_v<T>) {
                    for (size_t i = 0; i < count; ++i) {
                        ptr[i].~T();
                    }
                }
                deallocate(ptr);
            }
        }

        template<typename T, typename... Args>
        T* MemoryManager::createObject(Args&&... args) {
            void* ptr = allocate(sizeof(T), AlignmentType::Byte16, typeid(T).name());
            if (ptr) {
                try {
                    return new(ptr) T(std::forward<Args>(args)...);
                }
                catch (...) {
                    deallocate(ptr);
                    throw;
                }
            }
            return nullptr;
        }

        template<typename T>
        void MemoryManager::destroyObject(T* ptr) {
            if (ptr) {
                ptr->~T();
                deallocate(ptr);
            }
        }

        template<typename T>
        std::unique_ptr<T, std::function<void(T*)>>
        MemoryManager::makeUniquePtr(size_t size, AlignmentType alignment) {
            void* ptr = allocate(size, alignment, typeid(T).name());
            if (!ptr) return nullptr;

            auto deleter = [this](T* p) {
                if (p) {
                    p->~T();
                    deallocate(p);
                }
            };

            return std::unique_ptr<T, std::function<void(T*)>>(
                    static_cast<T*>(ptr), deleter);
        }

        template<typename T>
        std::shared_ptr<T> MemoryManager::makeSharedPtr(size_t size, AlignmentType alignment) {
            void* ptr = allocate(size, alignment, typeid(T).name());
            if (!ptr) return nullptr;

            auto deleter = [this](T* p) {
                if (p) {
                    p->~T();
                    deallocate(p);
                }
            };

            return std::shared_ptr<T>(static_cast<T*>(ptr), deleter);
        }

// ===========================================
// 便利宏定义
// ===========================================

#define OCEAN_ALLOC(size) \
    MemoryManager::getInstance().allocate(size, MemoryManager::AlignmentType::Byte32, \
                                         __FILE__ ":" + std::to_string(__LINE__))

#define OCEAN_FREE(ptr) \
    MemoryManager::getInstance().deallocate(ptr)

#define OCEAN_NEW(Type, ...) \
    MemoryManager::getInstance().createObject<Type>(__VA_ARGS__)

#define OCEAN_DELETE(ptr) \
    MemoryManager::getInstance().destroyObject(ptr)

    } // namespace Core
} // namespace OceanSimulation

#endif // MEMORY_MANAGER_H