#include "data/MemoryManager.h"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cassert>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace OceanSimulation {
    namespace Core {

// 线程本地存储
        thread_local MemoryManager* MemoryManager::threadLocalManager_ = nullptr;

        MemoryManager::MemoryManager(const Config& config) : config_(config) {
            initializePools();
            lastGcTime_ = std::chrono::system_clock::now();

            std::cout << "内存管理器初始化完成:" << std::endl;
            std::cout << "  策略: ";
            switch (config_.strategy) {
                case AllocationStrategy::LinearPool: std::cout << "线性内存池"; break;
                case AllocationStrategy::FreeListPool: std::cout << "自由列表内存池"; break;
                case AllocationStrategy::BuddySystem: std::cout << "伙伴系统"; break;
                case AllocationStrategy::SlabAllocator: std::cout << "Slab分配器"; break;
                case AllocationStrategy::ThreadLocal: std::cout << "线程本地分配器"; break;
            }
            std::cout << std::endl;
            std::cout << "  池大小: " << config_.poolSize / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  最大池数: " << config_.maxPoolCount << std::endl;
            std::cout << "  默认对齐: " << static_cast<int>(config_.defaultAlignment) << " 字节" << std::endl;
        }

        MemoryManager::~MemoryManager() {
            if (config_.enableLeakDetection) {
                auto leaks = detectMemoryLeaks();
                if (!leaks.empty()) {
                    std::cerr << "检测到内存泄漏: " << leaks.size() << " 个块" << std::endl;
                    for (const auto& leak : leaks) {
                        std::cerr << "  地址: " << leak.ptr
                                  << ", 大小: " << leak.size
                                  << ", 位置: " << leak.allocLocation << std::endl;
                    }
                }
            }

            cleanupPools();

            if (config_.enableStatistics) {
                printMemoryReport();
            }
        }

        MemoryManager& MemoryManager::getInstance() {
            static MemoryManager instance;
            return instance;
        }

        void MemoryManager::initializePools() {
            std::lock_guard<std::mutex> lock(poolMutex_);

            // 创建初始内存池
            size_t initialPoolIndex = createNewPool(config_.poolSize);
            if (initialPoolIndex == SIZE_MAX) {
                throw std::bad_alloc();
            }

            std::cout << "创建初始内存池: " << config_.poolSize / (1024 * 1024) << " MB" << std::endl;
        }

        void MemoryManager::cleanupPools() {
            std::lock_guard<std::mutex> lock(poolMutex_);

            // 清理所有内存池
            memoryPools_.clear();
            poolStats_.clear();
            pointerToPool_.clear();
            freeListHeads_.clear();
        }

        void* MemoryManager::allocate(size_t size, AlignmentType alignment, const std::string& location) {
            if (size == 0) return nullptr;

            auto startTime = std::chrono::high_resolution_clock::now();

            // 对齐大小
            size_t alignedSize = alignSize(size, static_cast<size_t>(alignment));
            size_t actualAlignment = static_cast<size_t>(alignment);

            void* ptr = nullptr;

            if (config_.enableThreadSafety) {
                std::lock_guard<std::mutex> lock(poolMutex_);
                ptr = allocateFromPools(alignedSize, actualAlignment);
            } else {
                ptr = allocateFromPools(alignedSize, actualAlignment);
            }

            if (ptr && config_.enableMemoryTracking) {
                recordAllocation(ptr, size, actualAlignment, location);
            }

            if (ptr) {
                updateGlobalStats(size, true);

                // 检查是否需要垃圾回收
                if (config_.enableGarbageCollection &&
                    globalStats_.currentUsage > config_.gcThreshold) {
                    performGarbageCollection();
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

            if (config_.enableStatistics && duration.count() > 1000) { // 超过1ms记录
                std::cout << "内存分配耗时: " << duration.count() << " μs, 大小: " << size << std::endl;
            }

            return ptr;
        }

        void* MemoryManager::allocateFromPools(size_t size, size_t alignment) {
            // 尝试从现有池分配
            for (size_t i = 0; i < memoryPools_.size(); ++i) {
                void* ptr = allocateFromPool(size, alignment, i);
                if (ptr) {
                    updatePoolStats(i);
                    return ptr;
                }
            }

            // 创建新池
            if (memoryPools_.size() < config_.maxPoolCount) {
                size_t newPoolSize = std::max(config_.poolSize, size * 2);
                size_t newPoolIndex = createNewPool(newPoolSize);
                if (newPoolIndex != SIZE_MAX) {
                    void* ptr = allocateFromPool(size, alignment, newPoolIndex);
                    if (ptr) {
                        updatePoolStats(newPoolIndex);
                        return ptr;
                    }
                }
            }

            // 所有池都满了，使用系统分配器
            return allocateAligned(size, alignment);
        }

        void* MemoryManager::allocateFromPool(size_t size, size_t alignment, size_t poolIndex) {
            if (poolIndex >= memoryPools_.size()) return nullptr;

            uint8_t* poolStart = memoryPools_[poolIndex].get();
            size_t poolSize = config_.poolSize;

            // 简化的自由列表分配算法
            FreeBlock** current = &freeListHeads_[poolIndex];
            FreeBlock* prev = nullptr;

            while (*current) {
                if ((*current)->size >= size) {
                    void* ptr = reinterpret_cast<void*>(*current);

                    // 检查对齐
                    void* alignedPtr = alignPointer(ptr, alignment);
                    size_t offset = static_cast<uint8_t*>(alignedPtr) - static_cast<uint8_t*>(ptr);

                    if ((*current)->size >= size + offset) {
                        // 移除当前块
                        FreeBlock* allocatedBlock = *current;
                        *current = allocatedBlock->next;

                        // 如果有剩余空间，创建新的自由块
                        if (allocatedBlock->size > size + offset + sizeof(FreeBlock)) {
                            FreeBlock* newFree = reinterpret_cast<FreeBlock*>(
                                    static_cast<uint8_t*>(alignedPtr) + size);
                            newFree->size = allocatedBlock->size - size - offset;
                            newFree->next = *current;
                            *current = newFree;
                        }

                        pointerToPool_[alignedPtr] = poolIndex;
                        return alignedPtr;
                    }
                }

                prev = *current;
                current = &((*current)->next);
            }

            return nullptr;
        }

        void MemoryManager::deallocate(void* ptr) {
            if (!ptr) return;

            if (config_.enableMemoryTracking) {
                recordDeallocation(ptr);
            }

            // 查找指针所属的池
            auto it = pointerToPool_.find(ptr);
            if (it != pointerToPool_.end()) {
                if (config_.enableThreadSafety) {
                    std::lock_guard<std::mutex> lock(poolMutex_);
                    deallocateFromPool(ptr, it->second);
                } else {
                    deallocateFromPool(ptr, it->second);
                }
                pointerToPool_.erase(it);
            } else {
                // 使用系统释放器
                deallocateAligned(ptr);
            }

            auto blockIt = allocatedBlocks_.find(ptr);
            if (blockIt != allocatedBlocks_.end()) {
                updateGlobalStats(blockIt->second.size, false);
                allocatedBlocks_.erase(blockIt);
            }
        }

        bool MemoryManager::deallocateFromPool(void* ptr, size_t poolIndex) {
            if (poolIndex >= freeListHeads_.size()) return false;

            // 获取块大小（需要在分配时存储）
            auto blockIt = allocatedBlocks_.find(ptr);
            if (blockIt == allocatedBlocks_.end()) return false;

            size_t blockSize = blockIt->second.size;

            // 创建新的自由块
            FreeBlock* newFree = static_cast<FreeBlock*>(ptr);
            newFree->size = blockSize;
            newFree->next = freeListHeads_[poolIndex];
            freeListHeads_[poolIndex] = newFree;

            // 尝试合并相邻的自由块
            mergeFreeBlocks(poolIndex);
            updatePoolStats(poolIndex);

            return true;
        }

        size_t MemoryManager::createNewPool(size_t minSize) {
            try {
                auto newPool = std::make_unique<uint8_t[]>(minSize);

                // 初始化自由列表
                FreeBlock* initialBlock = reinterpret_cast<FreeBlock*>(newPool.get());
                initialBlock->size = minSize;
                initialBlock->next = nullptr;

                memoryPools_.push_back(std::move(newPool));
                freeListHeads_.push_back(initialBlock);

                // 初始化池统计
                PoolStats stats;
                stats.totalSize = minSize;
                stats.usedSize = 0;
                stats.freeSize = minSize;
                stats.blockCount = 0;
                stats.peakUsage = 0;
                stats.fragmentationRatio = 0.0;
                stats.creationTime = std::chrono::system_clock::now();

                poolStats_.push_back(stats);

                return memoryPools_.size() - 1;
            }
            catch (const std::bad_alloc&) {
                return SIZE_MAX;
            }
        }

        void MemoryManager::updatePoolStats(size_t poolIndex) {
            if (poolIndex >= poolStats_.size()) return;

            PoolStats& stats = poolStats_[poolIndex];

            // 计算使用情况
            size_t freeSize = 0;
            size_t blockCount = 0;

            FreeBlock* current = freeListHeads_[poolIndex];
            while (current) {
                freeSize += current->size;
                ++blockCount;
                current = current->next;
            }

            stats.freeSize = freeSize;
            stats.usedSize = stats.totalSize - freeSize;
            stats.blockCount = blockCount;

            if (stats.usedSize > stats.peakUsage) {
                stats.peakUsage = stats.usedSize;
            }

            // 计算碎片化率
            if (stats.totalSize > 0) {
                stats.fragmentationRatio = static_cast<double>(blockCount) / stats.totalSize * 1000;
            }
        }

        void MemoryManager::mergeFreeBlocks(size_t poolIndex) {
            if (poolIndex >= freeListHeads_.size()) return;

            // 简化的合并算法：按地址排序后合并相邻块
            std::vector<FreeBlock*> blocks;

            FreeBlock* current = freeListHeads_[poolIndex];
            while (current) {
                blocks.push_back(current);
                current = current->next;
            }

            if (blocks.size() <= 1) return;

            // 按地址排序
            std::sort(blocks.begin(), blocks.end());

            // 合并相邻块
            for (size_t i = 0; i < blocks.size() - 1; ++i) {
                uint8_t* currentEnd = reinterpret_cast<uint8_t*>(blocks[i]) + blocks[i]->size;
                uint8_t* nextStart = reinterpret_cast<uint8_t*>(blocks[i + 1]);

                if (currentEnd == nextStart) {
                    // 合并块
                    blocks[i]->size += blocks[i + 1]->size;
                    blocks.erase(blocks.begin() + i + 1);
                    --i; // 重新检查当前位置
                }
            }

            // 重建自由列表
            freeListHeads_[poolIndex] = blocks.empty() ? nullptr : blocks[0];
            for (size_t i = 0; i < blocks.size() - 1; ++i) {
                blocks[i]->next = blocks[i + 1];
            }
            if (!blocks.empty()) {
                blocks.back()->next = nullptr;
            }
        }

        void* MemoryManager::allocateAligned(size_t size, size_t alignment) {
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

        void MemoryManager::deallocateAligned(void* ptr) {
            if (!ptr) return;

#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }

        size_t MemoryManager::alignSize(size_t size, size_t alignment) const {
            return (size + alignment - 1) & ~(alignment - 1);
        }

        void* MemoryManager::alignPointer(void* ptr, size_t alignment) const {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
            return reinterpret_cast<void*>(aligned);
        }

        bool MemoryManager::isAligned(void* ptr, size_t alignment) const {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        }

        void MemoryManager::recordAllocation(void* ptr, size_t size, size_t alignment, const std::string& location) {
            if (config_.enableThreadSafety) {
                std::lock_guard<std::mutex> lock(statsMutex_);
            }

            allocatedBlocks_[ptr] = MemoryBlock(ptr, size, alignment, location);
            ++globalStats_.allocationCount;
        }

        void MemoryManager::recordDeallocation(void* ptr) {
            if (config_.enableThreadSafety) {
                std::lock_guard<std::mutex> lock(statsMutex_);
            }

            auto it = allocatedBlocks_.find(ptr);
            if (it != allocatedBlocks_.end()) {
                allocatedBlocks_.erase(it);
                ++globalStats_.freeCount;
            }
        }

        void MemoryManager::updateGlobalStats(size_t size, bool isAllocation) {
            if (isAllocation) {
                globalStats_.totalAllocated += size;
                globalStats_.currentUsage += size;

                if (globalStats_.currentUsage > globalStats_.peakUsage) {
                    globalStats_.peakUsage = globalStats_.currentUsage.load();
                }
            } else {
                globalStats_.totalFreed += size;
                globalStats_.currentUsage -= size;
            }
        }

        void MemoryManager::performGarbageCollection() {
            if (!gcEnabled_) return;

            auto now = std::chrono::system_clock::now();
            auto timeSinceLastGc = std::chrono::duration_cast<std::chrono::seconds>(now - lastGcTime_);

            if (timeSinceLastGc.count() < 60) return; // 最小间隔1分钟

            std::cout << "执行垃圾回收..." << std::endl;

            // 合并所有池中的自由块
            for (size_t i = 0; i < memoryPools_.size(); ++i) {
                mergeFreeBlocks(i);
                updatePoolStats(i);
            }
        
        lastGcTime_ = now;
    }

} // namespace Core
} // namespace OceanSimulation