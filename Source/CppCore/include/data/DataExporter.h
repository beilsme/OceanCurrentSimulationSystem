#ifndef DATA_EXPORTER_H
#define DATA_EXPORTER_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <chrono>


namespace OceanSimulation {
    namespace Core {

/**
 * @brief 数据导出器类
 * 
 * 负责将海洋模拟计算结果导出为多种格式，包括NetCDF、HDF5、CSV、JSON等。
 * 支持并行导出、数据压缩和元数据管理。
 */
        class DataExporter {
        public:
            /**
             * @brief 支持的导出格式
             */
            enum class ExportFormat {
                NetCDF,          // 科学计算标准格式
                HDF5,            // 层次化数据格式
                CSV,             // 逗号分隔值
                JSON,            // JavaScript对象表示法
                Binary,          // 二进制格式
                VTK,             // 可视化工具格式
                MATLAB,          // MATLAB格式
                ParaView         // ParaView格式
            };

            /**
             * @brief 数据压缩级别
             */
            enum class CompressionLevel {
                None = 0,
                Low = 1,
                Medium = 5,
                High = 9
            };

            /**
             * @brief 导出配置参数
             */
            struct ExportConfig {
                ExportFormat format = ExportFormat::NetCDF;
                CompressionLevel compression = CompressionLevel::Medium;
                bool includeMetadata = true;
                bool enableParallelIO = true;
                size_t bufferSize = 1024 * 1024; // 1MB缓冲区
                std::string outputDirectory = "./output/";
                std::string filePrefix = "ocean_simulation";
                bool timestampFiles = true;
                bool overwriteExisting = false;
            };

            /**
             * @brief 数据集信息
             */
            struct DatasetInfo {
                std::string name;
                std::string description;
                std::string units;
                std::vector<size_t> dimensions;
                std::string dataType;
                std::unordered_map<std::string, std::string> attributes;
            };

            /**
             * @brief 时间序列数据结构
             */
            struct TimeSeriesData {
                std::vector<double> timestamps;
                std::vector<std::vector<float>> values;
                std::vector<std::string> variableNames;
                std::unordered_map<std::string, std::string> metadata;
            };

            /**
             * @brief 网格数据结构
             */
            struct GridData {
                std::vector<float> data;
                size_t width;
                size_t height;
                size_t depth;
                double dx, dy, dz;  // 空间分辨率
                double x0, y0, z0;  // 原点坐标
                std::string variableName;
                std::string units;
                double timestamp;
            };

            /**
             * @brief 粒子轨迹数据结构
             */
            struct ParticleTrajectory {
                std::vector<std::vector<float>> positions; // [粒子][时间][坐标]
                std::vector<std::vector<float>> velocities;
                std::vector<double> timestamps;
                std::vector<size_t> particleIds;
                std::unordered_map<std::string, std::vector<float>> properties;
            };

            /**
             * @brief 导出统计信息
             */
            struct ExportStats {
                size_t filesExported = 0;
                size_t totalDataSize = 0;
                double compressionRatio = 0.0;
                std::chrono::milliseconds exportTime{0};
                std::vector<std::string> errorMessages;
            };

        private:
            ExportConfig config_;
            mutable std::mutex exportMutex_;
            ExportStats stats_;
            
            std::unordered_map<std::string, std::string> globalMetadata_;
            std::string coordinateSystem_;
            std::string projection_;
            
            
            // 格式特定的导出器
            std::unordered_map<ExportFormat, std::function<bool(const std::string&, const void*)>> exporters_;

            // 初始化导出器映射
            void initializeExporters();

            // 创建输出目录
            bool createOutputDirectory(const std::string& path);

            // 生成文件名
            std::string generateFileName(const std::string& baseName, ExportFormat format) const;

            // 验证数据完整性
            bool validateData(const void* data, size_t size) const;
            std::string getFileExtension(ExportFormat format) const;

            // === 下面 4 个是粒子轨迹导出辅助函数 ===
            bool exportTrajectoriesToCSV   (const ParticleTrajectory& traj,
                                            const std::string& filename);
            bool exportTrajectoriesToJSON  (const ParticleTrajectory& traj,
                                            const std::string& filename);
            bool exportTrajectoriesToVTK   (const ParticleTrajectory& traj,
                                            const std::string& filename);
            bool exportTrajectoriesToBinary(const ParticleTrajectory& traj,
                                            const std::string& filename);

            // === 磁盘空间检查函数 ===
            bool checkDiskSpace(size_t requiredSpace) const;


        public:

            /**
            * @brief 默认构造——内部自动使用 ExportConfig 的默认值
            */
            DataExporter();                     
            
            /**
             * @brief 构造函数
             * @param config 导出配置
             */
            explicit DataExporter(const ExportConfig& config);

            /**
             * @brief 析构函数
             */
            ~DataExporter();

            // ===========================================
            // 网格数据导出接口
            // ===========================================

            /**
             * @brief 导出2D标量场数据
             * @param data 数据数组
             * @param width 网格宽度
             * @param height 网格高度
             * @param variableName 变量名称
             * @param units 单位
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportScalarField2D(const float* data, size_t width, size_t height,
                                     const std::string& variableName, const std::string& units,
                                     double timestamp, const std::string& filename = "");

            /**
             * @brief 导出3D标量场数据
             * @param data 数据数组
             * @param width 网格宽度
             * @param height 网格高度
             * @param depth 网格深度
             * @param variableName 变量名称
             * @param units 单位
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportScalarField3D(const float* data, size_t width, size_t height, size_t depth,
                                     const std::string& variableName, const std::string& units,
                                     double timestamp, const std::string& filename = "");

            /**
             * @brief 导出2D向量场数据
             * @param u X方向分量
             * @param v Y方向分量
             * @param width 网格宽度
             * @param height 网格高度
             * @param variableName 变量名称
             * @param units 单位
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportVectorField2D(const float* u, const float* v, size_t width, size_t height,
                                     const std::string& variableName, const std::string& units,
                                     double timestamp, const std::string& filename = "");

            /**
             * @brief 导出3D向量场数据
             * @param u X方向分量
             * @param v Y方向分量
             * @param w Z方向分量
             * @param width 网格宽度
             * @param height 网格高度
             * @param depth 网格深度
             * @param variableName 变量名称
             * @param units 单位
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportVectorField3D(const float* u, const float* v, const float* w,
                                     size_t width, size_t height, size_t depth,
                                     const std::string& variableName, const std::string& units,
                                     double timestamp, const std::string& filename = "");

            // ===========================================
            // 粒子数据导出接口
            // ===========================================

            /**
             * @brief 导出粒子轨迹数据
             * @param trajectory 粒子轨迹数据
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportParticleTrajectories(const ParticleTrajectory& trajectory,
                                            const std::string& filename = "");

            /**
             * @brief 导出粒子快照数据
             * @param positions 粒子位置
             * @param velocities 粒子速度
             * @param properties 粒子属性
             * @param numParticles 粒子数量
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportParticleSnapshot(const float* positions, const float* velocities,
                                        const std::unordered_map<std::string, const float*>& properties,
                                        size_t numParticles, double timestamp,
                                        const std::string& filename = "");

            // ===========================================
            // 时间序列数据导出接口
            // ===========================================

            /**
             * @brief 导出时间序列数据
             * @param timeSeries 时间序列数据
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportTimeSeries(const TimeSeriesData& timeSeries,
                                  const std::string& filename = "");

            /**
             * @brief 导出统计数据
             * @param statistics 统计数据映射
             * @param timestamp 时间戳
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportStatistics(const std::unordered_map<std::string, double>& statistics,
                                  double timestamp, const std::string& filename = "");

            // ===========================================
            // 批量导出接口
            // ===========================================

            /**
             * @brief 批量导出多个数据集
             * @param datasets 数据集列表
             * @param baseFilename 基础文件名
             * @return 导出是否成功
             */
            bool exportBatch(const std::vector<GridData>& datasets,
                             const std::string& baseFilename = "");

            /**
             * @brief 导出完整模拟结果
             * @param simulationData 完整模拟数据
             * @param filename 输出文件名
             * @return 导出是否成功
             */
            bool exportSimulationResults(const std::unordered_map<std::string, GridData>& simulationData,
                                         const std::string& filename = "");

            // ===========================================
            // 元数据和配置管理
            // ===========================================

            /**
             * @brief 添加全局元数据
             * @param key 键
             * @param value 值
             */
            void addGlobalMetadata(const std::string& key, const std::string& value);

            /**
             * @brief 设置坐标系信息
             * @param coordinateSystem 坐标系名称
             * @param projection 投影信息
             */
            void setCoordinateSystem(const std::string& coordinateSystem,
                                     const std::string& projection = "");

            /**
             * @brief 更新导出配置
             * @param newConfig 新配置
             */
            void updateConfig(const ExportConfig& newConfig);

            /**
             * @brief 获取导出统计信息
             * @return 统计信息
             */
            const ExportStats& getExportStats() const;

            /**
             * @brief 重置统计信息
             */
            void resetStats();

            // ===========================================
            // 格式特定的导出方法
            // ===========================================

            /**
             * @brief 导出为NetCDF格式
             * @param data 网格数据
             * @param filename 文件名
             * @return 导出是否成功
             */
            bool exportToNetCDF(const GridData& data, const std::string& filename);

            /**
             * @brief 导出为CSV格式
             * @param data 数据
             * @param headers 列标题
             * @param filename 文件名
             * @return 导出是否成功
             */
            bool exportToCSV(const std::vector<std::vector<double>>& data,
                             const std::vector<std::string>& headers,
                             const std::string& filename = "");

            /**
             * @brief 导出为JSON格式
             * @param data 数据
             * @param filename 文件名
             * @return 导出是否成功
             */
            bool exportToJSON(const std::unordered_map<std::string, std::vector<double>>& data,
                              const std::string& filename = "");

            /**
             * @brief 导出为VTK格式
             * @param data 网格数据
             * @param filename 文件名
             * @return 导出是否成功
             */
            bool exportToVTK(const GridData& data, const std::string& filename);

        private:
            bool exportNetCDFImpl(const std::string& filename, const void* data);
            bool exportHDF5Impl(const std::string& filename, const void* data);
            bool exportCSVImpl(const std::string& filename, const void* data);
            bool exportBinaryImpl(const std::string& filename, const void* data);
            bool exportJSONImpl(const std::string& filename, const void* data);
            bool exportVTKImpl(const std::string& filename, const void* data);
            bool exportMATLABImpl(const std::string& filename, const void* data);
            bool exportParaViewImpl(const std::string& filename, const void* data);
        };

    } // namespace Core
} // namespace OceanSimulation

#endif // DATA_EXPORTER_H
