#include "data/DataExporter.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace OceanSimulation {
    namespace Core {


        DataExporter::DataExporter()
                : DataExporter(ExportConfig{}) {}
        
        DataExporter::DataExporter(const ExportConfig& config) : config_(config), stats_{} {
            initializeExporters();
            createOutputDirectory(config_.outputDirectory);
        }




        DataExporter::~DataExporter() {
            // 清理资源
        }

        void DataExporter::initializeExporters() {
            exporters_[ExportFormat::NetCDF] = [this](const std::string& filename, const void* data) {
                return exportNetCDFImpl(filename, data);
            };

            exporters_[ExportFormat::HDF5] = [this](const std::string& filename, const void* data) {
                return exportHDF5Impl(filename, data);
            };

            exporters_[ExportFormat::CSV] = [this](const std::string& filename, const void* data) {
                return exportCSVImpl(filename, data);
            };

            exporters_[ExportFormat::JSON] = [this](const std::string& filename, const void* data) {
                return exportJSONImpl(filename, data);
            };

            exporters_[ExportFormat::Binary] = [this](const std::string& filename, const void* data) {
                return exportBinaryImpl(filename, data);
            };

            exporters_[ExportFormat::VTK] = [this](const std::string& filename, const void* data) {
                return exportVTKImpl(filename, data);
            };

            exporters_[ExportFormat::MATLAB] = [this](const std::string& filename, const void* data) {
                return exportMATLABImpl(filename, data);
            };

            exporters_[ExportFormat::ParaView] = [this](const std::string& filename, const void* data) {
                return exportParaViewImpl(filename, data);
            };
        }

        bool DataExporter::createOutputDirectory(const std::string& path) {
            try {
                std::filesystem::create_directories(path);
                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "创建输出目录失败: " << e.what() << std::endl;
                return false;
            }
        }

        std::string DataExporter::generateFileName(const std::string& baseName, ExportFormat format) const {
            std::stringstream ss;
            ss << config_.outputDirectory;

            if (!config_.outputDirectory.empty() && config_.outputDirectory.back() != '/' &&
                config_.outputDirectory.back() != '\\') {
                ss << "/";
            }

            if (!config_.filePrefix.empty()) {
                ss << config_.filePrefix << "_";
            }

            ss << baseName;

            if (config_.timestampFiles) {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;

                ss << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
                ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
            }

            ss << getFileExtension(format);
            return ss.str();
        }

        std::string DataExporter::getFileExtension(ExportFormat format) const {
            switch (format) {
                case ExportFormat::NetCDF: return ".nc";
                case ExportFormat::HDF5: return ".h5";
                case ExportFormat::CSV: return ".csv";
                case ExportFormat::JSON: return ".json";
                case ExportFormat::Binary: return ".bin";
                case ExportFormat::VTK: return ".vtk";
                case ExportFormat::MATLAB: return ".mat";
                case ExportFormat::ParaView: return ".pvtu";
                default: return ".dat";
            }
        }

        bool DataExporter::exportScalarField2D(const float* data, size_t width, size_t height,
                                               const std::string& variableName, const std::string& units,
                                               double timestamp, const std::string& filename) {
            std::lock_guard<std::mutex> lock(exportMutex_);

            auto startTime = std::chrono::high_resolution_clock::now();

            GridData gridData;
            gridData.data.assign(data, data + width * height);
            gridData.width = width;
            gridData.height = height;
            gridData.depth = 1;
            gridData.variableName = variableName;
            gridData.units = units;
            gridData.timestamp = timestamp;

            std::string outputFile = filename.empty() ?
                                     generateFileName(variableName + "_2D", config_.format) : filename;

            bool success = false;
            auto it = exporters_.find(config_.format);
            if (it != exporters_.end()) {
                success = it->second(outputFile, &gridData);
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            if (success) {
                ++stats_.filesExported;
                stats_.totalDataSize += width * height * sizeof(float);
                stats_.exportTime += duration;
                std::cout << "成功导出2D标量场: " << outputFile << " (耗时: " << duration.count() << "ms)" << std::endl;
            } else {
                stats_.errorMessages.push_back("导出2D标量场失败: " + outputFile);
                std::cerr << "导出2D标量场失败: " << outputFile << std::endl;
            }

            return success;
        }

        bool DataExporter::exportVectorField2D(const float* u, const float* v, size_t width, size_t height,
                                               const std::string& variableName, const std::string& units,
                                               double timestamp, const std::string& filename) {
            std::lock_guard<std::mutex> lock(exportMutex_);

            // 将U和V分量交错存储
            std::vector<float> vectorData(width * height * 2);
            for (size_t i = 0; i < width * height; ++i) {
                vectorData[i * 2] = u[i];
                vectorData[i * 2 + 1] = v[i];
            }

            GridData gridData;
            gridData.data = std::move(vectorData);
            gridData.width = width;
            gridData.height = height;
            gridData.depth = 1;
            gridData.variableName = variableName;
            gridData.units = units;
            gridData.timestamp = timestamp;

            std::string outputFile = filename.empty() ?
                                     generateFileName(variableName + "_vector2D", config_.format) : filename;

            auto it = exporters_.find(config_.format);
            if (it != exporters_.end()) {
                bool success = it->second(outputFile, &gridData);
                if (success) {
                    ++stats_.filesExported;
                    stats_.totalDataSize += width * height * 2 * sizeof(float);
                    std::cout << "成功导出2D向量场: " << outputFile << std::endl;
                }
                return success;
            }

            return false;
        }

        bool DataExporter::exportParticleTrajectories(const ParticleTrajectory& trajectory,
                                                      const std::string& filename) {
            std::string outputFile = filename.empty() ?
                                     generateFileName("particle_trajectories", config_.format) : filename;

            switch (config_.format) {
                case ExportFormat::CSV:
                    return exportTrajectoriesToCSV(trajectory, outputFile);
                case ExportFormat::JSON:
                    return exportTrajectoriesToJSON(trajectory, outputFile);
                case ExportFormat::VTK:
                    return exportTrajectoriesToVTK(trajectory, outputFile);
                default:
                    return exportTrajectoriesToBinary(trajectory, outputFile);
            }
        }

        bool DataExporter::exportTrajectoriesToCSV(const ParticleTrajectory& trajectory,
                                                   const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "无法打开文件进行写入: " << filename << std::endl;
                return false;
            }

            // 写入CSV头部
            file << "ParticleID,Timestamp,X,Y,Z,VX,VY,VZ";
            for (const auto& prop : trajectory.properties) {
                file << "," << prop.first;
            }
            file << "\n";

            // 写入轨迹数据
            for (size_t p = 0; p < trajectory.particleIds.size(); ++p) {
                for (size_t t = 0; t < trajectory.timestamps.size(); ++t) {
                    file << trajectory.particleIds[p] << ","
                         << std::fixed << std::setprecision(6) << trajectory.timestamps[t] << ","
                         << trajectory.positions[p][t * 3] << ","
                         << trajectory.positions[p][t * 3 + 1] << ","
                         << trajectory.positions[p][t * 3 + 2] << ","
                         << trajectory.velocities[p][t * 3] << ","
                         << trajectory.velocities[p][t * 3 + 1] << ","
                         << trajectory.velocities[p][t * 3 + 2];

                    for (const auto& prop : trajectory.properties) {
                        file << "," << prop.second[p * trajectory.timestamps.size() + t];
                    }
                    file << "\n";
                }
            }

            file.close();
            return true;
        }

        bool DataExporter::exportTrajectoriesToJSON(const ParticleTrajectory& trajectory,
                                                    const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }

            file << "{\n  \"particles\": [\n";
            for (size_t p = 0; p < trajectory.particleIds.size(); ++p) {
                file << "    {\"id\": " << trajectory.particleIds[p] << ", \"path\": [";
                for (size_t t = 0; t < trajectory.timestamps.size(); ++t) {
                    file << "[" << trajectory.positions[p][t * 3] << ","
                         << trajectory.positions[p][t * 3 + 1] << ","
                         << trajectory.positions[p][t * 3 + 2] << "]";
                    if (t + 1 < trajectory.timestamps.size()) file << ", ";
                }
                file << "]}";
                if (p + 1 < trajectory.particleIds.size()) file << ",";
                file << "\n";
            }
            file << "  ]\n}\n";
            file.close();
            return true;
        }

        bool DataExporter::exportTrajectoriesToVTK(const ParticleTrajectory& trajectory,
                                                   const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }

            file << "# vtk DataFile Version 3.0\nTrajectories\nASCII\nDATASET POLYDATA\n";
            size_t numPoints = trajectory.particleIds.size() * trajectory.timestamps.size();
            file << "POINTS " << numPoints << " float\n";
            for (size_t p = 0; p < trajectory.particleIds.size(); ++p) {
                for (size_t t = 0; t < trajectory.timestamps.size(); ++t) {
                    file << trajectory.positions[p][t * 3] << " "
                         << trajectory.positions[p][t * 3 + 1] << " "
                         << trajectory.positions[p][t * 3 + 2] << "\n";
                }
            }
            file.close();
            return true;
        }

        bool DataExporter::exportTrajectoriesToBinary(const ParticleTrajectory& trajectory,
                                                      const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) return false;

            size_t particleCount = trajectory.particleIds.size();
            size_t timeCount = trajectory.timestamps.size();
            file.write(reinterpret_cast<const char*>(&particleCount), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&timeCount), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(trajectory.timestamps.data()),
                       timeCount * sizeof(double));
            for (size_t p = 0; p < particleCount; ++p) {
                file.write(reinterpret_cast<const char*>(&trajectory.particleIds[p]), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(trajectory.positions[p].data()),
                           timeCount * 3 * sizeof(float));
                file.write(reinterpret_cast<const char*>(trajectory.velocities[p].data()),
                           timeCount * 3 * sizeof(float));
            }
            file.close();
            return true;
        }
        
        
        
        bool DataExporter::exportToCSV(const std::vector<std::vector<double>>& data,
                                       const std::vector<std::string>& headers,
                                       const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }

            // 写入标题行
            for (size_t i = 0; i < headers.size(); ++i) {
                file << headers[i];
                if (i < headers.size() - 1) file << ",";
            }
            file << "\n";

            // 写入数据行
            for (const auto& row : data) {
                for (size_t i = 0; i < row.size(); ++i) {
                    file << std::fixed << std::setprecision(6) << row[i];
                    if (i < row.size() - 1) file << ",";
                }
                file << "\n";
            }

            file.close();
            return true;
        }

        bool DataExporter::exportToJSON(const std::unordered_map<std::string, std::vector<double>>& data,
                                        const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }

            file << "{\n";

            // 添加元数据
            if (config_.includeMetadata) {
                file << "  \"metadata\": {\n";
                file << "    \"export_time\": \"" << std::chrono::system_clock::now().time_since_epoch().count() << "\",\n";
                file << "    \"coordinate_system\": \"" << coordinateSystem_ << "\",\n";
                file << "    \"projection\": \"" << projection_ << "\"\n";

                for (const auto& meta : globalMetadata_) {
                    file << "    ,\"" << meta.first << "\": \"" << meta.second << "\"\n";
                }

                file << "  },\n";
            }

            file << "  \"data\": {\n";

            bool first = true;
            for (const auto& dataset : data) {
                if (!first) file << ",\n";
                first = false;

                file << "    \"" << dataset.first << "\": [";
                for (size_t i = 0; i < dataset.second.size(); ++i) {
                    file << dataset.second[i];
                    if (i < dataset.second.size() - 1) file << ", ";
                }
                file << "]";
            }

            file << "\n  }\n";
            file << "}\n";

            file.close();
            return true;
        }

        bool DataExporter::exportToVTK(const GridData& data, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                return false;
            }

            // VTK文件头
            file << "# vtk DataFile Version 3.0\n";
            file << "Ocean Simulation Data\n";
            file << "ASCII\n";
            file << "DATASET STRUCTURED_POINTS\n";

            // 网格维度
            file << "DIMENSIONS " << data.width << " " << data.height << " " << data.depth << "\n";
            file << "ORIGIN " << data.x0 << " " << data.y0 << " " << data.z0 << "\n";
            file << "SPACING " << data.dx << " " << data.dy << " " << data.dz << "\n";

            // 数据
            size_t numPoints = data.width * data.height * data.depth;
            file << "POINT_DATA " << numPoints << "\n";
            file << "SCALARS " << data.variableName << " float 1\n";
            file << "LOOKUP_TABLE default\n";

            for (size_t i = 0; i < data.data.size(); ++i) {
                file << data.data[i] << "\n";
            }

            file.close();
            return true;
        }

// 格式特定的导出实现（简化版本）
        bool DataExporter::exportNetCDFImpl(const std::string& filename, const void* data) {
            // 这里应该使用NetCDF库进行实际实现
            // 目前提供简化的二进制导出作为占位符
            std::cout << "导出NetCDF格式: " << filename << " (需要NetCDF库支持)" << std::endl;
            return exportBinaryImpl(filename, data);
        }

        bool DataExporter::exportHDF5Impl(const std::string& filename, const void* data) {
            // 这里应该使用HDF5库进行实际实现
            std::cout << "导出HDF5格式: " << filename << " (需要HDF5库支持)" << std::endl;
            return exportBinaryImpl(filename, data);
        }

        bool DataExporter::exportCSVImpl(const std::string& filename, const void* data) {
            const GridData* gridData = static_cast<const GridData*>(data);

            std::ofstream file(filename);
            if (!file.is_open()) return false;

            // 写入2D网格数据为CSV格式
            for (size_t j = 0; j < gridData->height; ++j) {
                for (size_t i = 0; i < gridData->width; ++i) {
                    size_t idx = j * gridData->width + i;
                    file << gridData->data[idx];
                    if (i < gridData->width - 1) file << ",";
                }
                file << "\n";
            }

            file.close();
            return true;
        }

        bool DataExporter::exportBinaryImpl(const std::string& filename, const void* data) {
            const GridData* gridData = static_cast<const GridData*>(data);

            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) return false;

            // 写入头信息
            file.write(reinterpret_cast<const char*>(&gridData->width), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&gridData->height), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&gridData->depth), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&gridData->timestamp), sizeof(double));

            // 写入数据
            file.write(reinterpret_cast<const char*>(gridData->data.data()),
                       gridData->data.size() * sizeof(float));

            file.close();
            return true;
        }

        void DataExporter::addGlobalMetadata(const std::string& key, const std::string& value) {
            globalMetadata_[key] = value;
        }

        void DataExporter::setCoordinateSystem(const std::string& coordinateSystem,
                                               const std::string& projection) {
            coordinateSystem_ = coordinateSystem;
            projection_ = projection;
        }

        const DataExporter::ExportStats& DataExporter::getExportStats() const {
            return stats_;
        }

        void DataExporter::resetStats() {
            stats_ = ExportStats{};
        }

        bool DataExporter::checkDiskSpace(size_t requiredSpace) const {
            try {
                auto space = std::filesystem::space(config_.outputDirectory);
                return space.available >= requiredSpace;
            }
            catch (const std::exception&) {
                return false;
            }
        }

// 占位符实现
        bool DataExporter::exportJSONImpl(const std::string& filename, const void* data) { return true; }
        bool DataExporter::exportVTKImpl(const std::string& filename, const void* data) { return true; }
        bool DataExporter::exportMATLABImpl(const std::string& filename, const void* data) { return true; }
        bool DataExporter::exportParaViewImpl(const std::string& filename, const void* data) { return true; }

    } // namespace Core
} // namespace OceanSimulation