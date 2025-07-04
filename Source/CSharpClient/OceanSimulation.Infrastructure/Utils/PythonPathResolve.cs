using System;
using System.IO;

namespace OceanSimulation.Infrastructure.Utils
{
    /// <summary>
    /// Helper for resolving the Python engine locations at runtime.
    /// </summary>
    public static class PythonPathResolver
    {
        /// <summary>
        /// Locate the Python engine root directory by searching parent directories
        /// for the 'Source/PythonEngine' folder.
        /// </summary>
        public static string LocatePythonEngineRoot()
        {
            var dir = AppDomain.CurrentDomain.BaseDirectory;
            while (!string.IsNullOrEmpty(dir))
            {
                var candidate = Path.Combine(dir, "Source", "PythonEngine");
                if (Directory.Exists(candidate))
                    return candidate;

                dir = Directory.GetParent(dir)?.FullName;
            }
            throw new DirectoryNotFoundException("PythonEngine directory not found");
        }

        /// <summary>
        /// Locate the Python executable within the Python engine's virtual environment.
        /// </summary>
        public static string LocatePythonExecutable(string pythonEngineRoot)
        {
            var unixPath = Path.Combine(pythonEngineRoot, ".venv", "bin", "python");
            if (File.Exists(unixPath))
                return unixPath;
            var winPath = Path.Combine(pythonEngineRoot, ".venv", "Scripts", "python.exe");
            if (File.Exists(winPath))
                return winPath;
            return "python"; // Fallback
        }

        /// <summary>
        /// Get path for temporary working directory.
        /// </summary>
        public static string GetWorkingDirectory(string pythonEngineRoot)
        {
            var tempDir = Path.Combine(pythonEngineRoot, "Temp");
            Directory.CreateDirectory(tempDir);
            return tempDir;
        }
    }
}
