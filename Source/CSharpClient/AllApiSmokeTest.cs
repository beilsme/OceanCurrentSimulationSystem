/*****************************************************
 * 文件路径 : Source/CSharpClient/Tests/IntegrationTests/CoreIntegrationTests/AllApiSmokeTest.cs
 * 项目名称 : OceanCurrentSimulationSystem ▶ C# 集成测试
 * 接口     : 调用 OceanSim.Core 公开托管 API
 * 作者     : beilsm
 * 版本     : v0.1.0
 * 功能说明 :
 *   ▸ 枚举并验证 OceanSim.Core.dll 暴露的全部托管 API（P/Invoke 包装）
 *   ▸ 仅做签名验证与最小调用，便于其他开发者检索函数
 *   ▸ 程序编译并返回 0 视为接口完整可用
 * 最近更新 : 2025-07-02
 *****************************************************/
using System;
using OceanSim.Core;   // 假设托管封装命名空间

namespace IntegrationTests
{
    internal static class AllApiSmokeTest
    {
        /// <summary>
        /// 入口：对所有可用 API 做一次“能调用即可”的冒烟测试。
        /// </summary>
        private static int Main()
        {
            try
            {
                /******************* 性能分析器 *******************/
                IntPtr profiler = PerformanceProfiler.Create();                 // 创建句柄
                PerformanceProfiler.StartTimer(profiler, "init");              // 计时开始
                PerformanceProfiler.StopTimer(profiler, "init");               // 计时结束
                double elapsed = PerformanceProfiler.GetElapsedTime(profiler, "init");
                Console.WriteLine($"Profiler elapsed: {elapsed:F3}s");
                PerformanceProfiler.GenerateReport(profiler, "profiler.json"); // 导出报告
                PerformanceProfiler.Destroy(profiler);                           // 释放

                /******************* 向量运算 ===================*/
                float[] af = {1, 2, 3};
                float[] bf = {4, 5, 6};
                float[] rf = new float[3];
                Vectorized.vector_add(af, bf, rf, rf.Length);
                Vectorized.vector_sub(af, bf, rf, rf.Length);
                Vectorized.vector_mul(af, bf, rf, rf.Length);
                float dotF = Vectorized.dot_product(af, bf, af.Length);

                double[] ad = {1, 2, 3};
                double[] bd = {4, 5, 6};
                double[] rd = new double[3];
                Vectorized.vector_add(ad, bd, rd, rd.Length);
                double dotD = Vectorized.dot_product(ad, bd, ad.Length);

                /******************* 网格与数值算法 **************/
                using (var grid = new Grid(128, 128))
                {
                    var fdm = new FiniteDifferenceSolver(grid);
                    fdm.Step();

                    var rk = new RungeKuttaSolver(grid);
                    rk.Step();
                }

                /******************* 流场求解 *********************/
                using var solver = new CurrentFieldSolver();
                solver.Initialize(timeStep: 0.1);
                solver.Step();

                /******************* 颗粒模拟 *********************/
                using var particleSim = new ParticleSimulator(maxParticles: 1024);
                particleSim.SeedRandomParticles();
                particleSim.Advance(deltaT: 0.1);

                Console.WriteLine("All API smoke test passed.\n" +
                                  $"dotF={dotF}, dotD={dotD}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Smoke test failed: {ex}");
                return 1;
            }
        }
    }
}
