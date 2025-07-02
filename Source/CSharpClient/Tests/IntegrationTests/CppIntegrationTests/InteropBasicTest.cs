using System;
using OceanSim.Core;

namespace IntegrationTests
{
    /// <summary>
    /// 简单的 C++/C# 互操作验证测试。
    /// 构建后执行即可，如果加载动态库成功且能够调用基础函数，程序将输出版本号并返回 0。
    /// </summary>
    internal static class InteropBasicTest
    {
        private static int Main()
        {
            try
            {
                if (!OceanSimLibrary.Initialize())
                {
                    Console.Error.WriteLine("初始化 OceanSim 失败");
                    return 1;
                }

                Console.WriteLine($"OceanSim 版本: {OceanSimLibrary.Version}");
                OceanSimLibrary.Cleanup();
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"测试运行异常: {ex}");
                return 1;
            }
        }
    }
}