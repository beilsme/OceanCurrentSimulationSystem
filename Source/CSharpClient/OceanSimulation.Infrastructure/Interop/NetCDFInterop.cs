// ==============================================================================
// 文件: NetCDFInterop.cs
// 目录: Source/CSharpClient/OceanSimulation.Infrastructure/Interop/
// 功能: C# -> libnetcdf 动态库 P/Invoke
// 作者: beilsm
// 版本: v0.1.0
// 最近更新: 2025-07-04
// ==============================================================================
using System.Runtime.InteropServices;

namespace OceanSimulation.Infrastructure.Interop
{
    public static class NetCDFNative
    {
        [DllImport("netcdf", CallingConvention = CallingConvention.Cdecl)]
        public static extern int nc_open(string path, int mode, out int ncid);

        [DllImport("netcdf", CallingConvention = CallingConvention.Cdecl)]
        public static extern int nc_close(int ncid);

        [DllImport("netcdf", CallingConvention = CallingConvention.Cdecl)]
        public static extern int nc_get_var_int(int ncid, int varid, int[] data);

        [DllImport("netcdf", CallingConvention = CallingConvention.Cdecl)]
        public static extern int nc_inq_varid(int ncid, string name, out int varid);
    }
}
