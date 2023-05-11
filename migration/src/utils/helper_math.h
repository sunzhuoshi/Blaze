/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

/* DPCT_ORIG #include "cuda_runtime.h"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif


#define M_EPSILON  0.00001f
#define M_INF	   3.402823466e+38F
//#define M_E        2.71828182845904523536f
//#define M_LOG2E    1.44269504088896340736f
//#define M_LOG10E   0.434294481903251827651f
//#define M_LN2      0.693147180559945309417f
//#define M_LN10     2.30258509299404568402f
#define M_PI       3.14159265358979323846f
#define M_2PI      6.28318530717958647692f
//#define M_PI_2     1.57079632679489661923f
//#define M_PI_4     0.785398163397448309616f
#define M_1_PI     0.318309886183790671538f
//#define M_2_PI     0.636619772367581343076f
#define M_SQRT1_3  0.577350269189625764509f
#define M_1_180    0.005555555555555555556f

/* DPCT_ORIG #ifndef __CUDACC__*/
#ifndef SYCL_LANGUAGE_VERSION
#include <math.h>



////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 make_float2(float s)*/
inline sycl::float2 make_float2(float s)
{
/* DPCT_ORIG     return make_float2(s, s);*/
    return sycl::float2(s, s);
}
/* DPCT_ORIG inline __host__ __device__ float2 make_float2(float3 a)*/
inline sycl::float2 make_float2(sycl::float3 a)
{
/* DPCT_ORIG     return make_float2(a.x, a.y);*/
    return sycl::float2(a.x(), a.y());
}
/* DPCT_ORIG inline __host__ __device__ float2 make_float2(int2 a)*/
inline sycl::float2 make_float2(sycl::int2 a)
{
/* DPCT_ORIG     return make_float2(float(a.x), float(a.y));*/
    return sycl::float2(float(a.x()), float(a.y()));
}
/* DPCT_ORIG inline __host__ __device__ float2 make_float2(uint2 a)*/
inline sycl::float2 make_float2(sycl::uint2 a)
{
/* DPCT_ORIG     return make_float2(float(a.x), float(a.y));*/
    return sycl::float2(float(a.x()), float(a.y()));
}

/* DPCT_ORIG inline __host__ __device__ int2 make_int2(int s)*/
inline sycl::int2 make_int2(int s)
{
/* DPCT_ORIG     return make_int2(s, s);*/
    return sycl::int2(s, s);
}
/* DPCT_ORIG inline __host__ __device__ int2 make_int2(int3 a)*/
inline sycl::int2 make_int2(sycl::int3 a)
{
/* DPCT_ORIG     return make_int2(a.x, a.y);*/
    return sycl::int2(a.x(), a.y());
}
/* DPCT_ORIG inline __host__ __device__ int2 make_int2(uint2 a)*/
inline sycl::int2 make_int2(sycl::uint2 a)
{
/* DPCT_ORIG     return make_int2(int(a.x), int(a.y));*/
    return sycl::int2(int(a.x()), int(a.y()));
}
/* DPCT_ORIG inline __host__ __device__ int2 make_int2(float2 a)*/
inline sycl::int2 make_int2(sycl::float2 a)
{
/* DPCT_ORIG     return make_int2(int(a.x), int(a.y));*/
    return sycl::int2(int(a.x()), int(a.y()));
}

/* DPCT_ORIG inline __host__ __device__ uint2 make_uint2(uint s)*/
inline sycl::uint2 make_uint2(uint s)
{
/* DPCT_ORIG     return make_uint2(s, s);*/
    return sycl::uint2(s, s);
}
/* DPCT_ORIG inline __host__ __device__ uint2 make_uint2(uint3 a)*/
inline sycl::uint2 make_uint2(sycl::uint3 a)
{
/* DPCT_ORIG     return make_uint2(a.x, a.y);*/
    return sycl::uint2(a.x(), a.y());
}
/* DPCT_ORIG inline __host__ __device__ uint2 make_uint2(int2 a)*/
inline sycl::uint2 make_uint2(sycl::int2 a)
{
/* DPCT_ORIG     return make_uint2(uint(a.x), uint(a.y));*/
    return sycl::uint2(uint(a.x()), uint(a.y()));
}

/* DPCT_ORIG inline __host__ __device__ float3 make_float3(float s)*/
inline sycl::float3 make_float3(float s)
{
/* DPCT_ORIG     return make_float3(s, s, s);*/
    return sycl::float3(s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ float3 make_float3(float2 a)*/
inline sycl::float3 make_float3(sycl::float2 a)
{
/* DPCT_ORIG     return make_float3(a.x, a.y, 0.0f);*/
    return sycl::float3(a.x(), a.y(), 0.0f);
}
/* DPCT_ORIG inline __host__ __device__ float3 make_float3(float2 a, float s)*/
inline sycl::float3 make_float3(sycl::float2 a, float s)
{
/* DPCT_ORIG     return make_float3(a.x, a.y, s);*/
    return sycl::float3(a.x(), a.y(), s);
}
/* DPCT_ORIG inline __host__ __device__ float3 make_float3(float4 a)*/
inline sycl::float3 make_float3(sycl::float4 a)
{
/* DPCT_ORIG     return make_float3(a.x, a.y, a.z);*/
    return sycl::float3(a.x(), a.y(), a.z());
}
/* DPCT_ORIG inline __host__ __device__ float3 make_float3(int3 a)*/
inline sycl::float3 make_float3(sycl::int3 a)
{
/* DPCT_ORIG     return make_float3(float(a.x), float(a.y), float(a.z));*/
    return sycl::float3(float(a.x()), float(a.y()), float(a.z()));
}
/* DPCT_ORIG inline __host__ __device__ float3 make_float3(uint3 a)*/
inline sycl::float3 make_float3(sycl::uint3 a)
{
/* DPCT_ORIG     return make_float3(float(a.x), float(a.y), float(a.z));*/
    return sycl::float3(float(a.x()), float(a.y()), float(a.z()));
}

/* DPCT_ORIG inline __host__ __device__ int3 make_int3(int s)*/
inline sycl::int3 make_int3(int s)
{
/* DPCT_ORIG     return make_int3(s, s, s);*/
    return sycl::int3(s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ int3 make_int3(int2 a)*/
inline sycl::int3 make_int3(sycl::int2 a)
{
/* DPCT_ORIG     return make_int3(a.x, a.y, 0);*/
    return sycl::int3(a.x(), a.y(), 0);
}
/* DPCT_ORIG inline __host__ __device__ int3 make_int3(int2 a, int s)*/
inline sycl::int3 make_int3(sycl::int2 a, int s)
{
/* DPCT_ORIG     return make_int3(a.x, a.y, s);*/
    return sycl::int3(a.x(), a.y(), s);
}
/* DPCT_ORIG inline __host__ __device__ int3 make_int3(uint3 a)*/
inline sycl::int3 make_int3(sycl::uint3 a)
{
/* DPCT_ORIG     return make_int3(int(a.x), int(a.y), int(a.z));*/
    return sycl::int3(int(a.x()), int(a.y()), int(a.z()));
}
/* DPCT_ORIG inline __host__ __device__ int3 make_int3(float3 a)*/
inline sycl::int3 make_int3(sycl::float3 a)
{
/* DPCT_ORIG     return make_int3(int(a.x), int(a.y), int(a.z));*/
    return sycl::int3(int(a.x()), int(a.y()), int(a.z()));
}

/* DPCT_ORIG inline __host__ __device__ uint3 make_uint3(uint s)*/
inline sycl::uint3 make_uint3(uint s)
{
/* DPCT_ORIG     return make_uint3(s, s, s);*/
    return sycl::uint3(s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ uint3 make_uint3(uint2 a)*/
inline sycl::uint3 make_uint3(sycl::uint2 a)
{
/* DPCT_ORIG     return make_uint3(a.x, a.y, 0);*/
    return sycl::uint3(a.x(), a.y(), 0);
}
/* DPCT_ORIG inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)*/
inline sycl::uint3 make_uint3(sycl::uint2 a, uint s)
{
/* DPCT_ORIG     return make_uint3(a.x, a.y, s);*/
    return sycl::uint3(a.x(), a.y(), s);
}
/* DPCT_ORIG inline __host__ __device__ uint3 make_uint3(uint4 a)*/
inline sycl::uint3 make_uint3(sycl::uint4 a)
{
/* DPCT_ORIG     return make_uint3(a.x, a.y, a.z);*/
    return sycl::uint3(a.x(), a.y(), a.z());
}
/* DPCT_ORIG inline __host__ __device__ uint3 make_uint3(int3 a)*/
inline sycl::uint3 make_uint3(sycl::int3 a)
{
/* DPCT_ORIG     return make_uint3(uint(a.x), uint(a.y), uint(a.z));*/
    return sycl::uint3(uint(a.x()), uint(a.y()), uint(a.z()));
}

/* DPCT_ORIG inline __host__ __device__ float4 make_float4(float s)*/
inline sycl::float4 make_float4(float s)
{
/* DPCT_ORIG     return make_float4(s, s, s, s);*/
    return sycl::float4(s, s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ float4 make_float4(float3 a)*/
inline sycl::float4 make_float4(sycl::float3 a)
{
/* DPCT_ORIG     return make_float4(a.x, a.y, a.z, 0.0f);*/
    return sycl::float4(a.x(), a.y(), a.z(), 0.0f);
}
/* DPCT_ORIG inline __host__ __device__ float4 make_float4(float3 a, float w)*/
inline sycl::float4 make_float4(sycl::float3 a, float w)
{
/* DPCT_ORIG     return make_float4(a.x, a.y, a.z, w);*/
    return sycl::float4(a.x(), a.y(), a.z(), w);
}
/* DPCT_ORIG inline __host__ __device__ float4 make_float4(int4 a)*/
inline sycl::float4 make_float4(sycl::int4 a)
{
/* DPCT_ORIG     return make_float4(float(a.x), float(a.y), float(a.z),
 * float(a.w));*/
    return sycl::float4(float(a.x()), float(a.y()), float(a.z()), float(a.w()));
}
/* DPCT_ORIG inline __host__ __device__ float4 make_float4(uint4 a)*/
inline sycl::float4 make_float4(sycl::uint4 a)
{
/* DPCT_ORIG     return make_float4(float(a.x), float(a.y), float(a.z),
 * float(a.w));*/
    return sycl::float4(float(a.x()), float(a.y()), float(a.z()), float(a.w()));
}

/* DPCT_ORIG inline __host__ __device__ int4 make_int4(int s)*/
inline sycl::int4 make_int4(int s)
{
/* DPCT_ORIG     return make_int4(s, s, s, s);*/
    return sycl::int4(s, s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ int4 make_int4(int3 a)*/
inline sycl::int4 make_int4(sycl::int3 a)
{
/* DPCT_ORIG     return make_int4(a.x, a.y, a.z, 0);*/
    return sycl::int4(a.x(), a.y(), a.z(), 0);
}
/* DPCT_ORIG inline __host__ __device__ int4 make_int4(int3 a, int w)*/
inline sycl::int4 make_int4(sycl::int3 a, int w)
{
/* DPCT_ORIG     return make_int4(a.x, a.y, a.z, w);*/
    return sycl::int4(a.x(), a.y(), a.z(), w);
}
/* DPCT_ORIG inline __host__ __device__ int4 make_int4(uint4 a)*/
inline sycl::int4 make_int4(sycl::uint4 a)
{
/* DPCT_ORIG     return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));*/
    return sycl::int4(int(a.x()), int(a.y()), int(a.z()), int(a.w()));
}
/* DPCT_ORIG inline __host__ __device__ int4 make_int4(float4 a)*/
inline sycl::int4 make_int4(sycl::float4 a)
{
/* DPCT_ORIG     return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));*/
    return sycl::int4(int(a.x()), int(a.y()), int(a.z()), int(a.w()));
}

/* DPCT_ORIG inline __host__ __device__ uint4 make_uint4(uint s)*/
inline sycl::uint4 make_uint4(uint s)
{
/* DPCT_ORIG     return make_uint4(s, s, s, s);*/
    return sycl::uint4(s, s, s, s);
}
/* DPCT_ORIG inline __host__ __device__ uint4 make_uint4(uint3 a)*/
inline sycl::uint4 make_uint4(sycl::uint3 a)
{
/* DPCT_ORIG     return make_uint4(a.x, a.y, a.z, 0);*/
    return sycl::uint4(a.x(), a.y(), a.z(), 0);
}
/* DPCT_ORIG inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)*/
inline sycl::uint4 make_uint4(sycl::uint3 a, uint w)
{
/* DPCT_ORIG     return make_uint4(a.x, a.y, a.z, w);*/
    return sycl::uint4(a.x(), a.y(), a.z(), w);
}
/* DPCT_ORIG inline __host__ __device__ uint4 make_uint4(int4 a)*/
inline sycl::uint4 make_uint4(sycl::int4 a)
{
/* DPCT_ORIG     return make_uint4(uint(a.x), uint(a.y), uint(a.z),
 * uint(a.w));*/
    return sycl::uint4(uint(a.x()), uint(a.y()), uint(a.z()), uint(a.w()));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 operator-(float2 &a)*/
/*
DPCT1011:60: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(sycl::float2 &a)
{
/* DPCT_ORIG     return make_float2(-a.x, -a.y);*/
    return sycl::float2(-a.x(), -a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator-(int2 &a)*/
/*
DPCT1011:61: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator-(sycl::int2 &a)
{
/* DPCT_ORIG     return make_int2(-a.x, -a.y);*/
    return sycl::int2(-a.x(), -a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator-(float3 &a)*/
/*
DPCT1011:62: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator-(sycl::float3 &a)
{
/* DPCT_ORIG     return make_float3(-a.x, -a.y, -a.z);*/
    return sycl::float3(-a.x(), -a.y(), -a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator-(int3 &a)*/
/*
DPCT1011:63: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator-(sycl::int3 &a)
{
/* DPCT_ORIG     return make_int3(-a.x, -a.y, -a.z);*/
    return sycl::int3(-a.x(), -a.y(), -a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator-(float4 &a)*/
/*
DPCT1011:64: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator-(sycl::float4 &a)
{
/* DPCT_ORIG     return make_float4(-a.x, -a.y, -a.z, -a.w);*/
    return sycl::float4(-a.x(), -a.y(), -a.z(), -a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator-(int4 &a)*/
/*
DPCT1011:65: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator-(sycl::int4 &a)
{
/* DPCT_ORIG     return make_int4(-a.x, -a.y, -a.z, -a.w);*/
    return sycl::int4(-a.x(), -a.y(), -a.z(), -a.w());
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 operator+(float2 a, float2 b)*/
/*
DPCT1011:66: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(a.x + b.x, a.y + b.y);*/
    return sycl::float2(a.x() + b.x(), a.y() + b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float2 &a, float2 b)*/
/*
DPCT1011:67: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float2 &a, sycl::float2 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator+(float2 a, float b)*/
/*
DPCT1011:68: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(sycl::float2 a, float b)
{
/* DPCT_ORIG     return make_float2(a.x + b, a.y + b);*/
    return sycl::float2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator+(float b, float2 a)*/
/*
DPCT1011:69: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(float b, sycl::float2 a)
{
/* DPCT_ORIG     return make_float2(a.x + b, a.y + b);*/
    return sycl::float2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float2 &a, float b)*/
/*
DPCT1011:70: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float2 &a, float b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator+(int2 a, int2 b)*/
/*
DPCT1011:71: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator+(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(a.x + b.x, a.y + b.y);*/
    return sycl::int2(a.x() + b.x(), a.y() + b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int2 &a, int2 b)*/
/*
DPCT1011:72: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int2 &a, sycl::int2 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator+(int2 a, int b)*/
/*
DPCT1011:73: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator+(sycl::int2 a, int b)
{
/* DPCT_ORIG     return make_int2(a.x + b, a.y + b);*/
    return sycl::int2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator+(int b, int2 a)*/
/*
DPCT1011:74: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator+(int b, sycl::int2 a)
{
/* DPCT_ORIG     return make_int2(a.x + b, a.y + b);*/
    return sycl::int2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int2 &a, int b)*/
/*
DPCT1011:75: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int2 &a, int b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)*/
/*
DPCT1011:76: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator+(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(a.x + b.x, a.y + b.y);*/
    return sycl::uint2(a.x() + b.x(), a.y() + b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint2 &a, uint2 b)*/
/*
DPCT1011:77: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint2 &a, sycl::uint2 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator+(uint2 a, uint b)*/
/*
DPCT1011:78: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator+(sycl::uint2 a, uint b)
{
/* DPCT_ORIG     return make_uint2(a.x + b, a.y + b);*/
    return sycl::uint2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator+(uint b, uint2 a)*/
/*
DPCT1011:79: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator+(uint b, sycl::uint2 a)
{
/* DPCT_ORIG     return make_uint2(a.x + b, a.y + b);*/
    return sycl::uint2(a.x() + b, a.y() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint2 &a, uint b)*/
/*
DPCT1011:80: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint2 &a, uint b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator+(float3 a, float3 b)*/
/*
DPCT1011:81: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator+(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);*/
    return sycl::float3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float3 &a, float3 b)*/
/*
DPCT1011:82: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float3 &a, sycl::float3 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator+(float3 a, float b)*/
/*
DPCT1011:83: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator+(sycl::float3 a, float b)
{
/* DPCT_ORIG     return make_float3(a.x + b, a.y + b, a.z + b);*/
    return sycl::float3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float3 &a, float b)*/
/*
DPCT1011:84: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float3 &a, float b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator+(int3 a, int3 b)*/
/*
DPCT1011:85: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator+(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);*/
    return sycl::int3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int3 &a, int3 b)*/
/*
DPCT1011:86: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int3 &a, sycl::int3 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator+(int3 a, int b)*/
/*
DPCT1011:87: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator+(sycl::int3 a, int b)
{
/* DPCT_ORIG     return make_int3(a.x + b, a.y + b, a.z + b);*/
    return sycl::int3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int3 &a, int b)*/
/*
DPCT1011:88: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int3 &a, int b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)*/
/*
DPCT1011:89: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator+(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);*/
    return sycl::uint3(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint3 &a, uint3 b)*/
/*
DPCT1011:90: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint3 &a, sycl::uint3 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator+(uint3 a, uint b)*/
/*
DPCT1011:91: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator+(sycl::uint3 a, uint b)
{
/* DPCT_ORIG     return make_uint3(a.x + b, a.y + b, a.z + b);*/
    return sycl::uint3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint3 &a, uint b)*/
/*
DPCT1011:92: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint3 &a, uint b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator+(int b, int3 a)*/
/*
DPCT1011:93: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator+(int b, sycl::int3 a)
{
/* DPCT_ORIG     return make_int3(a.x + b, a.y + b, a.z + b);*/
    return sycl::int3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator+(uint b, uint3 a)*/
/*
DPCT1011:94: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator+(uint b, sycl::uint3 a)
{
/* DPCT_ORIG     return make_uint3(a.x + b, a.y + b, a.z + b);*/
    return sycl::uint3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator+(float b, float3 a)*/
/*
DPCT1011:95: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator+(float b, sycl::float3 a)
{
/* DPCT_ORIG     return make_float3(a.x + b, a.y + b, a.z + b);*/
    return sycl::float3(a.x() + b, a.y() + b, a.z() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator+(float4 a, float4 b)*/
/*
DPCT1011:96: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator+(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w +
 * b.w);*/
    return sycl::float4(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(),
                        a.w() + b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float4 &a, float4 b)*/
/*
DPCT1011:97: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float4 &a, sycl::float4 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
/* DPCT_ORIG     a.w += b.w;*/
    a.w() += b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator+(float4 a, float b)*/
/*
DPCT1011:98: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator+(sycl::float4 a, float b)
{
/* DPCT_ORIG     return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);*/
    return sycl::float4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator+(float b, float4 a)*/
/*
DPCT1011:99: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator+(float b, sycl::float4 a)
{
/* DPCT_ORIG     return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);*/
    return sycl::float4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(float4 &a, float b)*/
/*
DPCT1011:100: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::float4 &a, float b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
/* DPCT_ORIG     a.w += b;*/
    a.w() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator+(int4 a, int4 b)*/
/*
DPCT1011:101: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator+(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w +
 * b.w);*/
    return sycl::int4(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(),
                      a.w() + b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int4 &a, int4 b)*/
/*
DPCT1011:102: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int4 &a, sycl::int4 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
/* DPCT_ORIG     a.w += b.w;*/
    a.w() += b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator+(int4 a, int b)*/
/*
DPCT1011:103: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator+(sycl::int4 a, int b)
{
/* DPCT_ORIG     return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);*/
    return sycl::int4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator+(int b, int4 a)*/
/*
DPCT1011:104: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator+(int b, sycl::int4 a)
{
/* DPCT_ORIG     return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);*/
    return sycl::int4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(int4 &a, int b)*/
/*
DPCT1011:105: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::int4 &a, int b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
/* DPCT_ORIG     a.w += b;*/
    a.w() += b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)*/
/*
DPCT1011:106: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator+(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w +
 * b.w);*/
    return sycl::uint4(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(),
                       a.w() + b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint4 &a, uint4 b)*/
/*
DPCT1011:107: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint4 &a, sycl::uint4 b)
{
/* DPCT_ORIG     a.x += b.x;*/
    a.x() += b.x();
/* DPCT_ORIG     a.y += b.y;*/
    a.y() += b.y();
/* DPCT_ORIG     a.z += b.z;*/
    a.z() += b.z();
/* DPCT_ORIG     a.w += b.w;*/
    a.w() += b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator+(uint4 a, uint b)*/
/*
DPCT1011:108: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator+(sycl::uint4 a, uint b)
{
/* DPCT_ORIG     return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);*/
    return sycl::uint4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator+(uint b, uint4 a)*/
/*
DPCT1011:109: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator+(uint b, sycl::uint4 a)
{
/* DPCT_ORIG     return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);*/
    return sycl::uint4(a.x() + b, a.y() + b, a.z() + b, a.w() + b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator+=(uint4 &a, uint b)*/
/*
DPCT1011:110: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator+=(sycl::uint4 &a, uint b)
{
/* DPCT_ORIG     a.x += b;*/
    a.x() += b;
/* DPCT_ORIG     a.y += b;*/
    a.y() += b;
/* DPCT_ORIG     a.z += b;*/
    a.z() += b;
/* DPCT_ORIG     a.w += b;*/
    a.w() += b;
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 operator-(float2 a, float2 b)*/
/*
DPCT1011:111: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(a.x - b.x, a.y - b.y);*/
    return sycl::float2(a.x() - b.x(), a.y() - b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float2 &a, float2 b)*/
/*
DPCT1011:112: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float2 &a, sycl::float2 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator-(float2 a, float b)*/
/*
DPCT1011:113: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(sycl::float2 a, float b)
{
/* DPCT_ORIG     return make_float2(a.x - b, a.y - b);*/
    return sycl::float2(a.x() - b, a.y() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator-(float b, float2 a)*/
/*
DPCT1011:114: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(float b, sycl::float2 a)
{
/* DPCT_ORIG     return make_float2(b - a.x, b - a.y);*/
    return sycl::float2(b - a.x(), b - a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float2 &a, float b)*/
/*
DPCT1011:115: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float2 &a, float b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator-(int2 a, int2 b)*/
/*
DPCT1011:116: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator-(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(a.x - b.x, a.y - b.y);*/
    return sycl::int2(a.x() - b.x(), a.y() - b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int2 &a, int2 b)*/
/*
DPCT1011:117: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int2 &a, sycl::int2 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator-(int2 a, int b)*/
/*
DPCT1011:118: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator-(sycl::int2 a, int b)
{
/* DPCT_ORIG     return make_int2(a.x - b, a.y - b);*/
    return sycl::int2(a.x() - b, a.y() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator-(int b, int2 a)*/
/*
DPCT1011:119: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator-(int b, sycl::int2 a)
{
/* DPCT_ORIG     return make_int2(b - a.x, b - a.y);*/
    return sycl::int2(b - a.x(), b - a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int2 &a, int b)*/
/*
DPCT1011:120: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int2 &a, int b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)*/
/*
DPCT1011:121: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator-(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(a.x - b.x, a.y - b.y);*/
    return sycl::uint2(a.x() - b.x(), a.y() - b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint2 &a, uint2 b)*/
/*
DPCT1011:122: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint2 &a, sycl::uint2 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator-(uint2 a, uint b)*/
/*
DPCT1011:123: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator-(sycl::uint2 a, uint b)
{
/* DPCT_ORIG     return make_uint2(a.x - b, a.y - b);*/
    return sycl::uint2(a.x() - b, a.y() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator-(uint b, uint2 a)*/
/*
DPCT1011:124: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator-(uint b, sycl::uint2 a)
{
/* DPCT_ORIG     return make_uint2(b - a.x, b - a.y);*/
    return sycl::uint2(b - a.x(), b - a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint2 &a, uint b)*/
/*
DPCT1011:125: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint2 &a, uint b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator-(float3 a, float3 b)*/
/*
DPCT1011:126: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator-(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);*/
    return sycl::float3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float3 &a, float3 b)*/
/*
DPCT1011:127: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float3 &a, sycl::float3 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator-(float3 a, float b)*/
/*
DPCT1011:128: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator-(sycl::float3 a, float b)
{
/* DPCT_ORIG     return make_float3(a.x - b, a.y - b, a.z - b);*/
    return sycl::float3(a.x() - b, a.y() - b, a.z() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator-(float b, float3 a)*/
/*
DPCT1011:129: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator-(float b, sycl::float3 a)
{
/* DPCT_ORIG     return make_float3(b - a.x, b - a.y, b - a.z);*/
    return sycl::float3(b - a.x(), b - a.y(), b - a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float3 &a, float b)*/
/*
DPCT1011:130: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float3 &a, float b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator-(int3 a, int3 b)*/
/*
DPCT1011:131: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator-(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);*/
    return sycl::int3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int3 &a, int3 b)*/
/*
DPCT1011:132: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int3 &a, sycl::int3 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator-(int3 a, int b)*/
/*
DPCT1011:133: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator-(sycl::int3 a, int b)
{
/* DPCT_ORIG     return make_int3(a.x - b, a.y - b, a.z - b);*/
    return sycl::int3(a.x() - b, a.y() - b, a.z() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator-(int b, int3 a)*/
/*
DPCT1011:134: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator-(int b, sycl::int3 a)
{
/* DPCT_ORIG     return make_int3(b - a.x, b - a.y, b - a.z);*/
    return sycl::int3(b - a.x(), b - a.y(), b - a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int3 &a, int b)*/
/*
DPCT1011:135: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int3 &a, int b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)*/
/*
DPCT1011:136: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator-(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);*/
    return sycl::uint3(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint3 &a, uint3 b)*/
/*
DPCT1011:137: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint3 &a, sycl::uint3 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator-(uint3 a, uint b)*/
/*
DPCT1011:138: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator-(sycl::uint3 a, uint b)
{
/* DPCT_ORIG     return make_uint3(a.x - b, a.y - b, a.z - b);*/
    return sycl::uint3(a.x() - b, a.y() - b, a.z() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator-(uint b, uint3 a)*/
/*
DPCT1011:139: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator-(uint b, sycl::uint3 a)
{
/* DPCT_ORIG     return make_uint3(b - a.x, b - a.y, b - a.z);*/
    return sycl::uint3(b - a.x(), b - a.y(), b - a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint3 &a, uint b)*/
/*
DPCT1011:140: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint3 &a, uint b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator-(float4 a, float4 b)*/
/*
DPCT1011:141: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator-(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w -
 * b.w);*/
    return sycl::float4(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(),
                        a.w() - b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float4 &a, float4 b)*/
/*
DPCT1011:142: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float4 &a, sycl::float4 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
/* DPCT_ORIG     a.w -= b.w;*/
    a.w() -= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator-(float4 a, float b)*/
/*
DPCT1011:143: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator-(sycl::float4 a, float b)
{
/* DPCT_ORIG     return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);*/
    return sycl::float4(a.x() - b, a.y() - b, a.z() - b, a.w() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(float4 &a, float b)*/
/*
DPCT1011:144: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::float4 &a, float b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
/* DPCT_ORIG     a.w -= b;*/
    a.w() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator-(int4 a, int4 b)*/
/*
DPCT1011:145: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator-(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w -
 * b.w);*/
    return sycl::int4(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(),
                      a.w() - b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int4 &a, int4 b)*/
/*
DPCT1011:146: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int4 &a, sycl::int4 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
/* DPCT_ORIG     a.w -= b.w;*/
    a.w() -= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator-(int4 a, int b)*/
/*
DPCT1011:147: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator-(sycl::int4 a, int b)
{
/* DPCT_ORIG     return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);*/
    return sycl::int4(a.x() - b, a.y() - b, a.z() - b, a.w() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator-(int b, int4 a)*/
/*
DPCT1011:148: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator-(int b, sycl::int4 a)
{
/* DPCT_ORIG     return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);*/
    return sycl::int4(b - a.x(), b - a.y(), b - a.z(), b - a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(int4 &a, int b)*/
/*
DPCT1011:149: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::int4 &a, int b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
/* DPCT_ORIG     a.w -= b;*/
    a.w() -= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)*/
/*
DPCT1011:150: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator-(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w -
 * b.w);*/
    return sycl::uint4(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(),
                       a.w() - b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint4 &a, uint4 b)*/
/*
DPCT1011:151: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint4 &a, sycl::uint4 b)
{
/* DPCT_ORIG     a.x -= b.x;*/
    a.x() -= b.x();
/* DPCT_ORIG     a.y -= b.y;*/
    a.y() -= b.y();
/* DPCT_ORIG     a.z -= b.z;*/
    a.z() -= b.z();
/* DPCT_ORIG     a.w -= b.w;*/
    a.w() -= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator-(uint4 a, uint b)*/
/*
DPCT1011:152: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator-(sycl::uint4 a, uint b)
{
/* DPCT_ORIG     return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);*/
    return sycl::uint4(a.x() - b, a.y() - b, a.z() - b, a.w() - b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator-(uint b, uint4 a)*/
/*
DPCT1011:153: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator-(uint b, sycl::uint4 a)
{
/* DPCT_ORIG     return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);*/
    return sycl::uint4(b - a.x(), b - a.y(), b - a.z(), b - a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator-=(uint4 &a, uint b)*/
/*
DPCT1011:154: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator-=(sycl::uint4 &a, uint b)
{
/* DPCT_ORIG     a.x -= b;*/
    a.x() -= b;
/* DPCT_ORIG     a.y -= b;*/
    a.y() -= b;
/* DPCT_ORIG     a.z -= b;*/
    a.z() -= b;
/* DPCT_ORIG     a.w -= b;*/
    a.w() -= b;
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 operator*(float2 a, float2 b)*/
/*
DPCT1011:155: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(a.x * b.x, a.y * b.y);*/
    return sycl::float2(a.x() * b.x(), a.y() * b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float2 &a, float2 b)*/
/*
DPCT1011:156: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float2 &a, sycl::float2 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator*(float2 a, float b)*/
/*
DPCT1011:157: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(sycl::float2 a, float b)
{
/* DPCT_ORIG     return make_float2(a.x * b, a.y * b);*/
    return sycl::float2(a.x() * b, a.y() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator*(float b, float2 a)*/
/*
DPCT1011:158: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(float b, sycl::float2 a)
{
/* DPCT_ORIG     return make_float2(b * a.x, b * a.y);*/
    return sycl::float2(b * a.x(), b * a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float2 &a, float b)*/
/*
DPCT1011:159: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float2 &a, float b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator*(int2 a, int2 b)*/
/*
DPCT1011:160: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator*(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(a.x * b.x, a.y * b.y);*/
    return sycl::int2(a.x() * b.x(), a.y() * b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int2 &a, int2 b)*/
/*
DPCT1011:161: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int2 &a, sycl::int2 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator*(int2 a, int b)*/
/*
DPCT1011:162: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator*(sycl::int2 a, int b)
{
/* DPCT_ORIG     return make_int2(a.x * b, a.y * b);*/
    return sycl::int2(a.x() * b, a.y() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int2 operator*(int b, int2 a)*/
/*
DPCT1011:163: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int2 operator*(int b, sycl::int2 a)
{
/* DPCT_ORIG     return make_int2(b * a.x, b * a.y);*/
    return sycl::int2(b * a.x(), b * a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int2 &a, int b)*/
/*
DPCT1011:164: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int2 &a, int b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)*/
/*
DPCT1011:165: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator*(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(a.x * b.x, a.y * b.y);*/
    return sycl::uint2(a.x() * b.x(), a.y() * b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint2 &a, uint2 b)*/
/*
DPCT1011:166: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint2 &a, sycl::uint2 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator*(uint2 a, uint b)*/
/*
DPCT1011:167: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator*(sycl::uint2 a, uint b)
{
/* DPCT_ORIG     return make_uint2(a.x * b, a.y * b);*/
    return sycl::uint2(a.x() * b, a.y() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint2 operator*(uint b, uint2 a)*/
/*
DPCT1011:168: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint2 operator*(uint b, sycl::uint2 a)
{
/* DPCT_ORIG     return make_uint2(b * a.x, b * a.y);*/
    return sycl::uint2(b * a.x(), b * a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint2 &a, uint b)*/
/*
DPCT1011:169: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint2 &a, uint b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator*(float3 a, float3 b)*/
/*
DPCT1011:170: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator*(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);*/
    return sycl::float3(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float3 &a, float3 b)*/
/*
DPCT1011:171: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float3 &a, sycl::float3 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator*(float3 a, float b)*/
/*
DPCT1011:172: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator*(sycl::float3 a, float b)
{
/* DPCT_ORIG     return make_float3(a.x * b, a.y * b, a.z * b);*/
    return sycl::float3(a.x() * b, a.y() * b, a.z() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator*(float b, float3 a)*/
/*
DPCT1011:173: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator*(float b, sycl::float3 a)
{
/* DPCT_ORIG     return make_float3(b * a.x, b * a.y, b * a.z);*/
    return sycl::float3(b * a.x(), b * a.y(), b * a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float3 &a, float b)*/
/*
DPCT1011:174: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float3 &a, float b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator*(int3 a, int3 b)*/
/*
DPCT1011:175: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator*(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);*/
    return sycl::int3(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int3 &a, int3 b)*/
/*
DPCT1011:176: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int3 &a, sycl::int3 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator*(int3 a, int b)*/
/*
DPCT1011:177: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator*(sycl::int3 a, int b)
{
/* DPCT_ORIG     return make_int3(a.x * b, a.y * b, a.z * b);*/
    return sycl::int3(a.x() * b, a.y() * b, a.z() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int3 operator*(int b, int3 a)*/
/*
DPCT1011:178: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int3 operator*(int b, sycl::int3 a)
{
/* DPCT_ORIG     return make_int3(b * a.x, b * a.y, b * a.z);*/
    return sycl::int3(b * a.x(), b * a.y(), b * a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int3 &a, int b)*/
/*
DPCT1011:179: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int3 &a, int b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)*/
/*
DPCT1011:180: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator*(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);*/
    return sycl::uint3(a.x() * b.x(), a.y() * b.y(), a.z() * b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint3 &a, uint3 b)*/
/*
DPCT1011:181: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint3 &a, sycl::uint3 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator*(uint3 a, uint b)*/
/*
DPCT1011:182: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator*(sycl::uint3 a, uint b)
{
/* DPCT_ORIG     return make_uint3(a.x * b, a.y * b, a.z * b);*/
    return sycl::uint3(a.x() * b, a.y() * b, a.z() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint3 operator*(uint b, uint3 a)*/
/*
DPCT1011:183: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint3 operator*(uint b, sycl::uint3 a)
{
/* DPCT_ORIG     return make_uint3(b * a.x, b * a.y, b * a.z);*/
    return sycl::uint3(b * a.x(), b * a.y(), b * a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint3 &a, uint b)*/
/*
DPCT1011:184: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint3 &a, uint b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator*(float4 a, float4 b)*/
/*
DPCT1011:185: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator*(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w *
 * b.w);*/
    return sycl::float4(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(),
                        a.w() * b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float4 &a, float4 b)*/
/*
DPCT1011:186: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float4 &a, sycl::float4 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
/* DPCT_ORIG     a.w *= b.w;*/
    a.w() *= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator*(float4 a, float b)*/
/*
DPCT1011:187: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator*(sycl::float4 a, float b)
{
/* DPCT_ORIG     return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);*/
    return sycl::float4(a.x() * b, a.y() * b, a.z() * b, a.w() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator*(float b, float4 a)*/
/*
DPCT1011:188: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator*(float b, sycl::float4 a)
{
/* DPCT_ORIG     return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);*/
    return sycl::float4(b * a.x(), b * a.y(), b * a.z(), b * a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(float4 &a, float b)*/
/*
DPCT1011:189: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::float4 &a, float b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
/* DPCT_ORIG     a.w *= b;*/
    a.w() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator*(int4 a, int4 b)*/
/*
DPCT1011:190: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator*(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w *
 * b.w);*/
    return sycl::int4(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(),
                      a.w() * b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int4 &a, int4 b)*/
/*
DPCT1011:191: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int4 &a, sycl::int4 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
/* DPCT_ORIG     a.w *= b.w;*/
    a.w() *= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator*(int4 a, int b)*/
/*
DPCT1011:192: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator*(sycl::int4 a, int b)
{
/* DPCT_ORIG     return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);*/
    return sycl::int4(a.x() * b, a.y() * b, a.z() * b, a.w() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ int4 operator*(int b, int4 a)*/
/*
DPCT1011:193: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::int4 operator*(int b, sycl::int4 a)
{
/* DPCT_ORIG     return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);*/
    return sycl::int4(b * a.x(), b * a.y(), b * a.z(), b * a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(int4 &a, int b)*/
/*
DPCT1011:194: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::int4 &a, int b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
/* DPCT_ORIG     a.w *= b;*/
    a.w() *= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)*/
/*
DPCT1011:195: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator*(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w *
 * b.w);*/
    return sycl::uint4(a.x() * b.x(), a.y() * b.y(), a.z() * b.z(),
                       a.w() * b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint4 &a, uint4 b)*/
/*
DPCT1011:196: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint4 &a, sycl::uint4 b)
{
/* DPCT_ORIG     a.x *= b.x;*/
    a.x() *= b.x();
/* DPCT_ORIG     a.y *= b.y;*/
    a.y() *= b.y();
/* DPCT_ORIG     a.z *= b.z;*/
    a.z() *= b.z();
/* DPCT_ORIG     a.w *= b.w;*/
    a.w() *= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator*(uint4 a, uint b)*/
/*
DPCT1011:197: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator*(sycl::uint4 a, uint b)
{
/* DPCT_ORIG     return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);*/
    return sycl::uint4(a.x() * b, a.y() * b, a.z() * b, a.w() * b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ uint4 operator*(uint b, uint4 a)*/
/*
DPCT1011:198: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::uint4 operator*(uint b, sycl::uint4 a)
{
/* DPCT_ORIG     return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);*/
    return sycl::uint4(b * a.x(), b * a.y(), b * a.z(), b * a.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator*=(uint4 &a, uint b)*/
/*
DPCT1011:199: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator*=(sycl::uint4 &a, uint b)
{
/* DPCT_ORIG     a.x *= b;*/
    a.x() *= b;
/* DPCT_ORIG     a.y *= b;*/
    a.y() *= b;
/* DPCT_ORIG     a.z *= b;*/
    a.z() *= b;
/* DPCT_ORIG     a.w *= b;*/
    a.w() *= b;
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 operator/(float2 a, float2 b)*/
/*
DPCT1011:200: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(a.x / b.x, a.y / b.y);*/
    return sycl::float2(a.x() / b.x(), a.y() / b.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float2 &a, float2 b)*/
/*
DPCT1011:201: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float2 &a, sycl::float2 b)
{
/* DPCT_ORIG     a.x /= b.x;*/
    a.x() /= b.x();
/* DPCT_ORIG     a.y /= b.y;*/
    a.y() /= b.y();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator/(float2 a, float b)*/
/*
DPCT1011:202: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(sycl::float2 a, float b)
{
/* DPCT_ORIG     return make_float2(a.x / b, a.y / b);*/
    return sycl::float2(a.x() / b, a.y() / b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float2 &a, float b)*/
/*
DPCT1011:203: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float2 &a, float b)
{
/* DPCT_ORIG     a.x /= b;*/
    a.x() /= b;
/* DPCT_ORIG     a.y /= b;*/
    a.y() /= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float2 operator/(float b, float2 a)*/
/*
DPCT1011:204: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(float b, sycl::float2 a)
{
/* DPCT_ORIG     return make_float2(b / a.x, b / a.y);*/
    return sycl::float2(b / a.x(), b / a.y());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator/(float3 a, float3 b)*/
/*
DPCT1011:205: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator/(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);*/
    return sycl::float3(a.x() / b.x(), a.y() / b.y(), a.z() / b.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float3 &a, float3 b)*/
/*
DPCT1011:206: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float3 &a, sycl::float3 b)
{
/* DPCT_ORIG     a.x /= b.x;*/
    a.x() /= b.x();
/* DPCT_ORIG     a.y /= b.y;*/
    a.y() /= b.y();
/* DPCT_ORIG     a.z /= b.z;*/
    a.z() /= b.z();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator/(float3 a, float b)*/
/*
DPCT1011:207: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator/(sycl::float3 a, float b)
{
/* DPCT_ORIG     return make_float3(a.x / b, a.y / b, a.z / b);*/
    return sycl::float3(a.x() / b, a.y() / b, a.z() / b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float3 &a, float b)*/
/*
DPCT1011:208: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float3 &a, float b)
{
/* DPCT_ORIG     a.x /= b;*/
    a.x() /= b;
/* DPCT_ORIG     a.y /= b;*/
    a.y() /= b;
/* DPCT_ORIG     a.z /= b;*/
    a.z() /= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float3 operator/(float b, float3 a)*/
/*
DPCT1011:209: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float3 operator/(float b, sycl::float3 a)
{
/* DPCT_ORIG     return make_float3(b / a.x, b / a.y, b / a.z);*/
    return sycl::float3(b / a.x(), b / a.y(), b / a.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator/(float4 a, float4 b)*/
/*
DPCT1011:210: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator/(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w /
 * b.w);*/
    return sycl::float4(a.x() / b.x(), a.y() / b.y(), a.z() / b.z(),
                        a.w() / b.w());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float4 &a, float4 b)*/
/*
DPCT1011:211: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float4 &a, sycl::float4 b)
{
/* DPCT_ORIG     a.x /= b.x;*/
    a.x() /= b.x();
/* DPCT_ORIG     a.y /= b.y;*/
    a.y() /= b.y();
/* DPCT_ORIG     a.z /= b.z;*/
    a.z() /= b.z();
/* DPCT_ORIG     a.w /= b.w;*/
    a.w() /= b.w();
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator/(float4 a, float b)*/
/*
DPCT1011:212: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator/(sycl::float4 a, float b)
{
/* DPCT_ORIG     return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);*/
    return sycl::float4(a.x() / b, a.y() / b, a.z() / b, a.w() / b);
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ void operator/=(float4 &a, float b)*/
/*
DPCT1011:213: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline void operator/=(sycl::float4 &a, float b)
{
/* DPCT_ORIG     a.x /= b;*/
    a.x() /= b;
/* DPCT_ORIG     a.y /= b;*/
    a.y() /= b;
/* DPCT_ORIG     a.z /= b;*/
    a.z() /= b;
/* DPCT_ORIG     a.w /= b;*/
    a.w() /= b;
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ float4 operator/(float b, float4 a)*/
/*
DPCT1011:214: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float4 operator/(float b, sycl::float4 a)
{
/* DPCT_ORIG     return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);*/
    return sycl::float4(b / a.x(), b / a.y(), b / a.z(), b / a.w());
}
} // namespace dpct_operator_overloading

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline  __host__ __device__ float2 fminf(float2 a, float2 b)*/
inline sycl::float2 fminf(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));*/
    return sycl::float2(sycl::fmin(a.x(), b.x()), sycl::fmin(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 fminf(float3 a, float3 b)*/
inline sycl::float3 fminf(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(fminf(a.x,b.x), fminf(a.y,b.y),
 * fminf(a.z,b.z));*/
    return sycl::float3(sycl::fmin(a.x(), b.x()), sycl::fmin(a.y(), b.y()),
                        sycl::fmin(a.z(), b.z()));
}
/* DPCT_ORIG inline  __host__ __device__ float4 fminf(float4 a, float4 b)*/
inline sycl::float4 fminf(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(fminf(a.x,b.x), fminf(a.y,b.y),
 * fminf(a.z,b.z), fminf(a.w,b.w));*/
    return sycl::float4(sycl::fmin(a.x(), b.x()), sycl::fmin(a.y(), b.y()),
                        sycl::fmin(a.z(), b.z()), sycl::fmin(a.w(), b.w()));
}

/* DPCT_ORIG inline __host__ __device__ int2 min(int2 a, int2 b)*/
inline sycl::int2 min(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(min(a.x,b.x), min(a.y,b.y));*/
    return sycl::int2(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ int3 min(int3 a, int3 b)*/
inline sycl::int3 min(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));*/
    return sycl::int3(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()),
                      sycl::min(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ int4 min(int4 a, int4 b)*/
inline sycl::int4 min(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z),
 * min(a.w,b.w));*/
    return sycl::int4(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()),
                      sycl::min(a.z(), b.z()), sycl::min(a.w(), b.w()));
}

/* DPCT_ORIG inline __host__ __device__ uint2 min(uint2 a, uint2 b)*/
inline sycl::uint2 min(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(min(a.x,b.x), min(a.y,b.y));*/
    return sycl::uint2(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ uint3 min(uint3 a, uint3 b)*/
inline sycl::uint3 min(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));*/
    return sycl::uint3(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()),
                       sycl::min(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ uint4 min(uint4 a, uint4 b)*/
inline sycl::uint4 min(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z),
 * min(a.w,b.w));*/
    return sycl::uint4(sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y()),
                       sycl::min(a.z(), b.z()), sycl::min(a.w(), b.w()));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 fmaxf(float2 a, float2 b)*/
inline sycl::float2 fmaxf(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));*/
    return sycl::float2(sycl::fmax(a.x(), b.x()), sycl::fmax(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 fmaxf(float3 a, float3 b)*/
inline sycl::float3 fmaxf(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y),
 * fmaxf(a.z,b.z));*/
    return sycl::float3(sycl::fmax(a.x(), b.x()), sycl::fmax(a.y(), b.y()),
                        sycl::fmax(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ float4 fmaxf(float4 a, float4 b)*/
inline sycl::float4 fmaxf(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y),
 * fmaxf(a.z,b.z), fmaxf(a.w,b.w));*/
    return sycl::float4(sycl::fmax(a.x(), b.x()), sycl::fmax(a.y(), b.y()),
                        sycl::fmax(a.z(), b.z()), sycl::fmax(a.w(), b.w()));
}

/* DPCT_ORIG inline __host__ __device__ int2 max(int2 a, int2 b)*/
inline sycl::int2 max(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(max(a.x,b.x), max(a.y,b.y));*/
    return sycl::int2(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ int3 max(int3 a, int3 b)*/
inline sycl::int3 max(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));*/
    return sycl::int3(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()),
                      sycl::max(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ int4 max(int4 a, int4 b)*/
inline sycl::int4 max(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z),
 * max(a.w,b.w));*/
    return sycl::int4(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()),
                      sycl::max(a.z(), b.z()), sycl::max(a.w(), b.w()));
}

/* DPCT_ORIG inline __host__ __device__ uint2 max(uint2 a, uint2 b)*/
inline sycl::uint2 max(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(max(a.x,b.x), max(a.y,b.y));*/
    return sycl::uint2(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ uint3 max(uint3 a, uint3 b)*/
inline sycl::uint3 max(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));*/
    return sycl::uint3(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()),
                       sycl::max(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ uint4 max(uint4 a, uint4 b)*/
inline sycl::uint4 max(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z),
 * max(a.w,b.w));*/
    return sycl::uint4(sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y()),
                       sycl::max(a.z(), b.z()), sycl::max(a.w(), b.w()));
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __device__ __host__ float lerp(float a, float b, float t)*/
inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
/* DPCT_ORIG inline __device__ __host__ float2 lerp(float2 a, float2 b, float
 * t)*/
inline sycl::float2 lerp(sycl::float2 a, sycl::float2 b, float t)
{
/* DPCT_ORIG     return a + t*(b-a);*/
    return dpct_operator_overloading::operator+(
        a, dpct_operator_overloading::operator*(
               t, (dpct_operator_overloading::operator-(b, a))));
}
/* DPCT_ORIG inline __device__ __host__ float3 lerp(float3 a, float3 b, float
 * t)*/
inline sycl::float3 lerp(sycl::float3 a, sycl::float3 b, float t)
{
/* DPCT_ORIG     return a + t*(b-a);*/
    return dpct_operator_overloading::operator+(
        a, dpct_operator_overloading::operator*(
               t, (dpct_operator_overloading::operator-(b, a))));
}
/* DPCT_ORIG inline __device__ __host__ float4 lerp(float4 a, float4 b, float
 * t)*/
inline sycl::float4 lerp(sycl::float4 a, sycl::float4 b, float t)
{
/* DPCT_ORIG     return a + t*(b-a);*/
    return dpct_operator_overloading::operator+(
        a, dpct_operator_overloading::operator*(
               t, (dpct_operator_overloading::operator-(b, a))));
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __device__ __host__ float clamp(float f, float a, float b)*/
inline float clamp(float f, float a, float b)
{
/* DPCT_ORIG     return fmaxf(a, fminf(f, b));*/
    return sycl::fmax(a, sycl::fmin(f, b));
}
/* DPCT_ORIG inline __device__ __host__ int clamp(int f, int a, int b)*/
inline int clamp(int f, int a, int b)
{
/* DPCT_ORIG     return max(a, min(f, b));*/
    return sycl::max(a, sycl::min(f, b));
}
/* DPCT_ORIG inline __device__ __host__ uint clamp(uint f, uint a, uint b)*/
inline uint clamp(uint f, uint a, uint b)
{
/* DPCT_ORIG     return max(a, min(f, b));*/
    return sycl::max(a, sycl::min(f, b));
}

/* DPCT_ORIG inline __device__ __host__ float2 clamp(float2 v, float a, float
 * b)*/
inline sycl::float2 clamp(sycl::float2 v, float a, float b)
{
/* DPCT_ORIG     return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));*/
    return sycl::float2(clamp(v.x(), a, b), clamp(v.y(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ float2 clamp(float2 v, float2 a, float2
 * b)*/
inline sycl::float2 clamp(sycl::float2 v, sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y,
 * b.y));*/
    return sycl::float2(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()));
}
/* DPCT_ORIG inline __device__ __host__ float3 clamp(float3 v, float a, float
 * b)*/
inline sycl::float3 clamp(sycl::float3 v, float a, float b)
{
/* DPCT_ORIG     return make_float3(clamp(v.x, a, b), clamp(v.y, a, b),
 * clamp(v.z, a, b));*/
    return sycl::float3(clamp(v.x(), a, b), clamp(v.y(), a, b),
                        clamp(v.z(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ float3 clamp(float3 v, float3 a, float3
 * b)*/
inline sycl::float3 clamp(sycl::float3 v, sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z));*/
    return sycl::float3(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                        clamp(v.z(), a.z(), b.z()));
}
/* DPCT_ORIG inline __device__ __host__ float4 clamp(float4 v, float a, float
 * b)*/
inline sycl::float4 clamp(sycl::float4 v, float a, float b)
{
/* DPCT_ORIG     return make_float4(clamp(v.x, a, b), clamp(v.y, a, b),
 * clamp(v.z, a, b), clamp(v.w, a, b));*/
    return sycl::float4(clamp(v.x(), a, b), clamp(v.y(), a, b),
                        clamp(v.z(), a, b), clamp(v.w(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ float4 clamp(float4 v, float4 a, float4
 * b)*/
inline sycl::float4 clamp(sycl::float4 v, sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));*/
    return sycl::float4(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                        clamp(v.z(), a.z(), b.z()), clamp(v.w(), a.w(), b.w()));
}

/* DPCT_ORIG inline __device__ __host__ int2 clamp(int2 v, int a, int b)*/
inline sycl::int2 clamp(sycl::int2 v, int a, int b)
{
/* DPCT_ORIG     return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));*/
    return sycl::int2(clamp(v.x(), a, b), clamp(v.y(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)*/
inline sycl::int2 clamp(sycl::int2 v, sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));*/
    return sycl::int2(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()));
}
/* DPCT_ORIG inline __device__ __host__ int3 clamp(int3 v, int a, int b)*/
inline sycl::int3 clamp(sycl::int3 v, int a, int b)
{
/* DPCT_ORIG     return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z,
 * a, b));*/
    return sycl::int3(clamp(v.x(), a, b), clamp(v.y(), a, b),
                      clamp(v.z(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)*/
inline sycl::int3 clamp(sycl::int3 v, sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z));*/
    return sycl::int3(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                      clamp(v.z(), a.z(), b.z()));
}
/* DPCT_ORIG inline __device__ __host__ int4 clamp(int4 v, int a, int b)*/
inline sycl::int4 clamp(sycl::int4 v, int a, int b)
{
/* DPCT_ORIG     return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z,
 * a, b), clamp(v.w, a, b));*/
    return sycl::int4(clamp(v.x(), a, b), clamp(v.y(), a, b),
                      clamp(v.z(), a, b), clamp(v.w(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)*/
inline sycl::int4 clamp(sycl::int4 v, sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));*/
    return sycl::int4(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                      clamp(v.z(), a.z(), b.z()), clamp(v.w(), a.w(), b.w()));
}

/* DPCT_ORIG inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)*/
inline sycl::uint2 clamp(sycl::uint2 v, uint a, uint b)
{
/* DPCT_ORIG     return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));*/
    return sycl::uint2(clamp(v.x(), a, b), clamp(v.y(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)*/
inline sycl::uint2 clamp(sycl::uint2 v, sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y,
 * b.y));*/
    return sycl::uint2(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()));
}
/* DPCT_ORIG inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)*/
inline sycl::uint3 clamp(sycl::uint3 v, uint a, uint b)
{
/* DPCT_ORIG     return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b),
 * clamp(v.z, a, b));*/
    return sycl::uint3(clamp(v.x(), a, b), clamp(v.y(), a, b),
                       clamp(v.z(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)*/
inline sycl::uint3 clamp(sycl::uint3 v, sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z));*/
    return sycl::uint3(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                       clamp(v.z(), a.z(), b.z()));
}
/* DPCT_ORIG inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)*/
inline sycl::uint4 clamp(sycl::uint4 v, uint a, uint b)
{
/* DPCT_ORIG     return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b),
 * clamp(v.z, a, b), clamp(v.w, a, b));*/
    return sycl::uint4(clamp(v.x(), a, b), clamp(v.y(), a, b),
                       clamp(v.z(), a, b), clamp(v.w(), a, b));
}
/* DPCT_ORIG inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)*/
inline sycl::uint4 clamp(sycl::uint4 v, sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
 * clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));*/
    return sycl::uint4(clamp(v.x(), a.x(), b.x()), clamp(v.y(), a.y(), b.y()),
                       clamp(v.z(), a.z(), b.z()), clamp(v.w(), a.w(), b.w()));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float dot(float2 a, float2 b)*/
inline float dot(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y;*/
    return a.x() * b.x() + a.y() * b.y();
}
/* DPCT_ORIG inline __host__ __device__ float dot(float3 a, float3 b)*/
inline float dot(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
/* DPCT_ORIG inline __host__ __device__ float dot(float4 a, float4 b)*/
inline float dot(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

/* DPCT_ORIG inline __host__ __device__ int dot(int2 a, int2 b)*/
inline int dot(sycl::int2 a, sycl::int2 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y;*/
    return a.x() * b.x() + a.y() * b.y();
}
/* DPCT_ORIG inline __host__ __device__ int dot(int3 a, int3 b)*/
inline int dot(sycl::int3 a, sycl::int3 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
/* DPCT_ORIG inline __host__ __device__ int dot(int4 a, int4 b)*/
inline int dot(sycl::int4 a, sycl::int4 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

/* DPCT_ORIG inline __host__ __device__ uint dot(uint2 a, uint2 b)*/
inline uint dot(sycl::uint2 a, sycl::uint2 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y;*/
    return a.x() * b.x() + a.y() * b.y();
}
/* DPCT_ORIG inline __host__ __device__ uint dot(uint3 a, uint3 b)*/
inline uint dot(sycl::uint3 a, sycl::uint3 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
/* DPCT_ORIG inline __host__ __device__ uint dot(uint4 a, uint4 b)*/
inline uint dot(sycl::uint4 a, sycl::uint4 b)
{
/* DPCT_ORIG     return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;*/
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float length(float2 v)*/
inline float length(sycl::float2 v)
{
/* DPCT_ORIG     return sqrtf(dot(v, v));*/
    return sycl::sqrt(dot(v, v));
}
/* DPCT_ORIG inline __host__ __device__ float squared_length(float3 v)*/
inline float squared_length(sycl::float3 v)
{
/* DPCT_ORIG 	return v.x*v.x + v.y*v.y + v.z*v.z;*/
        return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}
/* DPCT_ORIG inline __host__ __device__ float length(float3 v)*/
inline float length(sycl::float3 v)
{
/* DPCT_ORIG     return sqrtf(dot(v, v));*/
    return sycl::sqrt(dot(v, v));
}
/* DPCT_ORIG inline __host__ __device__ float length(float4 v)*/
inline float length(sycl::float4 v)
{
/* DPCT_ORIG     return sqrtf(dot(v, v));*/
    return sycl::sqrt(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 normalize(float2 v)*/
inline sycl::float2 normalize(sycl::float2 v)
{
/* DPCT_ORIG     float invLen = rsqrtf(dot(v, v));*/
    float invLen = sycl::rsqrt(dot(v, v));
/* DPCT_ORIG     return v * invLen;*/
    return dpct_operator_overloading::operator*(v, invLen);
}
/* DPCT_ORIG inline __host__ __device__ float3 normalize(float3 v)*/
inline sycl::float3 normalize(sycl::float3 v)
{
/* DPCT_ORIG     float invLen = rsqrtf(dot(v, v));*/
    float invLen = sycl::rsqrt(dot(v, v));
/* DPCT_ORIG     return v * invLen;*/
    return dpct_operator_overloading::operator*(v, invLen);
}
/* DPCT_ORIG inline __host__ __device__ float4 normalize(float4 v)*/
inline sycl::float4 normalize(sycl::float4 v)
{
/* DPCT_ORIG     float invLen = rsqrtf(dot(v, v));*/
    float invLen = sycl::rsqrt(dot(v, v));
/* DPCT_ORIG     return v * invLen;*/
    return dpct_operator_overloading::operator*(v, invLen);
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 floorf(float2 v)*/
inline sycl::float2 floorf(sycl::float2 v)
{
/* DPCT_ORIG     return make_float2(floorf(v.x), floorf(v.y));*/
    return sycl::float2(sycl::floor(v.x()), sycl::floor(v.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 floorf(float3 v)*/
inline sycl::float3 floorf(sycl::float3 v)
{
/* DPCT_ORIG     return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));*/
    return sycl::float3(sycl::floor(v.x()), sycl::floor(v.y()),
                        sycl::floor(v.z()));
}
/* DPCT_ORIG inline __host__ __device__ float4 floorf(float4 v)*/
inline sycl::float4 floorf(sycl::float4 v)
{
/* DPCT_ORIG     return make_float4(floorf(v.x), floorf(v.y), floorf(v.z),
 * floorf(v.w));*/
    return sycl::float4(sycl::floor(v.x()), sycl::floor(v.y()),
                        sycl::floor(v.z()), sycl::floor(v.w()));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float fracf(float v)*/
inline float fracf(float v)
{
/* DPCT_ORIG     return v - floorf(v);*/
    return v - sycl::floor(v);
}
/* DPCT_ORIG inline __host__ __device__ float2 fracf(float2 v)*/
inline sycl::float2 fracf(sycl::float2 v)
{
/* DPCT_ORIG     return make_float2(fracf(v.x), fracf(v.y));*/
    return sycl::float2(fracf(v.x()), fracf(v.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 fracf(float3 v)*/
inline sycl::float3 fracf(sycl::float3 v)
{
/* DPCT_ORIG     return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));*/
    return sycl::float3(fracf(v.x()), fracf(v.y()), fracf(v.z()));
}
/* DPCT_ORIG inline __host__ __device__ float4 fracf(float4 v)*/
inline sycl::float4 fracf(sycl::float4 v)
{
/* DPCT_ORIG     return make_float4(fracf(v.x), fracf(v.y), fracf(v.z),
 * fracf(v.w));*/
    return sycl::float4(fracf(v.x()), fracf(v.y()), fracf(v.z()), fracf(v.w()));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 fmodf(float2 a, float2 b)*/
inline sycl::float2 fmodf(sycl::float2 a, sycl::float2 b)
{
/* DPCT_ORIG     return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));*/
    return sycl::float2(sycl::fmod(a.x(), b.x()), sycl::fmod(a.y(), b.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 fmodf(float3 a, float3 b)*/
inline sycl::float3 fmodf(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z,
 * b.z));*/
    return sycl::float3(sycl::fmod(a.x(), b.x()), sycl::fmod(a.y(), b.y()),
                        sycl::fmod(a.z(), b.z()));
}
/* DPCT_ORIG inline __host__ __device__ float4 fmodf(float4 a, float4 b)*/
inline sycl::float4 fmodf(sycl::float4 a, sycl::float4 b)
{
/* DPCT_ORIG     return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z,
 * b.z), fmodf(a.w, b.w));*/
    return sycl::float4(sycl::fmod(a.x(), b.x()), sycl::fmod(a.y(), b.y()),
                        sycl::fmod(a.z(), b.z()), sycl::fmod(a.w(), b.w()));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float2 fabs(float2 v)*/
inline sycl::float2 fabs(sycl::float2 v)
{
/* DPCT_ORIG     return make_float2(fabs(v.x), fabs(v.y));*/
    return sycl::float2(sycl::fabs(v.x()), sycl::fabs(v.y()));
}
/* DPCT_ORIG inline __host__ __device__ float3 fabs(float3 v)*/
inline sycl::float3 fabs(sycl::float3 v)
{
/* DPCT_ORIG     return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));*/
    return sycl::float3(sycl::fabs(v.x()), sycl::fabs(v.y()),
                        sycl::fabs(v.z()));
}
/* DPCT_ORIG inline __host__ __device__ float4 fabs(float4 v)*/
inline sycl::float4 fabs(sycl::float4 v)
{
/* DPCT_ORIG     return make_float4(fabs(v.x), fabs(v.y), fabs(v.z),
 * fabs(v.w));*/
    return sycl::float4(sycl::fabs(v.x()), sycl::fabs(v.y()), sycl::fabs(v.z()),
                        sycl::fabs(v.w()));
}

/* DPCT_ORIG inline __host__ __device__ int2 abs(int2 v)*/
inline sycl::int2 abs(sycl::int2 v)
{
/* DPCT_ORIG     return make_int2(abs(v.x), abs(v.y));*/
    return sycl::int2(abs(v.x()), abs(v.y()));
}
/* DPCT_ORIG inline __host__ __device__ int3 abs(int3 v)*/
inline sycl::int3 abs(sycl::int3 v)
{
/* DPCT_ORIG     return make_int3(abs(v.x), abs(v.y), abs(v.z));*/
    return sycl::int3(abs(v.x()), abs(v.y()), abs(v.z()));
}
/* DPCT_ORIG inline __host__ __device__ int4 abs(int4 v)*/
inline sycl::int4 abs(sycl::int4 v)
{
/* DPCT_ORIG     return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));*/
    return sycl::int4(abs(v.x()), abs(v.y()), abs(v.z()), abs(v.w()));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float3 reflect(float3 i, float3 n)*/
inline sycl::float3 reflect(sycl::float3 i, sycl::float3 n)
{
/* DPCT_ORIG     return i - 2.0f * n * dot(n,i);*/
    return dpct_operator_overloading::operator-(
        i, dpct_operator_overloading::operator*(
               dpct_operator_overloading::operator*(2.0f, n), dot(n, i)));
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float3 cross(float3 a, float3 b)*/
inline sycl::float3 cross(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG     return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z,
 * a.x*b.y - a.y*b.x);*/
    return sycl::float3(a.y() * b.z() - a.z() * b.y(),
                        a.z() * b.x() - a.x() * b.z(),
                        a.x() * b.y() - a.y() * b.x());
}

////////////////////////////////////////////////////////////////////////////////
// pow
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float3 powf(float3 a, float3 b)*/
inline sycl::float3 powf(sycl::float3 a, sycl::float3 b)
{
/* DPCT_ORIG 	return make_float3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z,
 * b.z));*/
        return sycl::float3(sycl::pow<double>(a.x(), b.x()),
                            sycl::pow<double>(a.y(), b.y()),
                            sycl::pow<double>(a.z(), b.z()));
}


////////////////////////////////////////////////////////////////////////////////
// expf
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float3 expf(float3 a)*/
inline sycl::float3 expf(sycl::float3 a)
{
/* DPCT_ORIG 	return make_float3(exp(a.x), exp(a.y), exp(a.z));*/
        return sycl::float3(sycl::exp(a.x()), sycl::exp(a.y()), sycl::exp(a.z()));
}


////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __device__ __host__ float smoothstep(float a, float b, float
 * x)*/
inline float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
/* DPCT_ORIG inline __device__ __host__ float2 smoothstep(float2 a, float2 b,
 * float2 x)*/
inline sycl::float2 smoothstep(sycl::float2 a, sycl::float2 b, sycl::float2 x)
{
/* DPCT_ORIG     float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);*/
    sycl::float2 y = clamp(dpct_operator_overloading::operator/(
                               (dpct_operator_overloading::operator-(x, a)),
                               (dpct_operator_overloading::operator-(b, a))),
                           0.0f, 1.0f);
/* DPCT_ORIG     return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));*/
    return (dpct_operator_overloading::operator*(
        dpct_operator_overloading::operator*(y, y),
        (dpct_operator_overloading::operator-(
            make_float2(3.0f),
            (dpct_operator_overloading::operator*(make_float2(2.0f), y))))));
}
/* DPCT_ORIG inline __device__ __host__ float3 smoothstep(float3 a, float3 b,
 * float3 x)*/
inline sycl::float3 smoothstep(sycl::float3 a, sycl::float3 b, sycl::float3 x)
{
/* DPCT_ORIG     float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);*/
    sycl::float3 y = clamp(dpct_operator_overloading::operator/(
                               (dpct_operator_overloading::operator-(x, a)),
                               (dpct_operator_overloading::operator-(b, a))),
                           0.0f, 1.0f);
/* DPCT_ORIG     return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));*/
    return (dpct_operator_overloading::operator*(
        dpct_operator_overloading::operator*(y, y),
        (dpct_operator_overloading::operator-(
            make_float3(3.0f),
            (dpct_operator_overloading::operator*(make_float3(2.0f), y))))));
}
/* DPCT_ORIG inline __device__ __host__ float4 smoothstep(float4 a, float4 b,
 * float4 x)*/
inline sycl::float4 smoothstep(sycl::float4 a, sycl::float4 b, sycl::float4 x)
{
/* DPCT_ORIG     float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);*/
    sycl::float4 y = clamp(dpct_operator_overloading::operator/(
                               (dpct_operator_overloading::operator-(x, a)),
                               (dpct_operator_overloading::operator-(b, a))),
                           0.0f, 1.0f);
/* DPCT_ORIG     return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));*/
    return (dpct_operator_overloading::operator*(
        dpct_operator_overloading::operator*(y, y),
        (dpct_operator_overloading::operator-(
            make_float4(3.0f),
            (dpct_operator_overloading::operator*(make_float4(2.0f), y))))));
}


////////////////////////////////////////////////////////////////////////////////
// comparisons
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ bool operator<(float3 &u, float3 v)*/
/*
DPCT1011:215: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator<(sycl::float3 &u, sycl::float3 v)

{
/* DPCT_ORIG 	return fabsf(u.x)<fabsf(v.x) && fabsf(u.y)<fabsf(v.y) &&
 * fabsf(u.z)<fabsf(v.z);*/
        return sycl::fabs(u.x()) < sycl::fabs(v.x()) &&
               sycl::fabs(u.y()) < sycl::fabs(v.y()) &&
               sycl::fabs(u.z()) < sycl::fabs(v.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ bool operator>(float3 &u, float3 v)*/
/*
DPCT1011:216: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator>(sycl::float3 &u, sycl::float3 v)

{

/* DPCT_ORIG 	return fabsf(u.x)>fabsf(v.x) && fabsf(u.y)>fabsf(v.y) &&
 * fabsf(u.z)>fabsf(v.z);*/
        return sycl::fabs(u.x()) > sycl::fabs(v.x()) &&
               sycl::fabs(u.y()) > sycl::fabs(v.y()) &&
               sycl::fabs(u.z()) > sycl::fabs(v.z());
}
} // namespace dpct_operator_overloading

/* DPCT_ORIG inline __host__ __device__ bool isBlack(float3 v)*/
inline bool isBlack(sycl::float3 v)

{
	return length(v) < 1.192092896e-07F;
}

/* DPCT_ORIG inline __host__ __device__ bool isNan(float3 v)*/
inline bool isNan(sycl::float3 v)

{
/* DPCT_ORIG 	return isnan(v.x) || isnan(v.y) || isnan(v.z);*/
        return sycl::isnan(v.x()) || sycl::isnan(v.y()) || sycl::isnan(v.z());
}

/* DPCT_ORIG inline __host__ __device__ bool isInf(float3 v)*/
inline bool isInf(sycl::float3 v)

{
/* DPCT_ORIG 	return isinf(v.x) || isinf(v.y) || isinf(v.z);*/
        return sycl::isinf(v.x()) || sycl::isinf(v.y()) || sycl::isinf(v.z());
}

/* DPCT_ORIG inline __host__ __device__ unsigned int TWHash(unsigned int s) {*/
inline unsigned int TWHash(unsigned int s) {
        s = (s ^ 61) ^ (s >> 16);
	s = s + (s << 3);
	s = s ^ (s >> 4);
	s = s * 0x27d4eb2d;
	s = s ^ (s >> 15);
	return s;
}

/* DPCT_ORIG inline __host__ __device__ float RadToDeg(float radians) { return
 * radians * 180.0f * M_1_PI; }*/
inline float RadToDeg(float radians) { return radians * 180.0f * M_1_PI; }
/* DPCT_ORIG inline __host__ __device__ float DegToRad(float degrees) { return
 * degrees * M_PI * M_1_180; }*/
inline float DegToRad(float degrees) { return degrees * M_PI * M_1_180; }

////////////////////////////////////////////////////////////////////////////////
// signs
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float signf(float a) {*/
inline float signf(float a) {

        return a < 0 ? -1 : 1;

}

/* DPCT_ORIG inline __host__ __device__ float2 signf(float2 a) {*/
inline sycl::float2 signf(sycl::float2 a) {

/* DPCT_ORIG 	return make_float2(a.x < 0 ? -1 : 1, a.y < 0 ? -1 :
     * 1);*/
        return sycl::float2(a.x() < 0 ? -1 : 1, a.y() < 0 ? -1 : 1);
}

/* DPCT_ORIG inline __host__ __device__ float3 signf(float3 a) {*/
inline sycl::float3 signf(sycl::float3 a) {

/* DPCT_ORIG 	return make_float3(a.x < 0 ? -1 : 1, a.y < 0 ? -1 : 1, a.z < 0 ?
 * -1 : 1) ;*/
        return sycl::float3(a.x() < 0 ? -1 : 1, a.y() < 0 ? -1 : 1,
                            a.z() < 0 ? -1 : 1);
}

////////////////////////////////////////////////////////////////////////////////
// Conversion
////////////////////////////////////////////////////////////////////////////////

/* DPCT_ORIG inline __host__ __device__ float3 quaternion_to_euler(double x,
 * double y, double z, double w) {*/
inline sycl::float3 quaternion_to_euler(double x, double y, double z,
                                        double w) {

        float heading, attitude, bank; // x, y, z rotations

	double sw = w*w;
	double sx = x*x;
	double sy = y*y;
	double sz = z*z;

	double unit = sx + sy + sz + sw;
	double test = x*y + z*w;

	if (test > 0.4999 * unit) { // singularity at north pole
/* DPCT_ORIG 		heading = 2.0f * atan2f(x, w);*/
                heading = 2.0f * sycl::atan2((float)x, (float)w);
                attitude = M_PI / 2.0f;
		bank = 0;
/* DPCT_ORIG 		return make_float3(attitude, heading, bank);*/
                return sycl::float3(attitude, heading, bank);
        }
	if (test < -0.4999*unit) { // singularity at south pole
                                   /* DPCT_ORIG 		heading = -2 * atan2f(x, w);*/
                heading = -2 * sycl::atan2((float)x, (float)w);
                attitude = M_PI / 2.0f;
		bank = 0;
/* DPCT_ORIG 		return make_float3(attitude, heading, bank);*/
                return sycl::float3(attitude, heading, bank);
        }

/* DPCT_ORIG 	heading = atan2(2 * y*w - 2 * x*z, sx - sy - sz + sw);*/
        heading = sycl::atan2(2 * y * w - 2 * x * z, sx - sy - sz + sw);
/* DPCT_ORIG 	attitude = asin(2 * test / unit);*/
        attitude = sycl::asin(2 * test / unit);
/* DPCT_ORIG 	bank = atan2(2 * x*w - 2 * y*z, -sx + sy - sz + sw);*/
        bank = sycl::atan2(2 * x * w - 2 * y * z, -sx + sy - sz + sw);

/* DPCT_ORIG 	return make_float3(attitude, heading, bank);*/
        return sycl::float3(attitude, heading, bank);
}

/* DPCT_ORIG inline __host__ __device__ float3 quaternion_to_euler(float4
 * quaternion) {*/
inline sycl::float3 quaternion_to_euler(sycl::float4 quaternion) {

/* DPCT_ORIG 	float x = quaternion.x;*/
        float x = quaternion.x();
/* DPCT_ORIG 	float y = quaternion.y;*/
        float y = quaternion.y();
/* DPCT_ORIG 	float z = quaternion.z;*/
        float z = quaternion.z();
/* DPCT_ORIG 	float w = quaternion.w;*/
        float w = quaternion.w();

        float heading, attitude, bank; // x, y, z rotations

	float sw = w*w;
	float sx = x*x;
	float sy = y*y;
	float sz = z*z;

	float unit = sx + sy + sz + sw;
	float test = x*y + z*w;

	if (test > 0.4999f * unit) { // singularity at north pole
/* DPCT_ORIG 		heading = 2.0f * atan2f(x, w);*/
                heading = 2.0f * sycl::atan2(x, w);
                attitude = M_PI / 2.0f;
		bank = 0;
/* DPCT_ORIG 		return make_float3(attitude, heading, bank);*/
                return sycl::float3(attitude, heading, bank);
        }
	if (test < -0.4999f * unit) { // singularity at south pole
/* DPCT_ORIG 		heading = -2.0f * atan2f(x, w);*/
                heading = -2.0f * sycl::atan2(x, w);
                attitude = M_PI / 2.0f;
		bank = .0f;
/* DPCT_ORIG 		return make_float3(attitude, heading, bank);*/
                return sycl::float3(attitude, heading, bank);
        }

/* DPCT_ORIG 	heading = atan2(2.0f * y*w - 2.0f * x*z, sx - sy - sz + sw);*/
        heading = sycl::atan2(2.0f * y * w - 2.0f * x * z, sx - sy - sz + sw);
/* DPCT_ORIG 	attitude = asin(2.0f * test / unit);*/
        attitude = sycl::asin(2.0f * test / unit);
/* DPCT_ORIG 	bank = atan2(2.0f * x*w - 2.0f * y*z, -sx + sy - sz +
     * sw);*/
        bank = sycl::atan2(2.0f * x * w - 2.0f * y * z, -sx + sy - sz + sw);

/* DPCT_ORIG 	return make_float3(attitude, heading, bank);*/
        return sycl::float3(attitude, heading, bank);
}


#endif
