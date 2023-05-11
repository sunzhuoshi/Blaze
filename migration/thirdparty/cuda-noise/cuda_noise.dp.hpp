// This code was taken as is from: https://github.com/covexp/cuda-noise

// cudaNoise
// Library of common 3D noise functions for CUDA kernels

#pragma once

/* DPCT_ORIG #include <cuda_runtime.h>*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace cudaNoise {

    // Basis functions
    typedef enum {
        BASIS_CHECKER,
        BASIS_DISCRETE,
        BASIS_LINEARVALUE,
        BASIS_FADEDVALUE,
        BASIS_CUBICVALUE,
        BASIS_PERLIN,
        BASIS_SIMPLEX,
        BASIS_WORLEY,
        BASIS_SPOTS
    } basisFunction;

    // Shaping functions
    typedef enum {
        SHAPE_STEP,
        SHAPE_LINEAR,
        SHAPE_QUADRATIC
    } profileShape;

    // Function blending operators
    typedef enum {
        OPERATOR_ADD,
        OPERATOR_AVG,
        OPERATOR_MUL,
        OPERATOR_MAX,
        OPERATOR_MIN
    } repeatOperator;

#define EPSILON 0.000000001f

    // Utility functions

    // Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
/* DPCT_ORIG     __device__ unsigned int hash(unsigned int seed)*/
    unsigned int hash(unsigned int seed)
    {
        seed = (seed + 0x7ed55d16) + (seed << 12);
        seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
        seed = (seed + 0x165667b1) + (seed << 5);
        seed = (seed + 0xd3a2646c) ^ (seed << 9);
        seed = (seed + 0xfd7046c5) + (seed << 3);
        seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

        return seed;
    }

    // Returns a random integer between [min, max]
/* DPCT_ORIG     __device__ int randomIntRange(int min, int max, int seed)*/
    int randomIntRange(int min, int max, int seed)
    {
        int base = hash(seed);
        base = base % (1 + max - min) + min;

        return base;
    }

    // Returns a random float between [0, 1]
/* DPCT_ORIG     __device__ float randomFloat(unsigned int seed)*/
    float randomFloat(unsigned int seed)
    {
        unsigned int noiseVal = hash(seed);

        return ((float)noiseVal / (float)0xffffffff);
    }

    // Clamps val between [min, max]
/* DPCT_ORIG     __device__ float clamp(float val, float min, float max)*/
    float clamp(float val, float min, float max)
    {
        if (val < 0.0f)
            return 0.0f;
        else if (val > 1.0f)
            return 1.0f;

        return val;
    }

    // Maps from the signed range [0, 1] to unsigned [-1, 1]
    // NOTE: no clamping
/* DPCT_ORIG     __device__ float mapToSigned(float input)*/
    float mapToSigned(float input)
    {
        return input * 2.0f - 1.0f;
    }

    // Maps from the unsigned range [-1, 1] to signed [0, 1]
    // NOTE: no clamping
/* DPCT_ORIG     __device__ float mapToUnsigned(float input)*/
    float mapToUnsigned(float input)
    {
        return input * 0.5f + 0.5f;
    }

    // Maps from the signed range [0, 1] to unsigned [-1, 1] with clamping
/* DPCT_ORIG     __device__ float clampToSigned(float input)*/
    float clampToSigned(float input)
    {
/* DPCT_ORIG         return __saturatef(input) * 2.0f - 1.0f;*/
        return sycl::clamp<float>(input, 0.0f, 1.0f) * 2.0f - 1.0f;
    }

    // Maps from the unsigned range [-1, 1] to signed [0, 1] with clamping
/* DPCT_ORIG     __device__ float clampToUnsigned(float input)*/
    float clampToUnsigned(float input)
    {
/* DPCT_ORIG         return __saturatef(input * 0.5f + 0.5f);*/
        return sycl::clamp<float>(input * 0.5f + 0.5f, 0.0f, 1.0f);
    }


    // Random float for a grid coordinate [-1, 1]
/* DPCT_ORIG     __device__ float randomGrid(int x, int y, int z, int seed =
 * 0)*/
    float randomGrid(int x, int y, int z, int seed = 0)
    {
        return mapToSigned(randomFloat((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824.0f + seed)));
    }

    // Random unsigned int for a grid coordinate [0, MAXUINT]
/* DPCT_ORIG     __device__ unsigned int randomIntGrid(float x, float y, float
 * z, float seed = 0.0f)*/
    unsigned int randomIntGrid(float x, float y, float z, float seed = 0.0f)
    {
        return hash((unsigned int)(x * 1723.0f + y * 93241.0f + z * 149812.0f + 3824 + seed));
    }

    // Random 3D vector as float3 from grid position
/* DPCT_ORIG     __device__ float3 vectorNoise(int x, int y, int z)*/
    sycl::float3 vectorNoise(int x, int y, int z)
    {
/* DPCT_ORIG         return make_float3(randomFloat(x * 8231.0f + y * 34612.0f +
   z * 11836.0f + 19283.0f) * 2.0f - 1.0f, randomFloat(x * 1171.0f + y * 9234.0f
   + z * 992903.0f + 1466.0f) * 2.0f - 1.0f, 0.0f);*/
        return sycl::float3(
            randomFloat(x * 8231.0f + y * 34612.0f + z * 11836.0f + 19283.0f) *
                    2.0f -
                1.0f,
            randomFloat(x * 1171.0f + y * 9234.0f + z * 992903.0f + 1466.0f) *
                    2.0f -
                1.0f,
            0.0f);
    }

    // Scale 3D vector by scalar value
/* DPCT_ORIG     __device__ float3 scaleVector(float3 v, float factor)*/
    sycl::float3 scaleVector(sycl::float3 v, float factor)
    {
/* DPCT_ORIG         return make_float3(v.x * factor, v.y * factor, v.z *
 * factor);*/
        return sycl::float3(v.x() * factor, v.y() * factor, v.z() * factor);
    }

    // Scale 3D vector by nonuniform parameters
/* DPCT_ORIG     __device__ float3 nonuniformScaleVector(float3 v, float xf,
 * float yf, float zf)*/
    sycl::float3 nonuniformScaleVector(sycl::float3 v, float xf, float yf, float zf)
    {
/* DPCT_ORIG         return make_float3(v.x * xf, v.y * yf, v.z * zf);*/
        return sycl::float3(v.x() * xf, v.y() * yf, v.z() * zf);
    }


    // Adds two 3D vectors
/* DPCT_ORIG     __device__ float3 addVectors(float3 v, float3 w)*/
    sycl::float3 addVectors(sycl::float3 v, sycl::float3 w)
    {
/* DPCT_ORIG         return make_float3(v.x + w.x, v.y + w.y, v.z + w.z);*/
        return sycl::float3(v.x() + w.x(), v.y() + w.y(), v.z() + w.z());
    }

    // Dot product between two vectors
/* DPCT_ORIG     __device__ float dotProduct(float3 u, float3 v)*/
    float dotProduct(sycl::float3 u, sycl::float3 v)
    {
/* DPCT_ORIG         return (u.x * v.x + u.y * v.y + u.z * v.z);*/
        return (u.x() * v.x() + u.y() * v.y() + u.z() * v.z());
    }

    // Device constants for noise

/* DPCT_ORIG     __device__ __constant__ float gradMap[16][3] = { { 1.0f, 1.0f,
   0.0f },{ -1.0f, 1.0f, 0.0f },{ 1.0f, -1.0f, 0.0f },{ -1.0f, -1.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f },{ -1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f, -1.0f },{ -1.0f,
   0.0f, -1.0f }, { 0.0f, 1.0f, 1.0f },{ 0.0f, -1.0f, 1.0f },{ 0.0f, 1.0f, -1.0f
   },{ 0.0f, -1.0f, -1.0f }};*/
    static dpct::constant_memory<float, 2> gradMap(sycl::range<2>(16, 3),
                                                   {{1.0f, 1.0f, 0.0f},
                                                    {-1.0f, 1.0f, 0.0f},
                                                    {1.0f, -1.0f, 0.0f},
                                                    {-1.0f, -1.0f, 0.0f},
                                                    {1.0f, 0.0f, 1.0f},
                                                    {-1.0f, 0.0f, 1.0f},
                                                    {1.0f, 0.0f, -1.0f},
                                                    {-1.0f, 0.0f, -1.0f},
                                                    {0.0f, 1.0f, 1.0f},
                                                    {0.0f, -1.0f, 1.0f},
                                                    {0.0f, 1.0f, -1.0f},
                                                    {0.0f, -1.0f, -1.0f}});

    // Helper functions for noise

    // Linearly interpolate between two float values
/* DPCT_ORIG     __device__  float lerp(float a, float b, float ratio)*/
    float lerp(float a, float b, float ratio)
    {
        return a * (1.0f - ratio) + b * ratio;
    }

    // 1D cubic interpolation with four points
/* DPCT_ORIG     __device__ float cubic(float p0, float p1, float p2, float p3,
 * float x)*/
    float cubic(float p0, float p1, float p2, float p3, float x)
    {
        return p1 + 0.5f * x * (p2 - p0 + x * (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3 + x * (3.0f * (p1 - p2) + p3 - p0)));
    }

    // Fast gradient function for gradient noise
/* DPCT_ORIG     __device__ float grad(int hash, float x, float y, float z)*/
    float grad(int hash, float x, float y, float z)
    {
        switch (hash & 0xF)
        {
        case 0x0: return x + y;
        case 0x1: return -x + y;
        case 0x2: return x - y;
        case 0x3: return -x - y;
        case 0x4: return x + z;
        case 0x5: return -x + z;
        case 0x6: return x - z;
        case 0x7: return -x - z;
        case 0x8: return y + z;
        case 0x9: return -y + z;
        case 0xA: return y - z;
        case 0xB: return -y - z;
        case 0xC: return y + x;
        case 0xD: return -y + z;
        case 0xE: return y - x;
        case 0xF: return -y - z;
        default: return 0; // never happens
        }
    }

    // Ken Perlin's fade function for Perlin noise
/* DPCT_ORIG     __device__ float fade(float t)*/
    float fade(float t)
    {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);     // 6t^5 - 15t^4 + 10t^3
    }

    // Dot product using a float[3] and float parameters
    // NOTE: could be cleaned up
/* DPCT_ORIG     __device__ float dot(float g[3], float x, float y, float z) {*/
    float dot(float g[3], float x, float y, float z) {
        return g[0] * x + g[1] * y + g[2] * z;
    }

    // Random value for simplex noise [0, 255]
/* DPCT_ORIG     __device__ unsigned char calcPerm(int p)*/
    unsigned char calcPerm(int p)
    {
        return (unsigned char)(hash(p));
    }

    // Random value for simplex noise [0, 11]
/* DPCT_ORIG     __device__ unsigned char calcPerm12(int p)*/
    unsigned char calcPerm12(int p)
    {
        return (unsigned char)(hash(p) % 12);
    }

    // Noise functions

    // Simplex noise adapted from Java code by Stefan Gustafson and Peter Eastman
/* DPCT_ORIG     __device__ float simplexNoise(float3 pos, float scale, int
 * seed)*/
    float simplexNoise(sycl::float3 pos, float scale, int seed,
                       dpct::accessor<float, dpct::constant, 2> gradMap)
    {
/* DPCT_ORIG         float xin = pos.x * scale;*/
        float xin = pos.x() * scale;
/* DPCT_ORIG         float yin = pos.y * scale;*/
        float yin = pos.y() * scale;
/* DPCT_ORIG         float zin = pos.z * scale;*/
        float zin = pos.z() * scale;

        // Skewing and unskewing factors for 3 dimensions
        float F3 = 1.0f / 3.0f;
        float G3 = 1.0f / 6.0f;

        float n0, n1, n2, n3; // Noise contributions from the four corners

                                // Skew the input space to determine which simplex cell we're in
        float s = (xin + yin + zin)*F3; // Very nice and simple skew factor for 3D
/* DPCT_ORIG         int i = floorf(xin + s);*/
        int i = sycl::floor(xin + s);
/* DPCT_ORIG         int j = floorf(yin + s);*/
        int j = sycl::floor(yin + s);
/* DPCT_ORIG         int k = floorf(zin + s);*/
        int k = sycl::floor(zin + s);
        float t = (i + j + k)*G3;
        float X0 = i - t; // Unskew the cell origin back to (x,y,z) space
        float Y0 = j - t;
        float Z0 = k - t;
        float x0 = xin - X0; // The x,y,z distances from the cell origin
        float y0 = yin - Y0;
        float z0 = zin - Z0;

        // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
        // Determine which simplex we are in.
        int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
        int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
        if (x0 >= y0) {
            if (y0 >= z0)
            {
                i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f;
            } // X Y Z order
            else if (x0 >= z0) { i1 = 1.0f; j1 = 0.0f; k1 = 0.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // X Z Y order
            else { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 1.0f; j2 = 0.0f; k2 = 1.0f; } // Z X Y order
        }
        else { // x0<y0
            if (y0 < z0) { i1 = 0.0f; j1 = 0.0f; k1 = 1.0f; i2 = 0.0f; j2 = 1; k2 = 1.0f; } // Z Y X order
            else if (x0 < z0) { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 0.0f; j2 = 1.0f; k2 = 1.0f; } // Y Z X order
            else { i1 = 0.0f; j1 = 1.0f; k1 = 0.0f; i2 = 1.0f; j2 = 1.0f; k2 = 0.0f; } // Y X Z order
        }

        // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
        // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
        // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
        // c = 1/6.
        float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2.0f*G3; // Offsets for third corner in (x,y,z) coords
        float y2 = y0 - j2 + 2.0f*G3;
        float z2 = z0 - k2 + 2.0f*G3;
        float x3 = x0 - 1.0f + 3.0f*G3; // Offsets for last corner in (x,y,z) coords
        float y3 = y0 - 1.0f + 3.0f*G3;
        float z3 = z0 - 1.0f + 3.0f*G3;

        int gi0 = calcPerm12(seed + (i * 607495) + (j * 359609) + (k * 654846));
        int gi1 = calcPerm12(seed + (i + i1) * 607495 + (j + j1) * 359609 + (k + k1) * 654846);
        int gi2 = calcPerm12(seed + (i + i2) * 607495 + (j + j2) * 359609 + (k + k2) * 654846);
        int gi3 = calcPerm12(seed + (i + 1) * 607495 + (j + 1) * 359609 + (k + 1) * 654846);

        // Calculate the contribution from the four corners
        float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 < 0.0f) n0 = 0.0f;
        else {
            t0 *= t0;
            n0 = t0 * t0 * dot(gradMap[gi0], x0, y0, z0);
        }
        float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 < 0.0f) n1 = 0.0f;
        else {
            t1 *= t1;
            n1 = t1 * t1 * dot(gradMap[gi1], x1, y1, z1);
        }
        float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 < 0.0f) n2 = 0.0f;
        else {
            t2 *= t2;
            n2 = t2 * t2 * dot(gradMap[gi2], x2, y2, z2);
        }
        float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 < 0.0f) n3 = 0.0f;
        else {
            t3 *= t3;
            n3 = t3 * t3 * dot(gradMap[gi3], x3, y3, z3);
        }

        // Add contributions from each corner to get the final noise value.
        // The result is scaled to stay just inside [-1,1]
        return 32.0f*(n0 + n1 + n2 + n3);
    }

    // Checker pattern
/* DPCT_ORIG     __device__ float checker(float3 pos, float scale, int seed)*/
    float checker(sycl::float3 pos, float scale, int seed)
    {
/* DPCT_ORIG         int ix = (int)(pos.x * scale);*/
        int ix = (int)(pos.x() * scale);
/* DPCT_ORIG         int iy = (int)(pos.y * scale);*/
        int iy = (int)(pos.y() * scale);
/* DPCT_ORIG         int iz = (int)(pos.z * scale);*/
        int iz = (int)(pos.z() * scale);

        if ((ix + iy + iz) % 2 == 0)
            return 1.0f;

        return -1.0f;
    }

    // Random spots
/* DPCT_ORIG     __device__ float spots(float3 pos, float scale, int seed, float
 * size, int minNum, int maxNum, float jitter, profileShape shape)*/
    float spots(sycl::float3 pos, float scale, int seed, float size, int minNum,
                int maxNum, float jitter, profileShape shape)
    {
        if (size < EPSILON)
            return 0.0f;

/* DPCT_ORIG         int ix = (int)(pos.x * scale);*/
        int ix = (int)(pos.x() * scale);
/* DPCT_ORIG         int iy = (int)(pos.y * scale);*/
        int iy = (int)(pos.y() * scale);
/* DPCT_ORIG         int iz = (int)(pos.z * scale);*/
        int iz = (int)(pos.z() * scale);

/* DPCT_ORIG         float u = pos.x - (float)ix;*/
        float u = pos.x() - (float)ix;
/* DPCT_ORIG         float v = pos.y - (float)iy;*/
        float v = pos.y() - (float)iy;
/* DPCT_ORIG         float w = pos.z - (float)iz;*/
        float w = pos.z() - (float)iz;

        float val = -1.0f;

        // We need to traverse the entire 3x3x3 neighborhood in case there are spots in neighbors near the edges of the cell
        for (int x = -1; x < 2; x++)
        {
            for (int y = -1; y < 2; y++)
            {
                for (int z = -1; z < 2; z++)
                {
                    int numSpots = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f + (iz + z) * 823452.0f + 3234874.0f);

                    for (int i = 0; i < numSpots; i++)
                    {
                        float distU = u - x - (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f + (iz + z) * 23874.0f * i + 27432.0f) * jitter - jitter / 2.0f);
                        float distV = v - y - (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f + (iz + z) * 144421.0f * i + 76671.0f) * jitter - jitter / 2.0f);
                        float distW = w - z - (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f + (iz + z) * 443241.0f * i + 199823.0f) * jitter - jitter / 2.0f);

                        float distanceSq = distU * distU + distV * distV + distW * distW;
                        float distanceAbs = 0.0f;

                        switch (shape)
                        {
                        case(SHAPE_STEP):
                            if (distanceSq < size)
/* DPCT_ORIG                                 val = fmaxf(val, 1.0f);*/
                                val = sycl::fmax(val, 1.0f);
                            else
/* DPCT_ORIG                                 val = fmaxf(val, -1.0f);*/
                                val = sycl::fmax(val, -1.0f);
                            break;
                        case(SHAPE_LINEAR):
/* DPCT_ORIG                             distanceAbs = fabsf(distU) +
 * fabsf(distV) + fabsf(distW);*/
                            distanceAbs = sycl::fabs(distU) +
                                          sycl::fabs(distV) + sycl::fabs(distW);
/* DPCT_ORIG                             val = fmaxf(val, 1.0f -
 * clamp(distanceAbs, 0.0f, size) / size);*/
                            val = sycl::fmax(
                                val,
                                1.0f - clamp(distanceAbs, 0.0f, size) / size);
                            break;
                        case(SHAPE_QUADRATIC):
/* DPCT_ORIG                             val = fmaxf(val, 1.0f -
 * clamp(distanceSq, 0.0f, size) / size);*/
                            val = sycl::fmax(
                                val,
                                1.0f - clamp(distanceSq, 0.0f, size) / size);
                            break;
                        }
                    }
                }
            }
        }

        return val;
    }

    // Worley cellular noise
/* DPCT_ORIG     __device__ float worleyNoise(float3 pos, float scale, int seed,
 * float size, int minNum, int maxNum, float jitter)*/
    float worleyNoise(sycl::float3 pos, float scale, int seed, float size,
                      int minNum, int maxNum, float jitter)
    {
        if (size < EPSILON)
            return 0.0f;

/* DPCT_ORIG         int ix = (int)(pos.x * scale);*/
        int ix = (int)(pos.x() * scale);
/* DPCT_ORIG         int iy = (int)(pos.y * scale);*/
        int iy = (int)(pos.y() * scale);
/* DPCT_ORIG         int iz = (int)(pos.z * scale);*/
        int iz = (int)(pos.z() * scale);

/* DPCT_ORIG         float u = pos.x - (float)ix;*/
        float u = pos.x() - (float)ix;
/* DPCT_ORIG         float v = pos.y - (float)iy;*/
        float v = pos.y() - (float)iy;
/* DPCT_ORIG         float w = pos.z - (float)iz;*/
        float w = pos.z() - (float)iz;

        float minDist = 1000000.0f;

        // Traverse the whole 3x3 neighborhood looking for the closest feature point
        for (int x = -1; x < 2; x++)
        {
            for (int y = -1; y < 2; y++)
            {
                for (int z = -1; z < 2; z++)
                {
                    int numPoints = randomIntRange(minNum, maxNum, seed + (ix + x) * 823746.0f + (iy + y) * 12306.0f + (iz + z) * 67262.0f);

                    for (int i = 0; i < numPoints; i++)
                    {
                        float distU = u - x - (randomFloat(seed + (ix + x) * 23784.0f + (iy + y) * 9183.0f + (iz + z) * 23874.0f * i + 27432.0f) * jitter - jitter / 2.0f);
                        float distV = v - y - (randomFloat(seed + (ix + x) * 12743.0f + (iy + y) * 45191.0f + (iz + z) * 144421.0f * i + 76671.0f) * jitter - jitter / 2.0f);
                        float distW = w - z - (randomFloat(seed + (ix + x) * 82734.0f + (iy + y) * 900213.0f + (iz + z) * 443241.0f * i + 199823.0f) * jitter - jitter / 2.0f);

                        float distanceSq = distU * distU + distV * distV + distW * distW;

                        if (distanceSq < minDist)
                            minDist = distanceSq;
                    }
                }
            }
        }

/* DPCT_ORIG         return __saturatef(minDist) * 2.0f - 1.0f;*/
        return sycl::clamp<float>(minDist, 0.0f, 1.0f) * 2.0f - 1.0f;
    }

    // Tricubic interpolation
/* DPCT_ORIG     __device__ float tricubic(int x, int y, int z, float u, float
 * v, float w)*/
    float tricubic(int x, int y, int z, float u, float v, float w)
    {
        // interpolate along x first
        float x00 = cubic(randomGrid(x - 1, y - 1, z - 1), randomGrid(x, y - 1, z - 1), randomGrid(x + 1, y - 1, z - 1), randomGrid(x + 2, y - 1, z - 1), u);
        float x01 = cubic(randomGrid(x - 1, y - 1, z), randomGrid(x, y - 1, z), randomGrid(x + 1, y - 1, z), randomGrid(x + 2, y - 1, z), u);
        float x02 = cubic(randomGrid(x - 1, y - 1, z + 1), randomGrid(x, y - 1, z + 1), randomGrid(x + 1, y - 1, z + 1), randomGrid(x + 2, y - 1, z + 1), u);
        float x03 = cubic(randomGrid(x - 1, y - 1, z + 2), randomGrid(x, y - 1, z + 2), randomGrid(x + 1, y - 1, z + 2), randomGrid(x + 2, y - 1, z + 2), u);

        float x10 = cubic(randomGrid(x - 1, y, z - 1), randomGrid(x, y, z - 1), randomGrid(x + 1, y, z - 1), randomGrid(x + 2, y, z - 1), u);
        float x11 = cubic(randomGrid(x - 1, y, z), randomGrid(x, y, z), randomGrid(x + 1, y, z), randomGrid(x + 2, y, z), u);
        float x12 = cubic(randomGrid(x - 1, y, z + 1), randomGrid(x, y, z + 1), randomGrid(x + 1, y, z + 1), randomGrid(x + 2, y, z + 1), u);
        float x13 = cubic(randomGrid(x - 1, y, z + 2), randomGrid(x, y, z + 2), randomGrid(x + 1, y, z + 2), randomGrid(x + 2, y, z + 2), u);

        float x20 = cubic(randomGrid(x - 1, y + 1, z - 1), randomGrid(x, y + 1, z - 1), randomGrid(x + 1, y + 1, z - 1), randomGrid(x + 2, y + 1, z - 1), u);
        float x21 = cubic(randomGrid(x - 1, y + 1, z), randomGrid(x, y + 1, z), randomGrid(x + 1, y + 1, z), randomGrid(x + 2, y + 1, z), u);
        float x22 = cubic(randomGrid(x - 1, y + 1, z + 1), randomGrid(x, y + 1, z + 1), randomGrid(x + 1, y + 1, z + 1), randomGrid(x + 2, y + 1, z + 1), u);
        float x23 = cubic(randomGrid(x - 1, y + 1, z + 2), randomGrid(x, y + 1, z + 2), randomGrid(x + 1, y + 1, z + 2), randomGrid(x + 2, y + 1, z + 2), u);

        float x30 = cubic(randomGrid(x - 1, y + 2, z - 1), randomGrid(x, y + 2, z - 1), randomGrid(x + 1, y + 2, z - 1), randomGrid(x + 2, y + 2, z - 1), u);
        float x31 = cubic(randomGrid(x - 1, y + 2, z), randomGrid(x, y + 2, z), randomGrid(x + 1, y + 2, z), randomGrid(x + 2, y + 2, z), u);
        float x32 = cubic(randomGrid(x - 1, y + 2, z + 1), randomGrid(x, y + 2, z + 1), randomGrid(x + 1, y + 2, z + 1), randomGrid(x + 2, y + 2, z + 1), u);
        float x33 = cubic(randomGrid(x - 1, y + 2, z + 2), randomGrid(x, y + 2, z + 2), randomGrid(x + 1, y + 2, z + 2), randomGrid(x + 2, y + 2, z + 2), u);

        // interpolate along y
        float y0 = cubic(x00, x10, x20, x30, v);
        float y1 = cubic(x01, x11, x21, x31, v);
        float y2 = cubic(x02, x12, x22, x32, v);
        float y3 = cubic(x03, x13, x23, x33, v);

        // interpolate along z
        return cubic(y0, y1, y2, y3, w);
    }

    // Discrete noise (nearest neighbor)
/* DPCT_ORIG     __device__ float discreteNoise(float3 pos, float scale, int
 * seed)*/
    float discreteNoise(sycl::float3 pos, float scale, int seed)
    {
/* DPCT_ORIG         int ix = (int)(pos.x * scale);*/
        int ix = (int)(pos.x() * scale);
/* DPCT_ORIG         int iy = (int)(pos.y * scale);*/
        int iy = (int)(pos.y() * scale);
/* DPCT_ORIG         int iz = (int)(pos.z * scale);*/
        int iz = (int)(pos.z() * scale);

        return randomGrid(ix, iy, iz, seed);
    }

    // Linear value noise
/* DPCT_ORIG     __device__ float linearValue(float3 pos, float scale, int
 * seed)*/
    float linearValue(sycl::float3 pos, float scale, int seed)
    {
        float fseed = (float)seed;

/* DPCT_ORIG         int ix = (int)pos.x;*/
        int ix = (int)pos.x();
/* DPCT_ORIG         int iy = (int)pos.y;*/
        int iy = (int)pos.y();
/* DPCT_ORIG         int iz = (int)pos.z;*/
        int iz = (int)pos.z();

/* DPCT_ORIG         float u = pos.x - ix;*/
        float u = pos.x() - ix;
/* DPCT_ORIG         float v = pos.y - iy;*/
        float v = pos.y() - iy;
/* DPCT_ORIG         float w = pos.z - iz;*/
        float w = pos.z() - iz;

        // Corner values
        float a000 = randomGrid(ix, iy, iz, fseed);
        float a100 = randomGrid(ix + 1, iy, iz, fseed);
        float a010 = randomGrid(ix, iy + 1, iz, fseed);
        float a110 = randomGrid(ix + 1, iy + 1, iz, fseed);
        float a001 = randomGrid(ix, iy, iz + 1, fseed);
        float a101 = randomGrid(ix + 1, iy, iz + 1, fseed);
        float a011 = randomGrid(ix, iy + 1, iz + 1, fseed);
        float a111 = randomGrid(ix + 1, iy + 1, iz + 1, fseed);

        // Linear interpolation
        float x00 = lerp(a000, a100, u);
        float x10 = lerp(a010, a110, u);
        float x01 = lerp(a001, a101, u);
        float x11 = lerp(a011, a111, u);

        float y0 = lerp(x00, x10, v);
        float y1 = lerp(x01, x11, v);

        return lerp(y0, y1, w);
    }

    // Linear value noise smoothed with Perlin's fade function
/* DPCT_ORIG     __device__ float fadedValue(float3 pos, float scale, int
 * seed)*/
    float fadedValue(sycl::float3 pos, float scale, int seed)
    {
        float fseed = (float)seed;

/* DPCT_ORIG         int ix = (int)(pos.x * scale);*/
        int ix = (int)(pos.x() * scale);
/* DPCT_ORIG         int iy = (int)(pos.y * scale);*/
        int iy = (int)(pos.y() * scale);
/* DPCT_ORIG         int iz = (int)(pos.z * scale);*/
        int iz = (int)(pos.z() * scale);

/* DPCT_ORIG         float u = fade(pos.x - ix);*/
        float u = fade(pos.x() - ix);
/* DPCT_ORIG         float v = fade(pos.y - iy);*/
        float v = fade(pos.y() - iy);
/* DPCT_ORIG         float w = fade(pos.z - iz);*/
        float w = fade(pos.z() - iz);

        // Corner values
        float a000 = randomGrid(ix, iy, iz, fseed);
        float a100 = randomGrid(ix + 1, iy, iz, fseed);
        float a010 = randomGrid(ix, iy + 1, iz, fseed);
        float a110 = randomGrid(ix + 1, iy + 1, iz, fseed);
        float a001 = randomGrid(ix, iy, iz + 1, fseed);
        float a101 = randomGrid(ix + 1, iy, iz + 1, fseed);
        float a011 = randomGrid(ix, iy + 1, iz + 1, fseed);
        float a111 = randomGrid(ix + 1, iy + 1, iz + 1, fseed);

        // Linear interpolation
        float x00 = lerp(a000, a100, u);
        float x10 = lerp(a010, a110, u);
        float x01 = lerp(a001, a101, u);
        float x11 = lerp(a011, a111, u);

        float y0 = lerp(x00, x10, v);
        float y1 = lerp(x01, x11, v);

        return lerp(y0, y1, w) / 2.0f * 1.0f;
    }

    // Tricubic interpolated value noise
/* DPCT_ORIG     __device__ float cubicValue(float3 pos, float scale, int
 * seed)*/
    float cubicValue(sycl::float3 pos, float scale, int seed)
    {
/* DPCT_ORIG         pos.x = pos.x * scale;*/
        pos.x() = pos.x() * scale;
/* DPCT_ORIG         pos.y = pos.y * scale;*/
        pos.y() = pos.y() * scale;
/* DPCT_ORIG         pos.z = pos.z * scale;*/
        pos.z() = pos.z() * scale;

/* DPCT_ORIG         int ix = (int)pos.x;*/
        int ix = (int)pos.x();
/* DPCT_ORIG         int iy = (int)pos.y;*/
        int iy = (int)pos.y();
/* DPCT_ORIG         int iz = (int)pos.z;*/
        int iz = (int)pos.z();

/* DPCT_ORIG         float u = pos.x - ix;*/
        float u = pos.x() - ix;
/* DPCT_ORIG         float v = pos.y - iy;*/
        float v = pos.y() - iy;
/* DPCT_ORIG         float w = pos.z - iz;*/
        float w = pos.z() - iz;

        return tricubic(ix, iy, iz, u, v, w);
    }

    // Perlin gradient noise
/* DPCT_ORIG     __device__ float perlinNoise(float3 pos, float scale, int
 * seed)*/
    float perlinNoise(sycl::float3 pos, float scale, int seed)
    {
        float fseed = (float)seed;

/* DPCT_ORIG         pos.x = pos.x * scale;*/
        pos.x() = pos.x() * scale;
/* DPCT_ORIG         pos.y = pos.y * scale;*/
        pos.y() = pos.y() * scale;
/* DPCT_ORIG         pos.z = pos.z * scale;*/
        pos.z() = pos.z() * scale;

        // zero corner integer position
/* DPCT_ORIG         float ix = floorf(pos.x);*/
        float ix = sycl::floor(pos.x());
/* DPCT_ORIG         float iy = floorf(pos.y);*/
        float iy = sycl::floor(pos.y());
/* DPCT_ORIG         float iz = floorf(pos.z);*/
        float iz = sycl::floor(pos.z());

        // current position within unit cube
/* DPCT_ORIG         pos.x -= ix;*/
        pos.x() -= ix;
/* DPCT_ORIG         pos.y -= iy;*/
        pos.y() -= iy;
/* DPCT_ORIG         pos.z -= iz;*/
        pos.z() -= iz;

        // adjust for fade
/* DPCT_ORIG         float u = fade(pos.x);*/
        float u = fade(pos.x());
/* DPCT_ORIG         float v = fade(pos.y);*/
        float v = fade(pos.y());
/* DPCT_ORIG         float w = fade(pos.z);*/
        float w = fade(pos.z());

        // influence values
/* DPCT_ORIG         float i000 = grad(randomIntGrid(ix, iy, iz, fseed), pos.x,
 * pos.y, pos.z);*/
        float i000 =
            grad(randomIntGrid(ix, iy, iz, fseed), pos.x(), pos.y(), pos.z());
/* DPCT_ORIG         float i100 = grad(randomIntGrid(ix + 1.0f, iy, iz, fseed),
 * pos.x - 1.0f, pos.y, pos.z);*/
        float i100 = grad(randomIntGrid(ix + 1.0f, iy, iz, fseed),
                          pos.x() - 1.0f, pos.y(), pos.z());
/* DPCT_ORIG         float i010 = grad(randomIntGrid(ix, iy + 1.0f, iz, fseed),
 * pos.x, pos.y - 1.0f, pos.z);*/
        float i010 = grad(randomIntGrid(ix, iy + 1.0f, iz, fseed), pos.x(),
                          pos.y() - 1.0f, pos.z());
/* DPCT_ORIG         float i110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz,
 * fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z);*/
        float i110 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz, fseed),
                          pos.x() - 1.0f, pos.y() - 1.0f, pos.z());
/* DPCT_ORIG         float i001 = grad(randomIntGrid(ix, iy, iz + 1.0f, fseed),
 * pos.x, pos.y, pos.z - 1.0f);*/
        float i001 = grad(randomIntGrid(ix, iy, iz + 1.0f, fseed), pos.x(),
                          pos.y(), pos.z() - 1.0f);
/* DPCT_ORIG         float i101 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f,
 * fseed), pos.x - 1.0f, pos.y, pos.z - 1.0f);*/
        float i101 = grad(randomIntGrid(ix + 1.0f, iy, iz + 1.0f, fseed),
                          pos.x() - 1.0f, pos.y(), pos.z() - 1.0f);
/* DPCT_ORIG         float i011 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f,
 * fseed), pos.x, pos.y - 1.0f, pos.z - 1.0f);*/
        float i011 = grad(randomIntGrid(ix, iy + 1.0f, iz + 1.0f, fseed),
                          pos.x(), pos.y() - 1.0f, pos.z() - 1.0f);
/* DPCT_ORIG         float i111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz
 * + 1.0f, fseed), pos.x - 1.0f, pos.y - 1.0f, pos.z - 1.0f);*/
        float i111 = grad(randomIntGrid(ix + 1.0f, iy + 1.0f, iz + 1.0f, fseed),
                          pos.x() - 1.0f, pos.y() - 1.0f, pos.z() - 1.0f);

        // interpolation
        float x00 = lerp(i000, i100, u);
        float x10 = lerp(i010, i110, u);
        float x01 = lerp(i001, i101, u);
        float x11 = lerp(i011, i111, u);

        float y0 = lerp(x00, x10, v);
        float y1 = lerp(x01, x11, v);

        float avg = lerp(y0, y1, w);

        return avg;
    }

// Derived noise functions

    // Fast function for fBm using perlin noise
/* DPCT_ORIG     __device__ float repeaterPerlin(float3 pos, float scale, int
 * seed, int n, float lacunarity, float decay)*/
    float repeaterPerlin(sycl::float3 pos, float scale, int seed, int n,
                         float lacunarity, float decay)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             acc += perlinNoise(make_float3(pos.x * scale, pos.y *
 * scale, pos.z * scale), 1.0f, (i + 38) * 27389482) * amp;*/
            acc += perlinNoise(sycl::float3(pos.x() * scale, pos.y() * scale,
                                            pos.z() * scale),
                               1.0f, (i + 38) * 27389482) *
                   amp;
            scale *= lacunarity;
            amp *= decay;
        }

        return acc;
    }

    // Fast function for fBm using perlin noise
/* DPCT_ORIG     __device__ float repeaterPerlinBounded(float3 pos, float scale,
 * int seed, int n, float lacunarity, float decay, float threshold)*/
    float repeaterPerlinBounded(sycl::float3 pos, float scale, int seed, int n,
                                float lacunarity, float decay, float threshold)
    {
        float acc = 1.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             acc *= 1.0f - __saturatef(0.5f + 0.5f *
 * perlinNoise(make_float3(pos.x * scale, pos.y * scale, pos.z * scale), 1.0f,
 * seed ^ ((i + 38) * 27389482))) * amp;*/
            acc *= 1.0f -
                   sycl::clamp<float>(
                       0.5f + 0.5f * perlinNoise(sycl::float3(pos.x() * scale,
                                                              pos.y() * scale,
                                                              pos.z() * scale),
                                                 1.0f,
                                                 seed ^ ((i + 38) * 27389482)),
                       0.0f, 1.0f) *
                       amp;

            if(acc < threshold)
            {
                return 0.0f;
            }

            scale *= lacunarity;
            amp *= decay;
        }

        return acc;
    }

    // Fast function for fBm using perlin absolute noise
    // Originally called "turbulence", this method takes the absolute value of each octave before adding
/* DPCT_ORIG     __device__ float repeaterPerlinAbs(float3 pos, float scale, int
 * seed, int n, float lacunarity, float decay)*/
    float repeaterPerlinAbs(sycl::float3 pos, float scale, int seed, int n,
                            float lacunarity, float decay)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG                         acc += fabsf(perlinNoise(make_float3(pos.x
 * * scale, pos.y * scale, pos.z * scale), 1.0f, seed)) * amp;*/
                        acc +=
                            sycl::fabs(perlinNoise(
                                sycl::float3(pos.x() * scale, pos.y() * scale,
                                             pos.z() * scale),
                                1.0f, seed)) *
                            amp;
            scale *= lacunarity;
            amp *= decay;
        }

        // Map the noise back to the standard expected range [-1, 1]
        return mapToSigned(acc);
    }

    // Fast function for fBm using simplex noise
/* DPCT_ORIG     __device__ float repeaterSimplex(float3 pos, float scale, int
 * seed, int n, float lacunarity, float decay)*/
    float repeaterSimplex(sycl::float3 pos, float scale, int seed, int n,
                          float lacunarity, float decay,
                          dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             acc += simplexNoise(make_float3(pos.x, pos.y, pos.z),
 * scale, seed) * amp * 0.35f;*/
            acc += simplexNoise(sycl::float3(pos.x(), pos.y(), pos.z()), scale,
                                seed, gradMap) *
                   amp * 0.35f;
            scale *= lacunarity;
            amp *= decay;
            seed = seed ^ ((i + 672381) * 200394);
        }

        return acc;
    }

    // Fast function for fBm using simplex absolute noise
/* DPCT_ORIG     __device__ float repeaterSimplexAbs(float3 pos, float scale,
 * int seed, int n, float lacunarity, float decay)*/
    float repeaterSimplexAbs(sycl::float3 pos, float scale, int seed, int n,
                             float lacunarity, float decay,
                             dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             acc += fabsf(simplexNoise(make_float3(pos.x, pos.y,
 * pos.z), scale, seed)) * amp * 0.35f;*/
            acc +=
                sycl::fabs(simplexNoise(sycl::float3(pos.x(), pos.y(), pos.z()),
                                        scale, seed, gradMap)) *
                amp * 0.35f;
            scale *= lacunarity;
            amp *= decay;
            seed = seed ^ ((i + 198273) * 928374);
        }

        return mapToSigned(acc);
    }

    // Bounded simplex repeater
/* DPCT_ORIG     __device__ float repeaterSimplexBounded(float3 pos, float
 * scale, int seed, int n, float lacunarity, float decay, float threshold)*/
    float
    repeaterSimplexBounded(sycl::float3 pos, float scale, int seed, int n,
                           float lacunarity, float decay, float threshold,
                           dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        float acc = 1.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             float val = __saturatef((simplexNoise(make_float3(pos.x
 * * scale + 32240.7922f, pos.y * scale + 835622.882f, pos.z * scale +
 * 824.371968f), 1.0f, seed) * 0.3f + 0.5f)) * amp;*/
            float val =
                sycl::clamp<float>(
                    (simplexNoise(sycl::float3(pos.x() * scale + 32240.7922f,
                                               pos.y() * scale + 835622.882f,
                                               pos.z() * scale + 824.371968f),
                                  1.0f, seed, gradMap) *
                         0.3f +
                     0.5f),
                    0.0f, 1.0f) *
                amp;
            acc -= val;

            if(acc < threshold)
            {
                return 0.0f;
            }

            scale *= lacunarity;
            amp *= decay;
        }

        return acc;
    }

    // Generic fBm repeater
    // NOTE: about 10% slower than the dedicated repeater functions
/* DPCT_ORIG     __device__ float repeater(float3 pos, float scale, int seed,
 * int n, float lacunarity, float decay, basisFunction basis)*/
    float repeater(sycl::float3 pos, float scale, int seed, int n,
                   float lacunarity, float decay, basisFunction basis,
                   dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        for (int i = 0; i < n; i++)
        {
            switch (basis)
            {
            case(BASIS_CHECKER):
/* DPCT_ORIG                 acc += checker(make_float3(pos.x * scale +
 * 53872.1923f, pos.y * scale + 58334.4081f, pos.z * scale + 9358.34667f), 1.0f,
 * seed) * amp;*/
                acc += checker(sycl::float3(pos.x() * scale + 53872.1923f,
                                            pos.y() * scale + 58334.4081f,
                                            pos.z() * scale + 9358.34667f),
                               1.0f, seed) *
                       amp;
                break;
            case(BASIS_DISCRETE):
/* DPCT_ORIG                 acc += discreteNoise(make_float3(pos.x * scale +
 * 7852.53114f, pos.y * scale + 319739.059f, pos.z * scale + 451336.504f), 1.0f,
 * seed) * amp;*/
                acc +=
                    discreteNoise(sycl::float3(pos.x() * scale + 7852.53114f,
                                               pos.y() * scale + 319739.059f,
                                               pos.z() * scale + 451336.504f),
                                  1.0f, seed) *
                    amp;
                break;
            case(BASIS_LINEARVALUE):
/* DPCT_ORIG                 acc += linearValue(make_float3(pos.x * scale +
 * 940.748139f, pos.y * scale + 10196.4500f, pos.z * scale + 25650.9789f), 1.0f,
 * seed) * amp;*/
                acc += linearValue(sycl::float3(pos.x() * scale + 940.748139f,
                                                pos.y() * scale + 10196.4500f,
                                                pos.z() * scale + 25650.9789f),
                                   1.0f, seed) *
                       amp;
                break;
            case(BASIS_FADEDVALUE):
/* DPCT_ORIG                 acc += fadedValue(make_float3(pos.x * scale +
 * 7683.26428f, pos.y * scale + 2417.78195f, pos.z * scale + 93889.4897f), 1.0f,
 * seed) * amp;*/
                acc += fadedValue(sycl::float3(pos.x() * scale + 7683.26428f,
                                               pos.y() * scale + 2417.78195f,
                                               pos.z() * scale + 93889.4897f),
                                  1.0f, seed) *
                       amp;
                break;
            case(BASIS_CUBICVALUE):
/* DPCT_ORIG                 acc += cubicValue(make_float3(pos.x * scale +
 * 6546.80178f, pos.y * scale + 14459.4682f, pos.z * scale + 11616.5811f), 1.0f,
 * seed) * amp;*/
                acc += cubicValue(sycl::float3(pos.x() * scale + 6546.80178f,
                                               pos.y() * scale + 14459.4682f,
                                               pos.z() * scale + 11616.5811f),
                                  1.0f, seed) *
                       amp;
                break;
            case(BASIS_PERLIN):
/* DPCT_ORIG                 acc += perlinNoise(make_float3(pos.x * scale +
 * 1764.66931f, pos.y * scale + 2593.55017f, pos.z * scale + 4813.24412f), 1.0f,
 * seed) * amp;*/
                acc += perlinNoise(sycl::float3(pos.x() * scale + 1764.66931f,
                                                pos.y() * scale + 2593.55017f,
                                                pos.z() * scale + 4813.24412f),
                                   1.0f, seed) *
                       amp;
                break;
            case(BASIS_SIMPLEX):
/* DPCT_ORIG                 acc += simplexNoise(make_float3(pos.x * scale +
 * 7442.93020f, pos.y * scale + 8341.06698f, pos.z * scale + 66848.7870f), 1.0f,
 * seed) * amp;*/
                acc += simplexNoise(sycl::float3(pos.x() * scale + 7442.93020f,
                                                 pos.y() * scale + 8341.06698f,
                                                 pos.z() * scale + 66848.7870f),
                                    1.0f, seed, gradMap) *
                       amp;
                break;
            case(BASIS_WORLEY):
/* DPCT_ORIG                 acc += worleyNoise(make_float3(pos.x * scale +
 * 7619.01285f, pos.y * scale + 57209.0681f, pos.z * scale + 1167.91397f), 1.0f,
 * seed, 0.1f, 4, 4, 1.0f) * amp;*/
                acc += worleyNoise(sycl::float3(pos.x() * scale + 7619.01285f,
                                                pos.y() * scale + 57209.0681f,
                                                pos.z() * scale + 1167.91397f),
                                   1.0f, seed, 0.1f, 4, 4, 1.0f) *
                       amp;
                break;
            case(BASIS_SPOTS):
/* DPCT_ORIG                 acc += spots(make_float3(pos.x * scale +
 * 33836.4116f, pos.y * scale + 2242.51045f, pos.z * scale + 6720.07486f), 1.0f,
 * seed, 0.1f, 0, 4, 1.0f, SHAPE_LINEAR) * amp;*/
                acc += spots(sycl::float3(pos.x() * scale + 33836.4116f,
                                          pos.y() * scale + 2242.51045f,
                                          pos.z() * scale + 6720.07486f),
                             1.0f, seed, 0.1f, 0, 4, 1.0f, SHAPE_LINEAR) *
                       amp;
                break;
            }

            scale *= lacunarity;
            amp *= decay;
        }

        return acc;
    }

    // Fractal Simplex noise
    // Unlike the repeater function, which calculates a fixed number of noise octaves, the fractal function continues until
    // the feature size is smaller than one pixel
/* DPCT_ORIG     __device__ float fractalSimplex(float3 pos, float scale, int
 * seed, float du, int n, float lacunarity, float decay)*/
    float fractalSimplex(sycl::float3 pos, float scale, int seed, float du,
                         int n, float lacunarity, float decay,
                         dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        float acc = 0.0f;
        float amp = 1.0f;

        float rdu = 1.0f / du;

        for (int i = 0; i < n; i++)
        {
/* DPCT_ORIG             acc += simplexNoise(make_float3(pos.x * scale +
 * 617.437379f, pos.y * scale + 196410.219f, pos.z * scale + 321280.627f), 1.0f,
 * seed * (i + 1)) * amp;*/
            acc += simplexNoise(sycl::float3(pos.x() * scale + 617.437379f,
                                             pos.y() * scale + 196410.219f,
                                             pos.z() * scale + 321280.627f),
                                1.0f, seed * (i + 1), gradMap) *
                   amp;
            scale *= lacunarity;
            amp *= decay;

            if (scale > rdu)
                break;
        }

        return acc;
    }

    // Generic turbulence function
    // Uses a first pass of noise to offset the input vectors for the second pass
/* DPCT_ORIG     __device__ float turbulence(float3 pos, float scaleIn, float
 * scaleOut, int seed, float strength, basisFunction inFunc, basisFunction
 * outFunc)*/
    float turbulence(sycl::float3 pos, float scaleIn, float scaleOut, int seed,
                     float strength, basisFunction inFunc,
                     basisFunction outFunc,
                     dpct::accessor<float, dpct::constant, 2> gradMap)
    {
        switch (inFunc)
        {
        case(BASIS_CHECKER):
/* DPCT_ORIG             pos.x += checker(pos, scaleIn, seed ^ 0x34ff8885) *
 * strength;*/
            pos.x() += checker(pos, scaleIn, seed ^ 0x34ff8885) * strength;
/* DPCT_ORIG             pos.y += checker(pos, scaleIn, seed ^ 0x2d03cba3) *
 * strength;*/
            pos.y() += checker(pos, scaleIn, seed ^ 0x2d03cba3) * strength;
/* DPCT_ORIG             pos.z += checker(pos, scaleIn, seed ^ 0x5a76fb1b) *
 * strength;*/
            pos.z() += checker(pos, scaleIn, seed ^ 0x5a76fb1b) * strength;
            break;
        case(BASIS_LINEARVALUE):
/* DPCT_ORIG             pos.x += linearValue(pos, scaleIn, seed ^ 0x5527fdb8) *
 * strength;*/
            pos.x() += linearValue(pos, scaleIn, seed ^ 0x5527fdb8) * strength;
/* DPCT_ORIG             pos.y += linearValue(pos, scaleIn, seed ^ 0x42af1a2e) *
 * strength;*/
            pos.y() += linearValue(pos, scaleIn, seed ^ 0x42af1a2e) * strength;
/* DPCT_ORIG             pos.z += linearValue(pos, scaleIn, seed ^ 0x1482ee8c) *
 * strength;*/
            pos.z() += linearValue(pos, scaleIn, seed ^ 0x1482ee8c) * strength;
            break;
        case(BASIS_FADEDVALUE):
/* DPCT_ORIG             pos.x += fadedValue(pos, scaleIn, seed ^ 0x295590fc) *
 * strength;*/
            pos.x() += fadedValue(pos, scaleIn, seed ^ 0x295590fc) * strength;
/* DPCT_ORIG             pos.y += fadedValue(pos, scaleIn, seed ^ 0x30731854) *
 * strength;*/
            pos.y() += fadedValue(pos, scaleIn, seed ^ 0x30731854) * strength;
/* DPCT_ORIG             pos.z += fadedValue(pos, scaleIn, seed ^ 0x73d2ca4c) *
 * strength;*/
            pos.z() += fadedValue(pos, scaleIn, seed ^ 0x73d2ca4c) * strength;
            break;
        case(BASIS_CUBICVALUE):
/* DPCT_ORIG             pos.x += cubicValue(pos, scaleIn, seed ^ 0x663a1f09) *
 * strength;*/
            pos.x() += cubicValue(pos, scaleIn, seed ^ 0x663a1f09) * strength;
/* DPCT_ORIG             pos.y += cubicValue(pos, scaleIn, seed ^ 0x429bf56b) *
 * strength;*/
            pos.y() += cubicValue(pos, scaleIn, seed ^ 0x429bf56b) * strength;
/* DPCT_ORIG             pos.z += cubicValue(pos, scaleIn, seed ^ 0x37fa6fe9) *
 * strength;*/
            pos.z() += cubicValue(pos, scaleIn, seed ^ 0x37fa6fe9) * strength;
            break;
        case(BASIS_PERLIN):
/* DPCT_ORIG             pos.x += perlinNoise(pos, scaleIn, seed ^ 0x74827384) *
 * strength;*/
            pos.x() += perlinNoise(pos, scaleIn, seed ^ 0x74827384) * strength;
/* DPCT_ORIG             pos.y += perlinNoise(pos, scaleIn, seed ^ 0x10938478) *
 * strength;*/
            pos.y() += perlinNoise(pos, scaleIn, seed ^ 0x10938478) * strength;
/* DPCT_ORIG             pos.z += perlinNoise(pos, scaleIn, seed ^ 0x62723883) *
 * strength;*/
            pos.z() += perlinNoise(pos, scaleIn, seed ^ 0x62723883) * strength;
            break;
        case(BASIS_SIMPLEX):
/* DPCT_ORIG             pos.x += simplexNoise(pos, scaleIn, seed ^ 0x47829472)
 * * strength;*/
            pos.x() += simplexNoise(pos, scaleIn, seed ^ 0x47829472, gradMap) *
                       strength;
/* DPCT_ORIG             pos.y += simplexNoise(pos, scaleIn, seed ^ 0x58273829)
 * * strength;*/
            pos.y() += simplexNoise(pos, scaleIn, seed ^ 0x58273829, gradMap) *
                       strength;
/* DPCT_ORIG             pos.z += simplexNoise(pos, scaleIn, seed ^ 0x10294647)
 * * strength;*/
            pos.z() += simplexNoise(pos, scaleIn, seed ^ 0x10294647, gradMap) *
                       strength;
            break;
        case(BASIS_WORLEY):
/* DPCT_ORIG             pos.x += worleyNoise(pos, scaleIn, seed ^
 * 0x1d96f515, 1.0f, 4, 4, 1.0f) * strength;*/
            pos.x() +=
                worleyNoise(pos, scaleIn, seed ^ 0x1d96f515, 1.0f, 4, 4, 1.0f) *
                strength;
/* DPCT_ORIG             pos.y += worleyNoise(pos, scaleIn, seed ^
 * 0x4df308f0, 1.0f, 4, 4, 1.0f) * strength;*/
            pos.y() +=
                worleyNoise(pos, scaleIn, seed ^ 0x4df308f0, 1.0f, 4, 4, 1.0f) *
                strength;
/* DPCT_ORIG             pos.z += worleyNoise(pos, scaleIn, seed ^
 * 0x2b79442a, 1.0f, 4, 4, 1.0f) * strength;*/
            pos.z() +=
                worleyNoise(pos, scaleIn, seed ^ 0x2b79442a, 1.0f, 4, 4, 1.0f) *
                strength;
            break;
        }

        switch (outFunc)
        {
        case(BASIS_CHECKER):
            return checker(pos, scaleOut, seed);
        case(BASIS_LINEARVALUE):
            return linearValue(pos, scaleOut, seed);
        case(BASIS_FADEDVALUE):
            return fadedValue(pos, scaleOut, seed);
        case(BASIS_CUBICVALUE):
            return cubicValue(pos, scaleOut, seed);
        case(BASIS_PERLIN):
            return perlinNoise(pos, scaleOut, seed);
        case(BASIS_SIMPLEX):
            return simplexNoise(pos, scaleIn, seed, gradMap);
        case(BASIS_WORLEY):
            return worleyNoise(pos, scaleIn, seed, 1.0f, 4, 4, 1.0f);
        }

        return 0.0f;
    }

    // Turbulence using repeaters for the first and second pass
/* DPCT_ORIG     __device__ float repeaterTurbulence(float3 pos, float scaleIn,
 * float scaleOut, int seed, float strength, int n, basisFunction basisIn,
 * basisFunction basisOut)*/
    float repeaterTurbulence(sycl::float3 pos, float scaleIn, float scaleOut,
                             int seed, float strength, int n,
                             basisFunction basisIn, basisFunction basisOut,
                             dpct::accessor<float, dpct::constant, 2> gradMap)
    {
/* DPCT_ORIG         pos.x += (repeater(make_float3(pos.x, pos.y, pos.z),
 * scaleIn, seed ^ 0x41728394, n, 2.0f, 0.5f, basisIn)) * strength;*/
        pos.x() +=
            (repeater(sycl::float3(pos.x(), pos.y(), pos.z()), scaleIn,
                      seed ^ 0x41728394, n, 2.0f, 0.5f, basisIn, gradMap)) *
            strength;
/* DPCT_ORIG         pos.y += (repeater(make_float3(pos.x, pos.y, pos.z),
 * scaleIn, seed ^ 0x72837263, n, 2.0f, 0.5f, basisIn)) * strength;*/
        pos.y() +=
            (repeater(sycl::float3(pos.x(), pos.y(), pos.z()), scaleIn,
                      seed ^ 0x72837263, n, 2.0f, 0.5f, basisIn, gradMap)) *
            strength;
/* DPCT_ORIG         pos.z += (repeater(make_float3(pos.x, pos.y, pos.z),
 * scaleIn, seed ^ 0x26837363, n, 2.0f, 0.5f, basisIn)) * strength;*/
        pos.z() +=
            (repeater(sycl::float3(pos.x(), pos.y(), pos.z()), scaleIn,
                      seed ^ 0x26837363, n, 2.0f, 0.5f, basisIn, gradMap)) *
            strength;

        return repeater(pos, scaleOut, seed ^ 0x3f821dab, n, 2.0f, 0.5f,
                        basisOut, gradMap);
    }

} // namespace
