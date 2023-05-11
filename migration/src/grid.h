/*
Copyright (c) 2021 Alexandre Sirois-Vigneux

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.
*/


#ifndef GRID_H
#define GRID_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

/* DPCT_ORIG #include "cuda_runtime.h"*/
/* DPCT_ORIG #include "device_launch_parameters.h"*/
#include "utils/helper_math.h"

#include "SceneSettings.h"


enum InterpolationType { point, linear, cubic };


class Grid {

private:
    float *mFrontBuffer;
    float *mBackBuffer;

    float mDx;

    int mByteSize;

    float mOx;
    float mOy;
    float mOz;

    bool mFlipValue;

/* DPCT_ORIG     __device__ inline float xObjToGrid(float x) {*/
    inline float xObjToGrid(float x) {
        return x/mDx - mOx;
    }

/* DPCT_ORIG     __device__ inline float yObjToGrid(float y) {*/
    inline float yObjToGrid(float y) {
        return y/mDx - mOy;
    }

/* DPCT_ORIG     __device__ inline float zObjToGrid(float z) {*/
    inline float zObjToGrid(float z) {
        return z/mDx - mOz;
    }

/* DPCT_ORIG     __device__ inline float xWorldToObj(float x) {*/
    inline float xWorldToObj(float x) {
/* DPCT_ORIG         return x-mDomMin.x;*/
        return x - mDomMin.x();
    }

/* DPCT_ORIG     __device__ inline float yWorldToObj(float y) {*/
    inline float yWorldToObj(float y) {
/* DPCT_ORIG         return mDomSize.y - (y-mDomMin.y);*/
        return mDomSize.y() - (y - mDomMin.y());
    }

/* DPCT_ORIG     __device__ inline float zWorldToObj(float z) {*/
    inline float zWorldToObj(float z) {
/* DPCT_ORIG         return z-mDomMin.z;*/
        return z - mDomMin.z();
    }

/* DPCT_ORIG     __device__ inline float3 worldToObj(float wPosx,float
 * wPosy,float wPosz) {*/
    inline sycl::float3 worldToObj(float wPosx, float wPosy, float wPosz) {
/* DPCT_ORIG         return make_float3(wPosx-mDomMin.x,
                           mDomSize.y - (wPosy-mDomMin.y),
                           wPosz-mDomMin.z);*/
        return sycl::float3(wPosx - mDomMin.x(),
                            mDomSize.y() - (wPosy - mDomMin.y()),
                            wPosz - mDomMin.z());
    }

/* DPCT_ORIG     __device__ inline float3 worldToObj(float3 wPos) {*/
    inline sycl::float3 worldToObj(sycl::float3 wPos) {
/* DPCT_ORIG         return make_float3(wPos.x-mDomMin.x,
                           mDomSize.y - (wPos.y-mDomMin.y),
                           wPos.z-mDomMin.z);*/
        return sycl::float3(wPos.x() - mDomMin.x(),
                            mDomSize.y() - (wPos.y() - mDomMin.y()),
                            wPos.z() - mDomMin.z());
    }

    // this is the optimised version running at 120% the original speed
    // see readable version commented bellow
/* DPCT_ORIG     __device__ float trilinear_interpolate(float3 oPos){*/
    float trilinear_interpolate(sycl::float3 oPos) {

/* DPCT_ORIG         const float gX = fmaxf( fminf(xObjToGrid(oPos.x),
 * mWidth-1.001f), 0.0f);*/
        const float gX =
            sycl::fmax(sycl::fmin(xObjToGrid(oPos.x()), mWidth - 1.001f), 0.0f);
/* DPCT_ORIG         const float gY = fmaxf( fminf(yObjToGrid(oPos.y),
 * mHeight-1.001f), 0.0f);*/
        const float gY = sycl::fmax(
            sycl::fmin(yObjToGrid(oPos.y()), mHeight - 1.001f), 0.0f);
/* DPCT_ORIG         const float gZ = fmaxf( fminf(zObjToGrid(oPos.z),
 * mDepth-1.001f), 0.0f);*/
        const float gZ =
            sycl::fmax(sycl::fmin(zObjToGrid(oPos.z()), mDepth - 1.001f), 0.0f);

        return (1.0f-(gZ-(int)gZ))*( (1.0f-(gY-(int)gY))*(at((int)gX, (int)gY, (int)gZ)*(1.0f-(gX-(int)gX))
                                                        + at(((int)gX+1), (int)gY, (int)gZ)*(gX-(int)gX))
                                          + (gY-(int)gY)*(at((int)gX, ((int)gY+1), (int)gZ)*(1.0f-(gX-(int)gX))
                                                        + at(((int)gX+1), ((int)gY+1), (int)gZ)*(gX-(int)gX)) )
                    + (gZ-(int)gZ)*( (1.0f-(gY-(int)gY))*(at((int)gX, (int)gY, ((int)gZ+1))*(1.0f-(gX-(int)gX))
                                                        + at(((int)gX+1), (int)gY, ((int)gZ+1))*(gX-(int)gX))
                                          + (gY-(int)gY)*(at((int)gX, ((int)gY+1), ((int)gZ+1))*(1.0f-(gX-(int)gX))
                                                        + at(((int)gX+1), ((int)gY+1), ((int)gZ+1))*(gX-(int)gX)) );
    }

//    __device__ float trilinear_interpolate(float3 oPos){
//
//        float gX = fmax( fmin(xObjToGrid(oPos.x), mWidth-1.001f), 0.0f);
//        float gY = fmax( fmin(yObjToGrid(oPos.y), mHeight-1.001f), 0.0f);
//        float gZ = fmax( fmin(zObjToGrid(oPos.z), mDepth-1.001f), 0.0f);
//
//        int i0 = (int) gX;
//        int i1 = i0 + 1;
//        int j0 = (int) gY;
//        int j1 = j0 + 1;
//        int k0 = (int) gZ;
//        int k1 = k0 + 1;
//
//        assert( !(i0 < 0 || i0 >= mWidth  || i1 < 0 || i1 >= mWidth ||
//                  j0 < 0 || j0 >= mHeight || j1 < 0 || j1 >= mHeight ||
//                  k0 < 0 || k0 >= mDepth  || k1 < 0 || k1 >= mDepth) );
//
//        float x0 = gX - i0;
//        float x1 = 1.0f - x0;
//        float y0 = gY - j0;
//        float y1 = 1.0f  - y0;
//        float z0 = gZ - k0;
//        float z1 = 1.0f  - z0;
//
//        float p000 = at(i0, j0, k0);
//        float p100 = at(i1, j0, k0);
//        float p010 = at(i0, j1, k0);
//        float p110 = at(i1, j1, k0);
//        float p001 = at(i0, j0, k1);
//        float p011 = at(i0, j1, k1);
//        float p101 = at(i1, j0, k1);
//        float p111 = at(i1, j1, k1);
//
//        return z1*( y1*(p000*x1 + p100*x0) + y0*(p010*x1 + p110*x0) )
//             + z0*( y1*(p001*x1 + p101*x0) + y0*(p011*x1 + p111*x0) );
//    }

    // cubic interpolation
/* DPCT_ORIG     inline __device__ float cerp(float q0, float q1, float q2,
 * float q3, float s) {*/
    inline float cerp(float q0, float q1, float q2, float q3, float s) {

            return (-1.0f/3.0f*s + 0.5f*s*s - 1.0f/6.0f*s*s*s) * q0
                 + (1.0f - s*s + 0.5f*(s*s*s - s)) * q1
                 + (s + 0.5f*(s*s - s*s*s)) * q2
                 + (1.0f/6.0f*(s*s*s - s)) * q3;
    }

    // cubic interpolation that clamps the output to positive values
/* DPCT_ORIG     inline __device__ float cerpc(float q0, float q1, float q2,
 * float q3, float s) {*/
    inline float cerpc(float q0, float q1, float q2, float q3, float s) {

/* DPCT_ORIG             return fmaxf((-1.0f/3.0f*s + 0.5f*s*s
   - 1.0f/6.0f*s*s*s) * q0
                       + (1.0f - s*s + 0.5f*(s*s*s - s)) * q1
                       + (s + 0.5f*(s*s - s*s*s)) * q2
                       + (1.0f/6.0f*(s*s*s - s)) * q3, 0.0f);*/
            return sycl::fmax(
                (-1.0f / 3.0f * s + 0.5f * s * s - 1.0f / 6.0f * s * s * s) *
                        q0 +
                    (1.0f - s * s + 0.5f * (s * s * s - s)) * q1 +
                    (s + 0.5f * (s * s - s * s * s)) * q2 +
                    (1.0f / 6.0f * (s * s * s - s)) * q3,
                0.0f);
    }

/* DPCT_ORIG     __device__ float tricubic_interpolate (float3 oPos, bool clamp)
 * {*/
    float tricubic_interpolate(sycl::float3 oPos, bool clamp) {

/* DPCT_ORIG         const float gX = fmaxf( fminf(xObjToGrid(oPos.x),
 * mWidth-1.001f), 0.0f);*/
        const float gX =
            sycl::fmax(sycl::fmin(xObjToGrid(oPos.x()), mWidth - 1.001f), 0.0f);
/* DPCT_ORIG         const float gY = fmaxf( fminf(yObjToGrid(oPos.y),
 * mHeight-1.001f), 0.0f);*/
        const float gY = sycl::fmax(
            sycl::fmin(yObjToGrid(oPos.y()), mHeight - 1.001f), 0.0f);
/* DPCT_ORIG         const float gZ = fmaxf( fminf(zObjToGrid(oPos.z),
 * mDepth-1.001f), 0.0f);*/
        const float gZ =
            sycl::fmax(sycl::fmin(zObjToGrid(oPos.z()), mDepth - 1.001f), 0.0f);

        const int i1 = (int) gX;
/* DPCT_ORIG         const int i0 = max(i1 - 1, 0);*/
        const int i0 = sycl::max((int)(i1 - 1), 0);
        const int i2 = i1 + 1;
/* DPCT_ORIG         const int i3 = min(i1 + 2, mWidth-1);*/
        const int i3 = sycl::min((int)(i1 + 2), (int)(mWidth - 1));
        const int j1 = (int) gY;
/* DPCT_ORIG         const int j0 = max(j1 - 1, 0);*/
        const int j0 = sycl::max((int)(j1 - 1), 0);
        const int j2 = j1 + 1;
/* DPCT_ORIG         const int j3 = min(j1 + 2, mHeight-1);*/
        const int j3 = sycl::min((int)(j1 + 2), (int)(mHeight - 1));
        const int k1 = (int) gZ;
/* DPCT_ORIG         const int k0 = max(k1 - 1, 0);*/
        const int k0 = sycl::max((int)(k1 - 1), 0);
        const int k2 = k1 + 1;
/* DPCT_ORIG         const int k3 = min(k1 + 2, mDepth-1);*/
        const int k3 = sycl::min((int)(k1 + 2), (int)(mDepth - 1));

        // cerpc and cerp are split in two functions to avoid the if/else which
        // makes the interpolation slower
        if (clamp) {
            return cerpc( cerpc( cerpc( at(i0,j0,k0), at(i1,j0,k0), at(i2,j0,k0), at(i3,j0,k0), gX-i1 ),
                                 cerpc( at(i0,j1,k0), at(i1,j1,k0), at(i2,j1,k0), at(i3,j1,k0), gX-i1 ),
                                 cerpc( at(i0,j2,k0), at(i1,j2,k0), at(i2,j2,k0), at(i3,j2,k0), gX-i1 ),
                                 cerpc( at(i0,j3,k0), at(i1,j3,k0), at(i2,j3,k0), at(i3,j3,k0), gX-i1 ), gY-j1 ),
                          cerpc( cerpc( at(i0,j0,k1), at(i1,j0,k1), at(i2,j0,k1), at(i3,j0,k1), gX-i1 ),
                                 cerpc( at(i0,j1,k1), at(i1,j1,k1), at(i2,j1,k1), at(i3,j1,k1), gX-i1 ),
                                 cerpc( at(i0,j2,k1), at(i1,j2,k1), at(i2,j2,k1), at(i3,j2,k1), gX-i1 ),
                                 cerpc( at(i0,j3,k1), at(i1,j3,k1), at(i2,j3,k1), at(i3,j3,k1), gX-i1 ), gY-j1 ),
                          cerpc( cerpc( at(i0,j0,k2), at(i1,j0,k2), at(i2,j0,k2), at(i3,j0,k2), gX-i1 ),
                                 cerpc( at(i0,j1,k2), at(i1,j1,k2), at(i2,j1,k2), at(i3,j1,k2), gX-i1 ),
                                 cerpc( at(i0,j2,k2), at(i1,j2,k2), at(i2,j2,k2), at(i3,j2,k2), gX-i1 ),
                                 cerpc( at(i0,j3,k2), at(i1,j3,k2), at(i2,j3,k2), at(i3,j3,k2), gX-i1 ), gY-j1 ),
                          cerpc( cerpc( at(i0,j0,k3), at(i1,j0,k3), at(i2,j0,k3), at(i3,j0,k3), gX-i1 ),
                                 cerpc( at(i0,j1,k3), at(i1,j1,k3), at(i2,j1,k3), at(i3,j1,k3), gX-i1 ),
                                 cerpc( at(i0,j2,k3), at(i1,j2,k3), at(i2,j2,k3), at(i3,j2,k3), gX-i1 ),
                                 cerpc( at(i0,j3,k3), at(i1,j3,k3), at(i2,j3,k3), at(i3,j3,k3), gX-i1 ), gY-j1 ), gZ-k1 );
        } else {
            return cerp( cerp( cerp( at(i0,j0,k0), at(i1,j0,k0), at(i2,j0,k0), at(i3,j0,k0), gX-i1 ),
                               cerp( at(i0,j1,k0), at(i1,j1,k0), at(i2,j1,k0), at(i3,j1,k0), gX-i1 ),
                               cerp( at(i0,j2,k0), at(i1,j2,k0), at(i2,j2,k0), at(i3,j2,k0), gX-i1 ),
                               cerp( at(i0,j3,k0), at(i1,j3,k0), at(i2,j3,k0), at(i3,j3,k0), gX-i1 ), gY-j1 ),
                         cerp( cerp( at(i0,j0,k1), at(i1,j0,k1), at(i2,j0,k1), at(i3,j0,k1), gX-i1 ),
                               cerp( at(i0,j1,k1), at(i1,j1,k1), at(i2,j1,k1), at(i3,j1,k1), gX-i1 ),
                               cerp( at(i0,j2,k1), at(i1,j2,k1), at(i2,j2,k1), at(i3,j2,k1), gX-i1 ),
                               cerp( at(i0,j3,k1), at(i1,j3,k1), at(i2,j3,k1), at(i3,j3,k1), gX-i1 ), gY-j1 ),
                         cerp( cerp( at(i0,j0,k2), at(i1,j0,k2), at(i2,j0,k2), at(i3,j0,k2), gX-i1 ),
                               cerp( at(i0,j1,k2), at(i1,j1,k2), at(i2,j1,k2), at(i3,j1,k2), gX-i1 ),
                               cerp( at(i0,j2,k2), at(i1,j2,k2), at(i2,j2,k2), at(i3,j2,k2), gX-i1 ),
                               cerp( at(i0,j3,k2), at(i1,j3,k2), at(i2,j3,k2), at(i3,j3,k2), gX-i1 ), gY-j1 ),
                         cerp( cerp( at(i0,j0,k3), at(i1,j0,k3), at(i2,j0,k3), at(i3,j0,k3), gX-i1 ),
                               cerp( at(i0,j1,k3), at(i1,j1,k3), at(i2,j1,k3), at(i3,j1,k3), gX-i1 ),
                               cerp( at(i0,j2,k3), at(i1,j2,k3), at(i2,j2,k3), at(i3,j2,k3), gX-i1 ),
                               cerp( at(i0,j3,k3), at(i1,j3,k3), at(i2,j3,k3), at(i3,j3,k3), gX-i1 ), gY-j1 ), gZ-k1 );
        }

    }

public:
    int mWidth;
    int mHeight;
    int mDepth;

/* DPCT_ORIG     float3 mDomMin;*/
    sycl::float3 mDomMin;
/* DPCT_ORIG     float3 mDomMax;*/
    sycl::float3 mDomMax;
/* DPCT_ORIG     float3 mDomSize;*/
    sycl::float3 mDomSize;

    bool mClosedBounds[6];

/* DPCT_ORIG     __device__ Grid(float *frontBuffer, float *backBuffer, int
   width, int heigth, int depth, float ox, float oy, float oz, float dx, float3
   domainMin, float3 domainMax, bool flipValue, SceneSettings scn)*/
    Grid(float *frontBuffer, float *backBuffer, int width, int heigth,
         int depth, float ox, float oy, float oz, float dx,
         sycl::float3 domainMin, sycl::float3 domainMax, bool flipValue,
         SceneSettings scn)
        : mWidth(width), mHeight(heigth), mDepth(depth), mOx(ox), mOy(oy),
          mOz(oz), mDx(dx), mByteSize(width * heigth * depth * sizeof(float)),
          mDomMin(domainMin), mDomMax(domainMax),
          /* DPCT_ORIG           mDomSize(domainMax-domainMin),
             mFlipValue(flipValue)*/
          mDomSize(dpct_operator_overloading::operator-(domainMax, domainMin)),
          mFlipValue(flipValue)
    {
        mFrontBuffer = frontBuffer;
        mBackBuffer = backBuffer;

        for (int i=0; i<6; i++) mClosedBounds[i] = scn.closedBounds[i];
    }

/* DPCT_ORIG     __device__ ~Grid() {}*/
    ~Grid() {}

/* DPCT_ORIG     __device__ inline void swap() {*/
    inline void swap() {
        float *tmp = mFrontBuffer;
        mFrontBuffer = mBackBuffer;
        mBackBuffer = tmp;
    }

/* DPCT_ORIG     __device__ inline float &atBack(int i, int j, int k) {*/
    inline float &atBack(int i, int j, int k) {
//        assert(i < mWidth);
//        assert(j < mHeight);
//        assert(k < mDepth);
        return mBackBuffer[i + j*mWidth + k*mWidth*mHeight];
    }

/* DPCT_ORIG     __device__ inline float &at(int i, int j, int k) {*/
    inline float &at(int i, int j, int k) {
//        assert(i < mWidth);
//        assert(j < mHeight);
//        assert(k < mDepth);
        return mFrontBuffer[i + j*mWidth + k*mWidth*mHeight];
    }

/* DPCT_ORIG     __device__ inline float iGridToObj(int i) {*/
    inline float iGridToObj(int i) {
//        assert(i < mWidth);
        return (i+mOx)*mDx;
    }

/* DPCT_ORIG     __device__ inline float jGridToObj(int j) {*/
    inline float jGridToObj(int j) {
//        assert(j < mHeight);
        return (j+mOy)*mDx;
    }

/* DPCT_ORIG     __device__ inline float kGridToObj(int k) {*/
    inline float kGridToObj(int k) {
//        assert(k < mDepth);
        return (k+mOz)*mDx;
    }

/* DPCT_ORIG     __device__ inline float3 gridToObj(int i, int j, int k) {*/
    inline sycl::float3 gridToObj(int i, int j, int k) {
//        assert(i < mWidth);
//        assert(j < mHeight);
//        assert(k < mDepth);
/* DPCT_ORIG         return make_float3(i+mOx, j+mOy, k+mOz)*mDx;*/
        return dpct_operator_overloading::operator*(
            sycl::float3(i + mOx, j + mOy, k + mOz), mDx);
    }

/* DPCT_ORIG     __device__ inline float3 worldToGridNoOffset(float3 wPos) {*/
    inline sycl::float3 worldToGridNoOffset(sycl::float3 wPos) {
/* DPCT_ORIG         return make_float3((wPos.x-mDomMin.x)/mDx,
                           (mDomSize.y - (wPos.y-mDomMin.y))/mDx,
                           (wPos.z-mDomMin.z)/mDx);*/
        return sycl::float3((wPos.x() - mDomMin.x()) / mDx,
                            (mDomSize.y() - (wPos.y() - mDomMin.y())) / mDx,
                            (wPos.z() - mDomMin.z()) / mDx);
    }

/* DPCT_ORIG     __device__ inline float sampleO(float3 pos, InterpolationType
 * mode, bool clamp = false) {*/
    inline float sampleO(sycl::float3 pos, InterpolationType mode,
                         bool clamp = false) {
        if (mode == linear) {
            return trilinear_interpolate(pos);
        } else if (mode == cubic) {
            return tricubic_interpolate(pos, clamp);
        } else {
            return 0.0f;
        }
    }

    // https://www.iquilezles.org/www/articles/functions/functions.htm
/* DPCT_ORIG     __device__ inline float cubicPulse(float x, float w) {*/
    inline float cubicPulse(float x, float w) {
        if( x > w ) return 0.0f;
        x /= w;
        return 1.0f - x*x*(3.0f - 2.0f*x);
    }

/* DPCT_ORIG     __device__ void sphereToGrid(float posx, float posy, float
 * posz, float pscale, float value) {*/
    void sphereToGrid(float posx, float posy, float posz, float pscale,
                      float value) {

        float radius = pscale/2.0f;

/* DPCT_ORIG         int imin = max( (int)xObjToGrid( xWorldToObj(posx - radius)
 * ), 0 );*/
        int imin = sycl::max((int)xObjToGrid(xWorldToObj(posx - radius)), 0);
/* DPCT_ORIG         int imax = min( (int)xObjToGrid( xWorldToObj(posx + radius)
 * ) + 1, mWidth);*/
        int imax = sycl::min(
            (int)((int)xObjToGrid(xWorldToObj(posx + radius)) + 1), mWidth);
/* DPCT_ORIG         int jmin = max( (int)yObjToGrid( yWorldToObj(posy + radius)
 * ), 0 );*/
        int jmin = sycl::max((int)yObjToGrid(yWorldToObj(posy + radius)), 0);
/* DPCT_ORIG         int jmax = min( (int)yObjToGrid( yWorldToObj(posy - radius)
 * ) + 1, mHeight);*/
        int jmax = sycl::min(
            (int)((int)yObjToGrid(yWorldToObj(posy - radius)) + 1), mHeight);
/* DPCT_ORIG         int kmin = max( (int)zObjToGrid( zWorldToObj(posz - radius)
 * ), 0 );*/
        int kmin = sycl::max((int)zObjToGrid(zWorldToObj(posz - radius)), 0);
/* DPCT_ORIG         int kmax = min( (int)zObjToGrid( zWorldToObj(posz + radius)
 * ) + 1, mDepth);*/
        int kmax = sycl::min(
            (int)((int)zObjToGrid(zWorldToObj(posz + radius)) + 1), mDepth);

        for (int i=imin; i<imax; i++) {
            for (int j=jmin; j<jmax; j++) {
                for (int k=kmin; k<kmax; k++) {
/* DPCT_ORIG                     float mask =
 * cubicPulse(length(worldToObj(posx,posy,posz) - gridToObj(i,j,k)), radius);*/
                    float mask = cubicPulse(
                        length(dpct_operator_overloading::operator-(
                            worldToObj(posx, posy, posz), gridToObj(i, j, k))),
                        radius);

/* DPCT_ORIG                     if (mask > 0.0f) atomicAdd(&(atBack(i, j, k)),
 * mFlipValue ? -value*mask : value*mask);*/
                    if (mask > 0.0f) dpct::atomic_fetch_add<
                        sycl::access::address_space::generic_space>(
                        &(atBack(i, j, k)),
                        mFlipValue ? -value * mask : value * mask);
                }
            }
        }
    }

};

#endif
