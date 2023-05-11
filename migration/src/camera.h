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


#ifndef CAMERA_H
#define CAMERA_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "Ray.h"

class Camera {

public:
/* DPCT_ORIG     const float3 mOrigin;*/
    const sycl::float3 mOrigin;
/* DPCT_ORIG     const float3 mU, mV, mW;*/
    const sycl::float3 mU, mV, mW;
/* DPCT_ORIG     const uint2 mRes;*/
    const sycl::uint2 mRes;
/* DPCT_ORIG     float3 mTopLeftCorner;*/
    sycl::float3 mTopLeftCorner;
/* DPCT_ORIG     float3 mHorizontal;*/
    sycl::float3 mHorizontal;
/* DPCT_ORIG     float3 mVertical;*/
    sycl::float3 mVertical;
    float mPixelWidth;

/* DPCT_ORIG     __device__ Camera(float3 origin, float3 u, float3 v, float3 w,
 * float focal, float aperture, int xres, int yres)*/
    Camera(sycl::float3 origin, sycl::float3 u, sycl::float3 v, sycl::float3 w,
           float focal, float aperture, int xres, int yres)
        /* DPCT_ORIG         : mOrigin(origin), mU(u), mV(v), mW(w),
           mRes(make_uint2(xres, yres))*/
        : mOrigin(origin), mU(u), mV(v), mW(w), mRes(sycl::uint2(xres, yres))
    {
        float aspect = (float)xres / (float)yres;
        float halfWidth = 0.5f*aperture;
        float halfHeight = halfWidth / aspect;

        mPixelWidth = halfWidth / (xres/2.0f);
/* DPCT_ORIG         mTopLeftCorner = origin - focal*mW - halfWidth*mU +
 * halfHeight*mV;*/
        mTopLeftCorner = dpct_operator_overloading::operator+(
            dpct_operator_overloading::operator-(
                dpct_operator_overloading::operator-(
                    origin, dpct_operator_overloading::operator*(focal, mW)),
                dpct_operator_overloading::operator*(halfWidth, mU)),
            dpct_operator_overloading::operator*(halfHeight, mV));
/* DPCT_ORIG         mHorizontal = 2.0f*halfWidth*u;*/
        mHorizontal = dpct_operator_overloading::operator*(2.0f * halfWidth, u);
/* DPCT_ORIG         mVertical = -2.0f*halfHeight*v;*/
        mVertical = dpct_operator_overloading::operator*(-2.0f * halfHeight, v);
    }

/* DPCT_ORIG     __device__ Ray getRay(int i, int j) {*/
    Ray getRay(int i, int j) {

/* DPCT_ORIG         return Ray(mOrigin, normalize(mTopLeftCorner +
 * ((i+0.5f)/mRes.x)*mHorizontal*/
        return Ray(mOrigin,
                   normalize(dpct_operator_overloading::operator-(
                       dpct_operator_overloading::operator+(
                           dpct_operator_overloading::operator+(
                               mTopLeftCorner,
                               dpct_operator_overloading::operator*(
                                   ((i + 0.5f) / mRes.x()), mHorizontal))
                           /* DPCT_ORIG + ((j+0.5f)/mRes.y)*mVertical*/
                           ,
                           dpct_operator_overloading::operator*(
                               ((j + 0.5f) / mRes.y()), mVertical))
                       /* DPCT_ORIG - mOrigin));*/
                       ,
                       mOrigin)));
    }

};

#endif
