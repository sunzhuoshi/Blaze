#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

#ifndef RAY_H
#define RAY_H


class Ray {

public:
/* DPCT_ORIG     const float3 mOrig;*/
    const sycl::float3 mOrig;
/* DPCT_ORIG     const float3 mDir;*/
    const sycl::float3 mDir;
/* DPCT_ORIG     const float3 mInvdir;*/
    const sycl::float3 mInvdir;
/* DPCT_ORIG     const uint3 mSign;*/
    const sycl::uint3 mSign;
    float mT0, mT1;

/* DPCT_ORIG     __device__ Ray(const float3 &orig, const float3 &dir)*/
    Ray(const sycl::float3 &orig, const sycl::float3 &dir)
        /* DPCT_ORIG         : mOrig(orig), mDir(dir), mInvdir(1.0f / dir),
           mT0(0.0f), mT1(uint_as_float(0x7f800000)),*/
        : mOrig(orig), mDir(dir),
          mInvdir(dpct_operator_overloading::operator/(1.0f, dir)), mT0(0.0f),
          mT1(uint_as_float(0x7f800000)),
          /* DPCT_ORIG           mSign(make_uint3( mInvdir.x < 0.0f, mInvdir.y <
             0.0f, mInvdir.z < 0.0f ))*/
          mSign(sycl::uint3(mInvdir.x() < 0.0f, mInvdir.y() < 0.0f,
                            mInvdir.z() < 0.0f))
    { }

/* DPCT_ORIG     __device__ inline float3 at(float t) const {*/
    inline sycl::float3 at(float t) const {
/* DPCT_ORIG         return mOrig + t*mDir;*/
        return dpct_operator_overloading::operator+(
            mOrig, dpct_operator_overloading::operator*(t, mDir));
    }

};

#endif
