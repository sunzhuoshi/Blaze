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

#ifndef LIGHT_H
#define LIGHT_H


class Light{

private:
/* DPCT_ORIG     const float3 mColor;*/
    const sycl::float3 mColor;
    const float mIntensity;
    const float mExposure;
    const int mSamples;
/* DPCT_ORIG     const float3 mDir;*/
    const sycl::float3 mDir;
/* DPCT_ORIG     const float3 mLightIllumination;*/
    const sycl::float3 mLightIllumination;

public:
/* DPCT_ORIG     __device__ Light( SceneSettings scn, int idx )*/
    Light(SceneSettings scn, int idx)
        : mColor(scn.lightColor[idx]), mIntensity(scn.lightIntensity[idx]),
          mExposure(scn.lightExposure[idx]), mSamples(scn.lightSamples[idx]),
          mDir(scn.lightDir[idx])
          /* DPCT_ORIG         , mLightIllumination(mColor * mIntensity *
             pow(2.0f, mExposure))*/
          ,
          mLightIllumination(dpct_operator_overloading::operator*(
              dpct_operator_overloading::operator*(mColor, mIntensity),
              sycl::pow<double>(2.0f, mExposure)))
    {}

/* DPCT_ORIG     __device__  void illuminate(const float3 &hitPos,
                                float3 &lightDir,
                                float3 &lightIllumination) const*/
    void illuminate(const sycl::float3 &hitPos, sycl::float3 &lightDir,
                    sycl::float3 &lightIllumination) const
    {
        lightDir = mDir;
        lightIllumination = mLightIllumination;
    }

};

#endif
