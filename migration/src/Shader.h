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


#ifndef SHADER_H
#define SHADER_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "SceneSettings.h"
#include "ColorRamp.h"


class Shader {

public:
    float mDensityScale;
    float mShadowDensityScale;

/* DPCT_ORIG     float3 mVolumeColor;*/
    sycl::float3 mVolumeColor;
/* DPCT_ORIG     float3 mScattering;*/
    sycl::float3 mScattering;
    float mAbsorption;
    float mSGain;

/* DPCT_ORIG     float3 mAlbedo;*/
    sycl::float3 mAlbedo;
/* DPCT_ORIG     float3 mExtinction;*/
    sycl::float3 mExtinction;

    float mEmissionScale;
    ColorRamp *mEmissionColorRamp;

    float mMultiScatterScale;
/* DPCT_ORIG     float2 mMultiScatterDensityMask;*/
    sycl::float2 mMultiScatterDensityMask;
/* DPCT_ORIG     float3 mMultiScatterColor;*/
    sycl::float3 mMultiScatterColor;

/* DPCT_ORIG     __device__ Shader(SceneSettings scn)*/
    Shader(SceneSettings scn)
        : mDensityScale(scn.shaderDensityScale),
          mShadowDensityScale(scn.shaderShadowDensityScale),
          mAbsorption(scn.shaderAbsorption), mScattering(scn.shaderScattering),
          mVolumeColor(scn.shaderVolumeColor)
          /* DPCT_ORIG       , mAlbedo((mVolumeColor*mScattering) / (mScattering
             + make_float3(mAbsorption)))*/
          ,
          mAlbedo(dpct_operator_overloading::operator/(
              (dpct_operator_overloading::operator*(mVolumeColor, mScattering)),
              (dpct_operator_overloading::operator+(
                  mScattering, make_float3(mAbsorption))))),
          mSGain(scn.shaderGain)
          /* DPCT_ORIG       , mExtinction((mScattering+mAbsorption)*-1.0f)*/
          ,
          mExtinction(dpct_operator_overloading::operator*(
              (dpct_operator_overloading::operator+(mScattering, mAbsorption)),
              -1.0f)),
          mEmissionColorRamp(new ColorRamp(scn)),
          mEmissionScale(scn.shaderEmissionScale),
          mMultiScatterScale(scn.multiScatterScale),
          mMultiScatterDensityMask(scn.multiScatterDensityMask),
          mMultiScatterColor(scn.multiScatterColor)
    {}

/* DPCT_ORIG     __device__ ~Shader(){*/
    ~Shader() {
        delete mEmissionColorRamp;
    }

/* DPCT_ORIG     __device__ float3 getEmissionColor(float temp) {*/
    sycl::float3 getEmissionColor(float temp) {
        return mEmissionColorRamp->getColor(temp);
    }

};

#endif
