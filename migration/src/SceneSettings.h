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


#ifndef SCENESETTINGS_H
#define SCENESETTINGS_H

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <string>


struct SceneSettings {
    char sceneDir[256];
    char sceneName[256];

    char particleFile[256];
    bool sourceAnimated;
/* DPCT_ORIG     int2 sourceRange;*/
    sycl::int2 sourceRange;
    int sourceMaxParticleCount;

    char renderFile[256];
/* DPCT_ORIG     int2 renderRange;*/
    sycl::int2 renderRange;

/* DPCT_ORIG     float3 domainBboxMin;*/
    sycl::float3 domainBboxMin;
/* DPCT_ORIG     float3 domainBboxMax;*/
    sycl::float3 domainBboxMax;
    bool closedBounds[6];

    int maxIterSolve;
    float dx;
    float dt;

/* DPCT_ORIG     int3 gridRes;*/
    sycl::int3 gridRes;
/* DPCT_ORIG     int2 renderRes;*/
    sycl::int2 renderRes;

    float density;
    float gravity;
    float buoyancy;
    float coolingRate;
    float vorticityConf;
    float drag;

    int windDir; // 0:X+, 1:X-, 2:Z+, 3:Z-
    float windAmp;
    float windSpeed;
    float windTurbAmp;
    float windTurbScale;

    float turbulence_amp;
    float turbulence_scale;
/* DPCT_ORIG     float2 turbMaskTempRamp;*/
    sycl::float2 turbMaskTempRamp;
/* DPCT_ORIG     float2 turbMaskVelRamp;*/
    sycl::float2 turbMaskVelRamp;

    int lightCount;
    float lightExposure[10]; // static maximum number of lights in scene
    float lightIntensity[10]; // static maximum number of lights in scene
    unsigned int lightSamples[10]; // static maximum number of lights in scene
                                   /* DPCT_ORIG     float3 lightDir[10]; */
    sycl::float3 lightDir[10];     // static maximum number of lights in scene
                                   /* DPCT_ORIG     float3 lightColor[10]; */
    sycl::float3 lightColor[10];   // static maximum number of lights in scene

    float camFocal;
    float camAperture;
/* DPCT_ORIG     float3 camU;*/
    sycl::float3 camU;
/* DPCT_ORIG     float3 camV;*/
    sycl::float3 camV;
/* DPCT_ORIG     float3 camW;*/
    sycl::float3 camW;
/* DPCT_ORIG     float3 camTrans;*/
    sycl::float3 camTrans;

    float rndrPStep;
    float rndrSStep;
    float rndrCutoff;

    float shaderDensityScale;
    float shaderShadowDensityScale;
    float shaderAbsorption;
/* DPCT_ORIG     float3 shaderScattering;*/
    sycl::float3 shaderScattering;
/* DPCT_ORIG     float3 shaderVolumeColor;*/
    sycl::float3 shaderVolumeColor;
    float shaderGain;

    float shaderEmissionScale;
    int shaderTempRampSize;
/* DPCT_ORIG     float2 shaderTempRange;*/
    sycl::float2 shaderTempRange;
    float shaderColorRampKeys[10]; // static maximum number of keys in the ramp
/* DPCT_ORIG     float3 shaderColorRampColors[10]; */
    sycl::float3
        shaderColorRampColors[10]; // static maximum number of keys in the ramp

    float multiScatterScale;
/* DPCT_ORIG     float2 multiScatterDensityMask;*/
    sycl::float2 multiScatterDensityMask;
/* DPCT_ORIG     float3 multiScatterColor;*/
    sycl::float3 multiScatterColor;
    int multiScatterBlurIter;

    std::string getSceneInfo() {
        std::string msg = "";
        char tmp[256];

        sprintf(tmp, "sceneDir: %s\n\n", sceneDir); msg += tmp;
        sprintf(tmp, "sceneName: %s\n\n", sceneName); msg += tmp;

        sprintf(tmp, "particleFile: %s\n", particleFile); msg += tmp;
        sprintf(tmp, "sourceAnimated: %d\n", sourceAnimated); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "sourceRange: (%d, %d)\n", sourceRange.x,
 * sourceRange.y); msg += tmp;*/
        sprintf(tmp, "sourceRange: (%d, %d)\n", sourceRange.x(),
                sourceRange.y());
            msg += tmp;
        sprintf(tmp, "sourceMaxParticleCount: %d\n\n", sourceMaxParticleCount); msg += tmp;

        sprintf(tmp, "renderFile: %s\n", renderFile); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "renderRange: (%d, %d)\n\n", renderRange.x,
 * renderRange.y); msg += tmp;*/
        sprintf(tmp, "renderRange: (%d, %d)\n\n", renderRange.x(),
                renderRange.y());
            msg += tmp;

/* DPCT_ORIG         sprintf(tmp, "domainBboxMin:  (%4.4f, %4.4f, %4.4f)\n",
 * domainBboxMin.x, domainBboxMin.y, domainBboxMin.z); msg += tmp;*/
        sprintf(tmp, "domainBboxMin:  (%4.4f, %4.4f, %4.4f)\n",
                domainBboxMin.x(), domainBboxMin.y(), domainBboxMin.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "domainBboxMax:  (%4.4f, %4.4f, %4.4f)\n",
 * domainBboxMax.x, domainBboxMax.y, domainBboxMax.z); msg += tmp;*/
        sprintf(tmp, "domainBboxMax:  (%4.4f, %4.4f, %4.4f)\n",
                domainBboxMax.x(), domainBboxMax.y(), domainBboxMax.z());
            msg += tmp;
        sprintf(tmp, "closedBounds:  (%d, %d, %d, %d, %d, %d)\n\n", closedBounds[0], closedBounds[1], closedBounds[2],
                                                                    closedBounds[3], closedBounds[4], closedBounds[5]);msg += tmp;

        sprintf(tmp, "maxIterSolve: %d\n", maxIterSolve); msg += tmp;
        sprintf(tmp, "dt: %4.4f\n", dt); msg += tmp;
        sprintf(tmp, "dx: %4.4f\n\n", dx); msg += tmp;

/* DPCT_ORIG         sprintf(tmp, "gridRes: (%d, %d, %d)\n", gridRes.x,
 * gridRes.y, gridRes.z); msg += tmp;*/
        sprintf(tmp, "gridRes: (%d, %d, %d)\n", gridRes.x(), gridRes.y(),
                gridRes.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "renderRes: (%d, %d)\n\n", renderRes.x,
 * renderRes.y); msg += tmp;*/
        sprintf(tmp, "renderRes: (%d, %d)\n\n", renderRes.x(), renderRes.y());
            msg += tmp;

        sprintf(tmp, "density: %4.4f\n", density); msg += tmp;
        sprintf(tmp, "gravity: %4.4f\n", gravity); msg += tmp;
        sprintf(tmp, "coolingRate: %4.4f\n", coolingRate); msg += tmp;
        sprintf(tmp, "buoyancy: %4.4f\n", buoyancy); msg += tmp;
        sprintf(tmp, "vorticityConf: %4.4f\n", vorticityConf); msg += tmp;
        sprintf(tmp, "drag: %4.4f\n\n", drag); msg += tmp;

        sprintf(tmp, "windDir: %d\n", windDir); msg += tmp;
        sprintf(tmp, "windAmp: %4.4f\n", windAmp); msg += tmp;
        sprintf(tmp, "windSpeed: %4.4f\n", windSpeed); msg += tmp;
        sprintf(tmp, "windTurbAmp: %4.4f\n", windTurbAmp); msg += tmp;
        sprintf(tmp, "windTurbScale: %4.4f\n\n", windTurbScale); msg += tmp;

        sprintf(tmp, "turbulence_amp: %4.4f\n", turbulence_amp); msg += tmp;
        sprintf(tmp, "turbulence_scale: %4.4f\n", turbulence_scale); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "turbMaskTempRamp: (%4.4f, %4.4f)\n",
 * turbMaskTempRamp.x, turbMaskTempRamp.y); msg += tmp;*/
        sprintf(tmp, "turbMaskTempRamp: (%4.4f, %4.4f)\n", turbMaskTempRamp.x(),
                turbMaskTempRamp.y());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "turbMaskVelRamp: (%4.4f, %4.4f)\n\n",
 * turbMaskVelRamp.x, turbMaskVelRamp.y); msg += tmp;*/
        sprintf(tmp, "turbMaskVelRamp: (%4.4f, %4.4f)\n\n", turbMaskVelRamp.x(),
                turbMaskVelRamp.y());
            msg += tmp;

        sprintf(tmp, "lightCount: %d\n", lightCount); msg += tmp;
        for (int i=0; i<lightCount; i++) {
            sprintf(tmp, "lightExposure: %4.4f\n", lightExposure[i]); msg += tmp;
            sprintf(tmp, "lightIntensity: %4.4f\n", lightIntensity[i]); msg += tmp;
            sprintf(tmp, "lightSamples: %d\n", lightSamples[i]); msg += tmp;
/* DPCT_ORIG             sprintf(tmp, "lightDir: (%4.4f, %4.4f, %4.4f)\n",
 * lightDir[i].x,*/
            sprintf(tmp, "lightDir: (%4.4f, %4.4f, %4.4f)\n", lightDir[i].x(),
                    /* DPCT_ORIG lightDir[i].y,*/
                    lightDir[i].y(),
                    /* DPCT_ORIG lightDir[i].z); msg += tmp;*/
                    lightDir[i].z());
                msg += tmp;
/* DPCT_ORIG             sprintf(tmp, "lightColor: (%4.4f, %4.4f, %4.4f)\n",
 * lightColor[i].x,*/
            sprintf(tmp, "lightColor: (%4.4f, %4.4f, %4.4f)\n",
                    lightColor[i].x(),
                    /* DPCT_ORIG lightColor[i].y,*/
                    lightColor[i].y(),
                    /* DPCT_ORIG lightColor[i].z); msg += tmp;*/
                    lightColor[i].z());
                msg += tmp;
        }
        sprintf(tmp, "\n"); msg += tmp;

        sprintf(tmp, "camFocal: %4.4f\n", camFocal); msg += tmp;
        sprintf(tmp, "camAperture: %4.4f\n", camAperture); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "camU: (%4.4f, %4.4f, %4.4f)\n", camU.x,
 * camU.y, camU.z); msg += tmp;*/
        sprintf(tmp, "camU: (%4.4f, %4.4f, %4.4f)\n", camU.x(), camU.y(),
                camU.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "camV: (%4.4f, %4.4f, %4.4f)\n", camV.x,
 * camV.y, camV.z); msg += tmp;*/
        sprintf(tmp, "camV: (%4.4f, %4.4f, %4.4f)\n", camV.x(), camV.y(),
                camV.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "camW: (%4.4f, %4.4f, %4.4f)\n", camW.x,
 * camW.y, camW.z); msg += tmp;*/
        sprintf(tmp, "camW: (%4.4f, %4.4f, %4.4f)\n", camW.x(), camW.y(),
                camW.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "camTrans: (%4.4f, %4.4f, %4.4f)\n\n",
 * camTrans.x, camTrans.y, camTrans.z); msg += tmp;*/
        sprintf(tmp, "camTrans: (%4.4f, %4.4f, %4.4f)\n\n", camTrans.x(),
                camTrans.y(), camTrans.z());
            msg += tmp;

        sprintf(tmp, "rndrPStep: %4.4f\n", rndrPStep); msg += tmp;
        sprintf(tmp, "rndrSStep: %4.4f\n", rndrSStep); msg += tmp;
        sprintf(tmp, "rndrCutoff: %4.4f\n\n", rndrCutoff); msg += tmp;

        sprintf(tmp, "shaderDensityScale: %4.4f\n", shaderDensityScale); msg += tmp;
        sprintf(tmp, "shaderShadowDensityScale: %4.4f\n", shaderShadowDensityScale); msg += tmp;
        sprintf(tmp, "shaderAbsorption: %4.4f\n", shaderAbsorption); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "shaderScattering: (%4.4f, %4.4f, %4.4f)\n",
 * shaderScattering.x, shaderScattering.y, shaderScattering.z); msg += tmp;*/
        sprintf(tmp, "shaderScattering: (%4.4f, %4.4f, %4.4f)\n",
                shaderScattering.x(), shaderScattering.y(),
                shaderScattering.z());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "shaderVolumeColor: (%4.4f, %4.4f, %4.4f)\n",
 * shaderVolumeColor.x, shaderVolumeColor.y, shaderVolumeColor.z); msg += tmp;*/
        sprintf(tmp, "shaderVolumeColor: (%4.4f, %4.4f, %4.4f)\n",
                shaderVolumeColor.x(), shaderVolumeColor.y(),
                shaderVolumeColor.z());
            msg += tmp;
        sprintf(tmp, "shaderGain: %4.4f\n\n", shaderGain); msg += tmp;

        sprintf(tmp, "shaderEmissionScale: %4.4f\n", shaderEmissionScale); msg += tmp;
        sprintf(tmp, "shaderTempRampSize: %d\n", shaderTempRampSize); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "shaderTempRange: (%4.4f, %4.4f)\n",
 * shaderTempRange.x, shaderTempRange.y); msg += tmp;*/
        sprintf(tmp, "shaderTempRange: (%4.4f, %4.4f)\n", shaderTempRange.x(),
                shaderTempRange.y());
            msg += tmp;
        for (int i=0; i<shaderTempRampSize; i++) {
            sprintf(tmp, "shaderColorRampKeys: %4.4f\n", shaderColorRampKeys[i]); msg += tmp;
/* DPCT_ORIG             sprintf(tmp, "shaderColorRampColors: (%4.4f, %4.4f,
 * %4.4f)\n", shaderColorRampColors[i].x,*/
            sprintf(tmp, "shaderColorRampColors: (%4.4f, %4.4f, %4.4f)\n",
                    shaderColorRampColors[i].x(),
                    /* DPCT_ORIG shaderColorRampColors[i].y,*/
                    shaderColorRampColors[i].y(),
                    /* DPCT_ORIG shaderColorRampColors[i].z); msg += tmp;*/
                    shaderColorRampColors[i].z());
                msg += tmp;
        }
        sprintf(tmp, "\n"); msg += tmp;

        sprintf(tmp, "multiScatterScale: %4.4f\n", multiScatterScale); msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "multiScatterDensityMask: (%4.4f, %4.4f)\n",
 * multiScatterDensityMask.x,*/
        sprintf(tmp, "multiScatterDensityMask: (%4.4f, %4.4f)\n",
                multiScatterDensityMask.x(),
                /* DPCT_ORIG multiScatterDensityMask.y); msg += tmp;*/
                multiScatterDensityMask.y());
            msg += tmp;
/* DPCT_ORIG         sprintf(tmp, "multiScatterColor: (%4.4f, %4.4f, %4.4f)\n",
 * multiScatterColor.x,*/
        sprintf(tmp, "multiScatterColor: (%4.4f, %4.4f, %4.4f)\n",
                multiScatterColor.x(),
                /* DPCT_ORIG multiScatterColor.y,*/
                multiScatterColor.y(),
                /* DPCT_ORIG multiScatterColor.z); msg += tmp;*/
                multiScatterColor.z());
            msg += tmp;
        sprintf(tmp, "multiScatterBlurIter: %d\n", multiScatterBlurIter); msg += tmp;

        return msg;
    }
};

#endif
