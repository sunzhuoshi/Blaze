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

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../thirdparty/lodepng/lodepng.h"
#define TINYEXR_IMPLEMENTATION
#include "../thirdparty/tinyexr/tinyexr.h"
#include "utils/helper_math.h"

#include "RenderEngine.h"
#include <time.h>

/* DPCT_ORIG __constant__ float4 c_filterData[3*3*3];*/
static dpct::constant_memory<sycl::float4, 1> c_filterData(3 * 3 * 3);

const float RenderEngine::mFilterBlur[RenderEngine::mFilterSize] =
{
    0,1,0,
    1,2,1,
    0,1,0,

    1,2,1,
    2,4,2,
    1,2,1,

    0,1,0,
    1,2,1,
    0,1,0,
};

/* DPCT_ORIG __global__ void gaussian_blur_texture_kernel(cudaTextureObject_t
   volumeTexIn, cudaSurfaceObject_t volumeTexOut, int filterSize, cudaExtent
   volumeSize)*/
void gaussian_blur_texture_kernel(
    dpct::image_wrapper_base_p volumeTexIn,
    dpct::image_wrapper_base_p volumeTexOut, int filterSize, sycl::range<3> volumeSize,
    const sycl::nd_item<3> &item_ct1, sycl::float4 *c_filterData)
{
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

/* DPCT_ORIG     if (!(i < volumeSize.width && j < volumeSize.height && k <
 * volumeSize.depth)) return;*/
    if (!(i < volumeSize[0] && j < volumeSize[1] && k < volumeSize[2])) return;

    float filtered = 0;
/* DPCT_ORIG     float4 basecoord = make_float4(i+0.5f, j+0.5f, k+0.5f, 0);*/
    sycl::float4 basecoord = sycl::float4(i + 0.5f, j + 0.5f, k + 0.5f, 0);

    for (int i=0; i<filterSize; i++)
    {
/* DPCT_ORIG         float4 coord = basecoord + c_filterData[i];*/
        sycl::float4 coord =
            dpct_operator_overloading::operator+(basecoord, c_filterData[i]);
/* DPCT_ORIG         filtered  += tex3D<float>(volumeTexIn, coord.x, coord.y,
 * coord.z) * c_filterData[i].w;*/
        //filtered += volumeTexIn.read(coord.x(), coord.y(), coord.z()) *
        //            c_filterData[i].w();
    }

    // surface writes need byte offsets for x!
    dpcx::surf3Dwrite(filtered, volumeTexOut, i*sizeof(float), j, k);
}

/* DPCT_ORIG __global__ void write_scatter_to_3d_tex_kernel(Grid **tempGrid,
   cudaTextureObject_t scatterTex, float scatterTempMin, int resRatio,
   cudaExtent volumeSize)*/
/*
DPCT1050:351: The template argument of the image_accessor_ext could not be
deduced. You need to update this code.
*/
void write_scatter_to_3d_tex_kernel(
    Grid **tempGrid,
    dpct::image_wrapper_base_p
        scatterTex,
    float scatterTempMin, int resRatio, sycl::range<3> volumeSize,
    const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

/* DPCT_ORIG     if (!(i < volumeSize.width && j < volumeSize.height && k <
 * volumeSize.depth)) return;*/
    if (!(i < volumeSize[0] && j < volumeSize[1] && k < volumeSize[2])) return;

    // downscale 64 temp voxel to 1 scatter voxel
    float output = 0.0f;
    for (int io=0; io<resRatio; io++) {
        for (int jo=0; jo<resRatio; jo++) {
            for (int ko=0; ko<resRatio; ko++) {
                float t = (*tempGrid)->at(i*resRatio+io, j*resRatio+jo, k*resRatio+ko);
                output += t>scatterTempMin?t:0.0f;
            }
        }
    }
    output /= (float)(resRatio*resRatio*resRatio);

    // surface writes need byte offsets for x!
    dpcx::surf3Dwrite(output, scatterTex, (int)i*sizeof(float), j, k);
}

/* DPCT_ORIG __global__ void write_temp_to_3d_tex_kernel(Grid **tempGrid,
 * cudaTextureObject_t tempTex)*/
/*
DPCT1050:352: The template argument of the image_accessor_ext could not be
deduced. You need to update this code.
*/
void write_temp_to_3d_tex_kernel(
    Grid **tempGrid,
    dpct::image_wrapper_base_p tempTex,
    const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

//    assert(i < (*tempGrid)->mWidth && j < (*tempGrid)->mHeight && k < (*tempGrid)->mDepth);

    float output = (*tempGrid)->at(i,j,k);

    // surface writes need byte offsets for x!
    dpcx::surf3Dwrite(output, tempTex, (int)i*sizeof(float), j, k);
}

// https://www.scratchapixel.com/
/* DPCT_ORIG __device__ bool bbox_intersect_kernel(Ray &r, float3 bounds[2]) {*/
bool bbox_intersect_kernel(Ray &r, sycl::float3 bounds[2]) {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

/* DPCT_ORIG     tmin = (bounds[r.mSign.x].x - r.mOrig.x) * r.mInvdir.x;*/
    tmin = (bounds[r.mSign.x()].x() - r.mOrig.x()) * r.mInvdir.x();
/* DPCT_ORIG     tmax = (bounds[1-r.mSign.x].x - r.mOrig.x) * r.mInvdir.x;*/
    tmax = (bounds[1 - r.mSign.x()].x() - r.mOrig.x()) * r.mInvdir.x();
/* DPCT_ORIG     tymin = (bounds[r.mSign.y].y - r.mOrig.y) * r.mInvdir.y;*/
    tymin = (bounds[r.mSign.y()].y() - r.mOrig.y()) * r.mInvdir.y();
/* DPCT_ORIG     tymax = (bounds[1-r.mSign.y].y - r.mOrig.y) * r.mInvdir.y;*/
    tymax = (bounds[1 - r.mSign.y()].y() - r.mOrig.y()) * r.mInvdir.y();

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

/* DPCT_ORIG     tzmin = (bounds[r.mSign.z].z - r.mOrig.z) * r.mInvdir.z;*/
    tzmin = (bounds[r.mSign.z()].z() - r.mOrig.z()) * r.mInvdir.z();
/* DPCT_ORIG     tzmax = (bounds[1-r.mSign.z].z - r.mOrig.z) * r.mInvdir.z;*/
    tzmax = (bounds[1 - r.mSign.z()].z() - r.mOrig.z()) * r.mInvdir.z();

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

/* DPCT_ORIG     r.mT0 = fmaxf(tmin, 0.0f);*/
    r.mT0 = sycl::fmax(tmin, 0.0f);
    r.mT1 = tmax;

    if (r.mT1 < 0.0f)
        return false;

    return true;
}

/* DPCT_ORIG __global__ void render_kernel(float3 *frameBuffer, Grid **tempGrid,
                              cudaTextureObject_t tempTex,
                              cudaTextureObject_t scatterTex,
                              Shader **shader, Light **lights, int lightCount,
                              Camera **cam, float dx, int scatterResRatio,
                              float pStep, float sStep, float cutoff)*/
void render_kernel(sycl::float3 *frameBuffer, Grid **tempGrid,
                   dpct::image_accessor_ext<float, 3> tempTex,
                   dpct::image_accessor_ext<float, 3> scatterTex,
                   Shader **shader, Light **lights, int lightCount,
                   Camera **cam, float dx, int scatterResRatio, float pStep,
                   float sStep, float cutoff, const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);

    // initialize the pixels to black
/* DPCT_ORIG     frameBuffer[i + (*cam)->mRes.x*j] = make_float3(0.0f);*/
    frameBuffer[i + (*cam)->mRes.x() * j] = make_float3(0.0f);

    Ray pRay = (*cam)->getRay(i,j);
/* DPCT_ORIG     float3 bounds[] = {(*tempGrid)->mDomMin,
 * (*tempGrid)->mDomMax};*/
    sycl::float3 bounds[] = {(*tempGrid)->mDomMin, (*tempGrid)->mDomMax};

    bool hit = bbox_intersect_kernel(pRay, bounds);
    if (!hit) return;

/* DPCT_ORIG     float3 pLumi = make_float3(0.0f);*/
    sycl::float3 pLumi = make_float3(0.0f);
/* DPCT_ORIG     float3 pTrans = make_float3(1.0f);*/
    sycl::float3 pTrans = make_float3(1.0f);
/* DPCT_ORIG     float3 one = make_float3(1.0f);*/
    sycl::float3 one = make_float3(1.0f);

    // primary ray loop
    for (float pT = pRay.mT0+dx*0.5; pT <= pRay.mT1; pT += pStep) {

/* DPCT_ORIG         float3 pPos = pRay.at(pT);*/
        sycl::float3 pPos = pRay.at(pT);
/* DPCT_ORIG         float3 pPosGrid = (*tempGrid)->worldToGridNoOffset( pPos
 * );*/
        sycl::float3 pPosGrid = (*tempGrid)->worldToGridNoOffset(pPos);

/* DPCT_ORIG         const float pTemp = tex3D<float>(tempTex, pPosGrid.x,
 * pPosGrid.y, pPosGrid.z);*/
        const float pTemp =
            tempTex.read(pPosGrid.x(), pPosGrid.y(), pPosGrid.z());
/* DPCT_ORIG         const float pScatter = tex3D<float>(scatterTex,
   pPosGrid.x/(float)scatterResRatio, pPosGrid.y/(float)scatterResRatio,
                                                        pPosGrid.z/(float)scatterResRatio);*/
        const float pScatter =
            scatterTex.read(pPosGrid.x() / (float)scatterResRatio,
                            pPosGrid.y() / (float)scatterResRatio,
                            pPosGrid.z() / (float)scatterResRatio);
        const float pDensity = pTemp * (*shader)->mDensityScale;

        if (pDensity < cutoff) continue;

/* DPCT_ORIG         float3 emission = make_float3(0.0f);*/
        sycl::float3 emission = make_float3(0.0f);
        if ((*shader)->mEmissionScale > 0.0f) {
/* DPCT_ORIG             emission = (*shader)->getEmissionColor(pTemp) *
 * (*shader)->mEmissionScale*/
            emission = dpct_operator_overloading::operator+(
                dpct_operator_overloading::operator*(
                    (*shader)->getEmissionColor(pTemp),
                    (*shader)->mEmissionScale)
                /* DPCT_ORIG                      +
                   (*shader)->mMultiScatterColor * pScatter *
                   (*shader)->mMultiScatterScale*/
                ,
                dpct_operator_overloading::operator*(
                    dpct_operator_overloading::operator*(
                        dpct_operator_overloading::operator*(
                            (*shader)->mMultiScatterColor, pScatter),
                        (*shader)->mMultiScatterScale)
                    /* DPCT_ORIG                      *
                       smoothstep((*shader)->mMultiScatterDensityMask.x,
                       (*shader)->mMultiScatterDensityMask.y, pTemp);*/
                    ,
                    smoothstep((*shader)->mMultiScatterDensityMask.x(),
                               (*shader)->mMultiScatterDensityMask.y(),
                               pTemp)));
        }

        // compute the delta transmitance using Lambert-Beers law (P.176 PVR)
/* DPCT_ORIG         const float3 dT = expf((*shader)->mExtinction * pDensity *
 * pStep);*/
        const sycl::float3 dT = expf(dpct_operator_overloading::operator*(
            dpct_operator_overloading::operator*((*shader)->mExtinction,
                                                 pDensity),
            pStep));

        // light loop
        for (int i=0; i<lightCount; i++) {
/* DPCT_ORIG             float3 lightDir, lightIntensity;*/
            sycl::float3 lightDir, lightIntensity;
            lights[i]->illuminate(pPos, lightDir, lightIntensity);

/* DPCT_ORIG             Ray sRay(pPos, -1.0f*lightDir);*/
            Ray sRay(pPos,
                     dpct_operator_overloading::operator*(-1.0f, lightDir));
            hit = bbox_intersect_kernel(sRay, bounds);

/* DPCT_ORIG             float3 sTrans = make_float3(1.0f);*/
            sycl::float3 sTrans = make_float3(1.0f);

            // secondary ray loop
            for (float sT = sRay.mT0+dx*0.5; sT <= sRay.mT1; sT += sStep) {
/* DPCT_ORIG                 float3 sPosGrid = (*tempGrid)->worldToGridNoOffset(
 * sRay.at(sT) );*/
                sycl::float3 sPosGrid =
                    (*tempGrid)->worldToGridNoOffset(sRay.at(sT));
/* DPCT_ORIG                 const float sDensity = tex3D<float>(tempTex,
 * sPosGrid.x, sPosGrid.y, sPosGrid.z) * (*shader)->mDensityScale *
 * (*shader)->mShadowDensityScale;*/
                const float sDensity =
                    tempTex.read(sPosGrid.x(), sPosGrid.y(), sPosGrid.z()) *
                    (*shader)->mDensityScale * (*shader)->mShadowDensityScale;
                if (sDensity < cutoff) continue;
/* DPCT_ORIG                 sTrans *= expf((*shader)->mExtinction * sDensity *
 * sStep/(1.0f+sT*(*shader)->mSGain));*/
                dpct_operator_overloading::operator*=(
                    sTrans, expf(dpct_operator_overloading::operator/(
                                dpct_operator_overloading::operator*(
                                    dpct_operator_overloading::operator*(
                                        (*shader)->mExtinction, sDensity),
                                    sStep),
                                (1.0f + sT * (*shader)->mSGain))));
                if (squared_length(sTrans) < cutoff) break;
            }

/* DPCT_ORIG             pLumi += (*shader)->mAlbedo * sTrans * pTrans *
 * lightIntensity * (one-dT);*/
            dpct_operator_overloading::operator+=(
                pLumi, dpct_operator_overloading::operator*(
                           dpct_operator_overloading::operator*(
                               dpct_operator_overloading::operator*(
                                   dpct_operator_overloading::operator*(
                                       (*shader)->mAlbedo, sTrans),
                                   pTrans),
                               lightIntensity),
                           (dpct_operator_overloading::operator-(one, dT))));
        }
/* DPCT_ORIG         pLumi += pTrans * emission * (one-dT);*/
        dpct_operator_overloading::operator+=(
            pLumi, dpct_operator_overloading::operator*(
                       dpct_operator_overloading::operator*(pTrans, emission),
                       (dpct_operator_overloading::operator-(one, dT))));
/* DPCT_ORIG         pTrans *= dT;*/
        dpct_operator_overloading::operator*=(pTrans, dT);

        if (squared_length(pTrans) < cutoff) break;
    }

/* DPCT_ORIG     frameBuffer[i + (*cam)->mRes.x*j] = pLumi;*/
    frameBuffer[i + (*cam)->mRes.x() * j] = pLumi;
}

/* DPCT_ORIG __global__ void create_scene_kernel(Camera **d_mCamera, Shader
   **d_mShader, Light **d_mLights, SceneSettings scn, float dx)*/
void create_scene_kernel(Camera **d_mCamera, Shader **d_mShader,
                         Light **d_mLights, SceneSettings scn, float dx)
{
/* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
    *d_mCamera =
        new(*d_mCamera) Camera(scn.camTrans, scn.camU, scn.camV, scn.camW, scn.camFocal,
                    /* DPCT_ORIG scn.camAperture, scn.renderRes.x,
                        scn.renderRes.y);*/
                    scn.camAperture, scn.renderRes.x(), scn.renderRes.y());
    *d_mShader = new(*d_mShader) Shader(scn);
    for (int i=0; i<scn.lightCount; i++) {
        d_mLights[i] = new(d_mLights[i]) Light(scn, i);
    }
}

/* DPCT_ORIG __global__ void delete_scene_kernel(Light **d_mLights, int
   lightCount, Shader **d_mShader, Camera **d_mCamera) {*/
void delete_scene_kernel(Light **d_mLights, int lightCount, Shader **d_mShader,
                         Camera **d_mCamera, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
        delete *d_mShader;
        delete *d_mCamera;
        for (int i=0; i<lightCount; i++) {
            delete d_mLights[i];
        }
    }
}

RenderEngine::RenderEngine(Timer *tmr, SceneSettings scn)
    : mTmr(tmr)
      /* DPCT_ORIG     , mWidth(scn.gridRes.x)*/
      ,
      mWidth(scn.gridRes.x())
      /* DPCT_ORIG     , mHeight(scn.gridRes.y)*/
      ,
      mHeight(scn.gridRes.y())
      /* DPCT_ORIG     , mDepth(scn.gridRes.z)*/
      ,
      mDepth(scn.gridRes.z())
      /* DPCT_ORIG     , mRdrWidth(scn.renderRes.x)*/
      ,
      mRdrWidth(scn.renderRes.x())
      /* DPCT_ORIG     , mRdrHeight(scn.renderRes.y)*/
      ,
      mRdrHeight(scn.renderRes.y()), mScatterScale(scn.multiScatterScale)
      /* DPCT_ORIG     , mScatterTempMin(scn.shaderTempRange.x)*/
      ,
      mScatterTempMin(scn.shaderTempRange.x()),
      mScatterBlurIter(scn.multiScatterBlurIter), mDx(scn.dx),
      mLightCount(scn.lightCount), mPStep(scn.rndrPStep), mSStep(scn.rndrSStep),
      mCutoff(scn.rndrCutoff), mRenderFile(scn.renderFile),
      mSceneDir(scn.sceneDir)
{
/* DPCT_ORIG     int byteSizeFloat3 = mRdrWidth*mRdrHeight*sizeof(float3);*/
    /*
    DPCT1083:221: The size of float3 in the migrated code may be different from
    the original code. Check that the allocated memory size in the migrated code
    is correct.
    */
    int byteSizeFloat3 = mRdrWidth * mRdrHeight * sizeof(sycl::float3);

    // allocate frameBuffer shared between CPU and GPU
/* DPCT_ORIG     checkCudaErrors(cudaMallocManaged((void **)&mFrameBuffer,
 * byteSizeFloat3));*/
    /*
    DPCT1003:222: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((mFrameBuffer = (sycl::float3 *)sycl::malloc_shared(
                         byteSizeFloat3, dpct::get_default_queue()),
                     0));

/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mShader, sizeof(Shader
 * *)));*/
    /*
    DPCT1003:223: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((
        d_mShader = sycl::malloc_device<Shader *>(1, dpct::get_default_queue()),
        0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mCamera, sizeof(Camera
 * *)));*/
    /*
    DPCT1003:224: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((
        d_mCamera = sycl::malloc_device<Camera *>(1, dpct::get_default_queue()),
        0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mLights,
 * scn.lightCount*sizeof(Light *)));*/
    /*
    DPCT1003:225: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mLights = sycl::malloc_device<Light *>(
                         scn.lightCount, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     create_scene_kernel <<< 1, 1 >>> (d_mCamera, d_mShader,
 * d_mLights, scn, mDx);*/
    create_scene_kernel(d_mCamera, d_mShader, d_mLights, scn, mDx);
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:226: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:227: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    mTempVol = new Volume(mWidth, mHeight, mDepth);
    mScatterFrontVol = new Volume(mWidth/mScatterResRatio, mHeight/mScatterResRatio, mDepth/mScatterResRatio);
    mScatterBackVol = new Volume(mWidth/mScatterResRatio, mHeight/mScatterResRatio, mDepth/mScatterResRatio);

    // setup gaussian blur filter weights
/* DPCT_ORIG     float4 weights[mFilterSize];*/
    sycl::float4 weights[mFilterSize];

    float sum = 0;
    for (int i=0; i<mFilterSize; i++) sum += mFilterBlur[i];

    int idx = 0;
    for (int k=-mFilterWidth/2; k<mFilterWidth/2+1; k++) {
        for (int j=-mFilterWidth/2; j<mFilterWidth/2+1; j++) {
            for (int i=-mFilterWidth/2; i<mFilterWidth/2+1; i++, idx++) {
/* DPCT_ORIG                 weights[idx] = make_float4(i, j, k,
 * mFilterBlur[idx]/sum);*/
                weights[idx] = sycl::float4(i, j, k, mFilterBlur[idx] / sum);
            }
        }
    }
/* DPCT_ORIG     checkCudaErrors(cudaMemcpyToSymbol(c_filterData, weights,
 * sizeof(float4)*mFilterSize));*/
    /*
    DPCT1003:228: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_default_queue()
                         .memcpy(c_filterData.get_ptr(), weights,
                                 sizeof(sycl::float4) * mFilterSize)
                         .wait(),
                     0));
}

RenderEngine::~RenderEngine() {
    delete mTempVol;
    delete mScatterFrontVol;
    delete mScatterBackVol;

/* DPCT_ORIG     checkCudaErrors(cudaFree(mFrameBuffer));*/
    /*
    DPCT1003:229: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(mFrameBuffer, dpct::get_default_queue()), 0));
/* DPCT_ORIG     delete_scene_kernel <<< 1, 1 >>> (d_mLights, mLightCount,
 * d_mShader, d_mCamera);*/
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mLights_ct0 = d_mLights;
        auto mLightCount_ct1 = mLightCount;
        auto d_mShader_ct2 = d_mShader;
        auto d_mCamera_ct3 = d_mCamera;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                delete_scene_kernel(d_mLights_ct0, mLightCount_ct1,
                                    d_mShader_ct2, d_mCamera_ct3, item_ct1);
            });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:230: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
}

void RenderEngine::render(Grid **tempGrid) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

    // write temp to 3d texture
/* DPCT_ORIG     dim3 blockSize(8,8,8);*/
    sycl::range<3> blockSize(8, 8, 8);
/* DPCT_ORIG     dim3 gridSize(mWidth/blockSize.x, mHeight/blockSize.y,
 * mDepth/blockSize.z);*/
    sycl::range<3> gridSize(mDepth / blockSize[0], mHeight / blockSize[1],
                            mWidth / blockSize[2]);
/* DPCT_ORIG     write_temp_to_3d_tex_kernel <<< gridSize, blockSize >>>
 * (tempGrid, mTempVol->volumeSurf);*/
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto mTempVol_volumeSurf_ct1 = mTempVol->volumeSurf;

        cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<3> item_ct1) {
                             write_temp_to_3d_tex_kernel(
                                 tempGrid, mTempVol_volumeSurf_ct1, item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:231: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:232: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:233: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->scatter_in = clock();

    if (mScatterScale > 0.0f) {
/* DPCT_ORIG         dim3 gridSizeScatter((mWidth/mScatterResRatio)
   /blockSize.x+1, (mHeight/mScatterResRatio)/blockSize.y+1,
                             (mDepth/mScatterResRatio) /blockSize.z+1 );*/
        sycl::range<3> gridSizeScatter(
            (mDepth / mScatterResRatio) / blockSize[0] + 1,
            (mHeight / mScatterResRatio) / blockSize[1] + 1,
            (mWidth / mScatterResRatio) / blockSize[2] + 1);
/* DPCT_ORIG         write_scatter_to_3d_tex_kernel <<< gridSizeScatter,
   blockSize >>> (tempGrid, mScatterFrontVol->volumeSurf, mScatterTempMin,
                                                                           mScatterResRatio,
                                                                           mScatterFrontVol->size);*/
        /*
        DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto mScatterFrontVol_volumeSurf_ct1 = mScatterFrontVol->volumeSurf;
            auto mScatterTempMin_ct2 = mScatterTempMin;
            auto mScatterResRatio_ct3 = mScatterResRatio;
            auto mScatterFrontVol_size_ct4 = mScatterFrontVol->size;

            cgh.parallel_for(
                sycl::nd_range<3>(gridSizeScatter * blockSize, blockSize),
                [=](sycl::nd_item<3> item_ct1) {
                    write_scatter_to_3d_tex_kernel(
                        tempGrid, mScatterFrontVol_volumeSurf_ct1,
                        mScatterTempMin_ct2, mScatterResRatio_ct3,
                        mScatterFrontVol_size_ct4, item_ct1);
                });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:234: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:235: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));

        Volume *swap = 0;
        for (int i=0; i<mScatterBlurIter; i++) {
/* DPCT_ORIG             gaussian_blur_texture_kernel <<< gridSizeScatter,
   blockSize >>> (mScatterFrontVol->volumeTex, mScatterBackVol->volumeSurf,
                                                                             mFilterSize,
   mScatterFrontVol->size);*/
            /*
            DPCT1049:6: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                c_filterData.init();

                auto c_filterData_ptr_ct1 = c_filterData.get_ptr();

                auto mScatterFrontVol_volumeTex_ct0 =
                    mScatterFrontVol->volumeTex;
                auto mScatterBackVol_volumeSurf_ct1 =
                    mScatterBackVol->volumeSurf;
                auto mFilterSize_ct2 = mFilterSize;
                auto mScatterFrontVol_size_ct3 = mScatterFrontVol->size;

                cgh.parallel_for(
                    sycl::nd_range<3>(gridSizeScatter * blockSize, blockSize),
                    [=](sycl::nd_item<3> item_ct1) {
                        gaussian_blur_texture_kernel(
                            mScatterFrontVol_volumeTex_ct0,
                            mScatterBackVol_volumeSurf_ct1, mFilterSize_ct2,
                            mScatterFrontVol_size_ct3, item_ct1,
                            c_filterData_ptr_ct1);
                    });
            });
/* DPCT_ORIG             checkCudaErrors(cudaGetLastError());*/
            /*
            DPCT1010:236: SYCL uses exceptions to report errors and does not use
            the error codes. The call was replaced with 0. You need to rewrite
            this code.
            */
            checkCudaErrors(0);
/* DPCT_ORIG             checkCudaErrors(cudaDeviceSynchronize());*/
            /*
            DPCT1003:237: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_current_device().queues_wait_and_throw(), 0));

            // swap textures for iterative blur
            swap = mScatterFrontVol;
            mScatterFrontVol = mScatterBackVol;
            mScatterBackVol = swap;
        }
    }

    /*
    DPCT1008:238: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->scatter_out = clock();

    /*
    DPCT1008:239: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->render_in = clock();

    // rendered image should be multiple of 8
/* DPCT_ORIG     dim3 block(8, 8, 1);*/
    sycl::range<3> block(1, 8, 8);
/* DPCT_ORIG     dim3 grid(mRdrWidth/block.x, mRdrHeight/block.y, 1);*/
    sycl::range<3> grid(1, mRdrHeight / block[1], mRdrWidth / block[2]);
/* DPCT_ORIG     render_kernel <<< grid, block >>> (mFrameBuffer, tempGrid,
   mTempVol->volumeTex, mScatterFrontVol->volumeTex, d_mShader, d_mLights,
   mLightCount, d_mCamera, mDx, mScatterResRatio, mPStep, mSStep, mCutoff);*/
    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto mFrameBuffer_ct0 = mFrameBuffer;
        auto mTempVol_volumeTex_ct2 = mTempVol->volumeTex;
        auto mScatterFrontVol_volumeTex_ct3 = mScatterFrontVol->volumeTex;
        auto d_mShader_ct4 = d_mShader;
        auto d_mLights_ct5 = d_mLights;
        auto mLightCount_ct6 = mLightCount;
        auto d_mCamera_ct7 = d_mCamera;
        auto mDx_ct8 = mDx;
        auto mScatterResRatio_ct9 = mScatterResRatio;
        auto mPStep_ct10 = mPStep;
        auto mSStep_ct11 = mSStep;
        auto mCutoff_ct12 = mCutoff;

        auto mTempVol_volumeTex = dpcx::ImageToAccessorExt<float, 3>(mTempVol_volumeTex_ct2, cgh);
        auto mScatterFrontVol_volumeTex = dpcx::ImageToAccessorExt<float, 3>(mScatterFrontVol_volumeTex_ct3, cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             render_kernel(
                                 mFrameBuffer_ct0, tempGrid,
                                 mTempVol_volumeTex,
                                 mScatterFrontVol_volumeTex, d_mShader_ct4,
                                 d_mLights_ct5, mLightCount_ct6, d_mCamera_ct7,
                                 mDx_ct8, mScatterResRatio_ct9, mPStep_ct10,
                                 mSStep_ct11, mCutoff_ct12, item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:240: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:241: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:242: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->render_out = clock();
}

void RenderEngine::writeToDisk() {

    /*
    DPCT1008:243: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->writeToDisk_in = clock();

    char filename[256];
    std::string stringRenderFile(mRenderFile);
    std::string subStrName = stringRenderFile.substr(0, stringRenderFile.length()-9);
    std::string subStrExt = stringRenderFile.substr(stringRenderFile.length()-3, stringRenderFile.length());
    sprintf(filename, "%s/%s_%04d.%s", mSceneDir.c_str(), subStrName.c_str(), mTmr->iter, subStrExt.c_str());

    std::string renderDir(Parser::getDirPath(filename));
    mkDir775(renderDir.c_str());

    if (subStrExt == "ppm") {

        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
        ofs << "P3\n" << mRdrWidth << " " << mRdrHeight << "\n255\n";
        for (int i=0; i<mRdrWidth*mRdrHeight; i++) {
/* DPCT_ORIG             int ir = int(255.99*mFrameBuffer[i].x);*/
            int ir = int(255.99 * mFrameBuffer[i].x());
/* DPCT_ORIG             int ig = int(255.99*mFrameBuffer[i].y);*/
            int ig = int(255.99 * mFrameBuffer[i].y());
/* DPCT_ORIG             int ib = int(255.99*mFrameBuffer[i].z);*/
            int ib = int(255.99 * mFrameBuffer[i].z());
            ofs << ir << " " << ig << " " << ib << "\n";
        }
        ofs.close();

    } else if (subStrExt == "png") {

        unsigned char *rgba = new unsigned char[mRdrWidth*mRdrHeight*4];

        for (int i=0; i<mRdrWidth*mRdrHeight; i++) {
/* DPCT_ORIG             int valuer = (int) ((mFrameBuffer[i].x)*255.0);*/
            int valuer = (int)((mFrameBuffer[i].x()) * 255.0);
            valuer = std::max( std::min(valuer, 255), 0 );
/* DPCT_ORIG             int valueg = (int) ((mFrameBuffer[i].y)*255.0);*/
            int valueg = (int)((mFrameBuffer[i].y()) * 255.0);
            valueg = std::max( std::min(valueg, 255), 0 );
/* DPCT_ORIG             int valueb = (int) ((mFrameBuffer[i].z)*255.0);*/
            int valueb = (int)((mFrameBuffer[i].z()) * 255.0);
            valueb = std::max( std::min(valueb, 255), 0 );

            rgba[i*4 + 0] = valuer;
            rgba[i*4 + 1] = valueg;
            rgba[i*4 + 2] = valueb;
            rgba[i*4 + 3] = 255;
        }

        lodepng_encode32_file(filename, rgba, mRdrWidth, mRdrHeight);

        delete rgba;

    } else if (subStrExt == "exr") {

        EXRHeader header;
        InitEXRHeader(&header);

        header.compression_type = TINYEXR_COMPRESSIONTYPE_RLE;

        EXRImage image;
        InitEXRImage(&image);

        image.num_channels = 3;

        std::vector<float> images[3];
        for (int i = 0; i < image.num_channels; i++) images[i].resize(mRdrWidth*mRdrHeight);

        for (int i = 0; i < mRdrHeight; i++) {
          for (int j = 0; j < mRdrWidth; j++) {

              int idx = i * mRdrWidth + j;

/* DPCT_ORIG               images[0][idx] = mFrameBuffer[idx].x;*/
              images[0][idx] = mFrameBuffer[idx].x();
/* DPCT_ORIG               images[1][idx] = mFrameBuffer[idx].y;*/
              images[1][idx] = mFrameBuffer[idx].y();
/* DPCT_ORIG               images[2][idx] = mFrameBuffer[idx].z;*/
              images[2][idx] = mFrameBuffer[idx].z();
          }
        }

        float* image_ptr[3];

        image_ptr[0] = &(images[2].at(0)); // B
        image_ptr[1] = &(images[1].at(0)); // G
        image_ptr[2] = &(images[0].at(0)); // R

        image.images = (unsigned char**)image_ptr;
        image.width = mRdrWidth;
        image.height = mRdrHeight;

        header.num_channels = 3;
        header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
        // Must be (A)BGR order, since most of EXR viewers expect this channel order.
        strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
        strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
        strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

        header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        for (int i = 0; i < header.num_channels; i++) {
          header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
          // for some reason HALF does not work here, so we use FLOAT
          header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
        }

        const char* err = NULL; // or nullptr in C++11 or later.
        int ret = SaveEXRImageToFile(&image, &header, filename, &err);
        assert(ret == TINYEXR_SUCCESS);

        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);

    }

    /*
    DPCT1008:244: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->writeToDisk_out = clock();
}
