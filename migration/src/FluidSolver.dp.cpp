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

/* DPCT_ORIG #include "../thirdparty/cuda-noise/cuda_noise.cuh"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../thirdparty/cuda-noise/cuda_noise.dp.hpp"
#include "dpcx/dpcx.hpp"

#include "FluidSolver.h"
#include <time.h>

/* DPCT_ORIG __global__ void create_grids_kernel(Grid **d_mT, Grid **d_mU, Grid
   **d_mV, Grid **d_mW, float *d_mTFront, float *d_mTBack, float *d_mUFront,
   float *d_mUBack, float *d_mVFront, float *d_mVBack, float *d_mWFront, float
   *d_mWBack, int width, int height, int depth, float dx, SceneSettings scn)*/
void create_grids_kernel(Grid *d_mT, Grid *d_mU, Grid *d_mV, Grid *d_mW,
                         float *d_mTFront, float *d_mTBack, float *d_mUFront,
                         float *d_mUBack, float *d_mVFront, float *d_mVBack,
                         float *d_mWFront, float *d_mWBack, int width,
                         int height, int depth, float dx, SceneSettings scn)
{
/* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
    dpcx::init_device_mem(d_mT, new Grid(d_mTFront, d_mTBack, width, height, depth, 0.5f, 0.5f, 0.5f, dx,
                        scn.domainBboxMin, scn.domainBboxMax, false, scn));
    dpcx::init_device_mem(d_mU, new Grid(d_mUFront, d_mUBack, width+1, height, depth, 0.0f, 0.5f, 0.5f, dx,
                        scn.domainBboxMin, scn.domainBboxMax, false, scn));
    dpcx::init_device_mem(d_mV, new Grid(d_mVFront, d_mVBack, width, height+1, depth, 0.5f, 0.0f, 0.5f, dx,
                        scn.domainBboxMin, scn.domainBboxMax, true,  scn));
    dpcx::init_device_mem(d_mW, new Grid(d_mWFront, d_mWBack, width, height, depth+1, 0.5f, 0.5f, 0.0f, dx,
                        scn.domainBboxMin, scn.domainBboxMax, false, scn));
}

/* DPCT_ORIG __global__ void free_grids_kernel(Grid **d_mT, Grid **d_mU, Grid
 * **d_mV, Grid **d_mW)*/
void free_grids_kernel(Grid *d_mT, Grid *d_mU, Grid *d_mV, Grid *d_mW)
{
    dpcx::free(d_mT);
    dpcx::free(d_mU);
    dpcx::free(d_mV);
    dpcx::free(d_mW);
}

/* DPCT_ORIG __global__ void clear_back_buffer_kernel(Grid **d_mT, Grid **d_mU,
 * Grid **d_mV, Grid **d_mW) {*/
void clear_back_buffer_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                              Grid **d_mW, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*d_mT)->mWidth && j < (*d_mT)->mHeight && k < (*d_mT)->mDepth) {
        (*d_mT)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->atBack(i,j,k) = 0.0f;
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->atBack(i,j,k) = 0.0f;
    }
}

/* DPCT_ORIG __global__ void add_source_to_back_buffer_kernel(Grid **d_mT, Grid
   **d_mU, Grid **d_mV, Grid **d_mW, float3 *d_mPartPos, float3 *d_mPartVel,
                                  float *d_mPartPscale, float *d_mPartTemp,
   float dx, int pointCount)*/
void add_source_to_back_buffer_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                                      Grid **d_mW, sycl::float3 *d_mPartPos,
                                      sycl::float3 *d_mPartVel,
                                      float *d_mPartPscale, float *d_mPartTemp,
                                      float dx, int pointCount,
                                      const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int idx = threadIdx.x + blockIdx.x * blockDim.x;*/
    int idx = item_ct1.get_local_id(2) +
              item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (idx >= pointCount) return;

    if (d_mPartPscale[idx] > dx/2.0f) {
/* DPCT_ORIG         if (abs(d_mPartTemp[idx]) > 1e-5)*/
        if (sycl::fabs(d_mPartTemp[idx]) > 1e-5)
/* DPCT_ORIG             (*d_mT)->sphereToGrid(d_mPartPos[idx].x,
 * d_mPartPos[idx].y, d_mPartPos[idx].z,*/
            (*d_mT)->sphereToGrid(d_mPartPos[idx].x(), d_mPartPos[idx].y(),
                                  d_mPartPos[idx].z(), d_mPartPscale[idx],
                                  d_mPartTemp[idx]);
/* DPCT_ORIG         if (abs(d_mPartVel[idx].x) > 1e-5)*/
        if (sycl::fabs(d_mPartVel[idx].x()) > 1e-5)
/* DPCT_ORIG             (*d_mU)->sphereToGrid(d_mPartPos[idx].x,
 * d_mPartPos[idx].y, d_mPartPos[idx].z,*/
            (*d_mU)->sphereToGrid(
                d_mPartPos[idx].x(), d_mPartPos[idx].y(), d_mPartPos[idx].z(),
                /* DPCT_ORIG d_mPartPscale[idx], d_mPartVel[idx].x);*/
                d_mPartPscale[idx], d_mPartVel[idx].x());
/* DPCT_ORIG         if (abs(d_mPartVel[idx].y) > 1e-5)*/
        if (sycl::fabs(d_mPartVel[idx].y()) > 1e-5)
/* DPCT_ORIG             (*d_mV)->sphereToGrid(d_mPartPos[idx].x,
 * d_mPartPos[idx].y, d_mPartPos[idx].z,*/
            (*d_mV)->sphereToGrid(
                d_mPartPos[idx].x(), d_mPartPos[idx].y(), d_mPartPos[idx].z(),
                /* DPCT_ORIG d_mPartPscale[idx], d_mPartVel[idx].y);*/
                d_mPartPscale[idx], d_mPartVel[idx].y());
/* DPCT_ORIG         if (abs(d_mPartVel[idx].z) > 1e-5)*/
        if (sycl::fabs(d_mPartVel[idx].z()) > 1e-5)
/* DPCT_ORIG             (*d_mW)->sphereToGrid(d_mPartPos[idx].x,
 * d_mPartPos[idx].y, d_mPartPos[idx].z,*/
            (*d_mW)->sphereToGrid(
                d_mPartPos[idx].x(), d_mPartPos[idx].y(), d_mPartPos[idx].z(),
                /* DPCT_ORIG d_mPartPscale[idx], d_mPartVel[idx].z);*/
                d_mPartPscale[idx], d_mPartVel[idx].z());
    }
}

/* DPCT_ORIG __global__ void set_source_from_back_to_front_kernel(Grid **d_mT,
 * Grid **d_mU, Grid **d_mV, Grid **d_mW) {*/
void set_source_from_back_to_front_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                                          Grid **d_mW,
                                          const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

/* DPCT_ORIG     if (i < (*d_mT)->mWidth && j < (*d_mT)->mHeight && k <
 * (*d_mT)->mDepth && abs((*d_mT)->atBack(i,j,k)) > 1e-5) {*/
    if (i < (*d_mT)->mWidth && j < (*d_mT)->mHeight && k < (*d_mT)->mDepth &&
        sycl::fabs((*d_mT)->atBack(i, j, k)) > 1e-5) {
            (*d_mT)->at(i,j,k) = (*d_mT)->atBack(i,j,k);
    }
/* DPCT_ORIG     if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k <
 * (*d_mU)->mDepth && abs((*d_mU)->atBack(i,j,k)) > 1e-5) {*/
    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth &&
        sycl::fabs((*d_mU)->atBack(i, j, k)) > 1e-5) {
            (*d_mU)->at(i,j,k) = (*d_mU)->atBack(i,j,k);
    }
/* DPCT_ORIG     if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k <
 * (*d_mV)->mDepth && abs((*d_mV)->atBack(i,j,k)) > 1e-5) {*/
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth &&
        sycl::fabs((*d_mV)->atBack(i, j, k)) > 1e-5) {
            (*d_mV)->at(i,j,k) = (*d_mV)->atBack(i,j,k);
    }
/* DPCT_ORIG     if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k <
 * (*d_mW)->mDepth && abs((*d_mW)->atBack(i,j,k)) > 1e-5) {*/
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth &&
        sycl::fabs((*d_mW)->atBack(i, j, k)) > 1e-5) {
            (*d_mW)->at(i,j,k) = (*d_mW)->atBack(i,j,k);
    }
}

/* DPCT_ORIG __global__ void temperature_cooldown_kernel(Grid **d_mT, float
 * coolingRate, float dt) {*/
void temperature_cooldown_kernel(Grid **d_mT, float coolingRate, float dt,
                                 const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    (*d_mT)->at(i,j,k) *= (1.0f - dt*coolingRate);
}

/* DPCT_ORIG __global__ void apply_drag_kernel(Grid **d_mU, Grid **d_mV, Grid
 * **d_mW, float dragRate, float dt) {*/
void apply_drag_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, float dragRate,
                       float dt, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->at(i,j,k) *= (1.0f - dt*dragRate);
    }
}

/* DPCT_ORIG __global__ void add_buoyancy_kernel(Grid **d_mT, Grid **d_mV, float
 * beta, float gravity, float dt) {*/
void add_buoyancy_kernel(Grid **d_mT, Grid **d_mV, float beta, float gravity,
                         float dt, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
/* DPCT_ORIG         float3 pos = (*d_mV)->gridToObj(i,j,k);*/
        sycl::float3 pos = (*d_mV)->gridToObj(i, j, k);

        float temp = (*d_mT)->sampleO(pos, linear);

        float buoyancyForce = beta*temp*gravity; // gravity is negative

        (*d_mV)->at(i,j,k) += dt*buoyancyForce;
    }
}

/* DPCT_ORIG __global__ void copy_velocity_to_back_buffer_kernel(Grid **d_mU,
 * Grid **d_mV, Grid **d_mW) {*/
void copy_velocity_to_back_buffer_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW,
                                         const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        (*d_mU)->atBack(i,j,k) = (*d_mU)->at(i,j,k);
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        (*d_mV)->atBack(i,j,k) = (*d_mV)->at(i,j,k);
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        (*d_mW)->atBack(i,j,k) = (*d_mW)->at(i,j,k);
    }
}

/* DPCT_ORIG __device__ float3 computeVorticity(Grid **d_mT, Grid **d_mU, Grid
 * **d_mV, Grid **d_mW, int i, int j, int k, float dx) {*/
sycl::float3 computeVorticity(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                              Grid **d_mW, int i, int j, int k, float dx) {
    float x0 = (*d_mT)->iGridToObj(i-1);
    float x1 = (*d_mT)->iGridToObj(i);
    float x2 = (*d_mT)->iGridToObj(i+1);
    float y0 = (*d_mT)->jGridToObj(j-1);
    float y1 = (*d_mT)->jGridToObj(j);
    float y2 = (*d_mT)->jGridToObj(j+1);
    float z0 = (*d_mT)->kGridToObj(k-1);
    float z1 = (*d_mT)->kGridToObj(k);
    float z2 = (*d_mT)->kGridToObj(k+1);

/* DPCT_ORIG     float w_121 = (*d_mW)->sampleO(make_float3(x1, y2, z1),
 * linear);*/
    float w_121 = (*d_mW)->sampleO(sycl::float3(x1, y2, z1), linear);
/* DPCT_ORIG     float w_101 = (*d_mW)->sampleO(make_float3(x1, y0, z1),
 * linear);*/
    float w_101 = (*d_mW)->sampleO(sycl::float3(x1, y0, z1), linear);
/* DPCT_ORIG     float v_112 = (*d_mV)->sampleO(make_float3(x1, y1, z2),
 * linear);*/
    float v_112 = (*d_mV)->sampleO(sycl::float3(x1, y1, z2), linear);
/* DPCT_ORIG     float v_110 = (*d_mV)->sampleO(make_float3(x1, y1, z0),
 * linear);*/
    float v_110 = (*d_mV)->sampleO(sycl::float3(x1, y1, z0), linear);
/* DPCT_ORIG     float u_112 = (*d_mU)->sampleO(make_float3(x1, y1, z2),
 * linear);*/
    float u_112 = (*d_mU)->sampleO(sycl::float3(x1, y1, z2), linear);
/* DPCT_ORIG     float u_110 = (*d_mU)->sampleO(make_float3(x1, y1, z0),
 * linear);*/
    float u_110 = (*d_mU)->sampleO(sycl::float3(x1, y1, z0), linear);
/* DPCT_ORIG     float w_211 = (*d_mW)->sampleO(make_float3(x2, y1, z1),
 * linear);*/
    float w_211 = (*d_mW)->sampleO(sycl::float3(x2, y1, z1), linear);
/* DPCT_ORIG     float w_011 = (*d_mW)->sampleO(make_float3(x0, y1, z1),
 * linear);*/
    float w_011 = (*d_mW)->sampleO(sycl::float3(x0, y1, z1), linear);
/* DPCT_ORIG     float v_211 = (*d_mV)->sampleO(make_float3(x2, y1, z1),
 * linear);*/
    float v_211 = (*d_mV)->sampleO(sycl::float3(x2, y1, z1), linear);
/* DPCT_ORIG     float v_011 = (*d_mV)->sampleO(make_float3(x0, y1, z1),
 * linear);*/
    float v_011 = (*d_mV)->sampleO(sycl::float3(x0, y1, z1), linear);
/* DPCT_ORIG     float u_121 = (*d_mU)->sampleO(make_float3(x1, y2, z1),
 * linear);*/
    float u_121 = (*d_mU)->sampleO(sycl::float3(x1, y2, z1), linear);
/* DPCT_ORIG     float u_101 = (*d_mU)->sampleO(make_float3(x1, y0, z1),
 * linear);*/
    float u_101 = (*d_mU)->sampleO(sycl::float3(x1, y0, z1), linear);

/* DPCT_ORIG     return make_float3(w_121 - w_101 - (v_112 - v_110),
                       u_112 - u_110 - (w_211 - w_011),
                       v_211 - v_011 - (u_121 - u_101)) / (2.0f*dx);*/
    return dpct_operator_overloading::operator/(
        sycl::float3(w_121 - w_101 - (v_112 - v_110),
                     u_112 - u_110 - (w_211 - w_011),
                     v_211 - v_011 - (u_121 - u_101)),
        (2.0f * dx));
}

/* DPCT_ORIG __global__ void vorticity_confinement_kernel(Grid **d_mT, Grid
   **d_mU, Grid **d_mV, Grid **d_mW, float conf, float dx, float dt)*/
void vorticity_confinement_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                                  Grid **d_mW, float conf, float dx, float dt,
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

    if (i <= 1 || i >= (*d_mT)->mWidth-2 || j <= 1 || j >= (*d_mT)->mHeight-2 || k <= 1 || k >= (*d_mT)->mDepth-2 )
        return;

/* DPCT_ORIG     float3 omega = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,   j,
 * k,   dx);*/
    sycl::float3 omega = computeVorticity(d_mT, d_mU, d_mV, d_mW, i, j, k, dx);

/* DPCT_ORIG     float3 omega_211 = computeVorticity(d_mT, d_mU, d_mV, d_mW,
 * i+1, j,   k,   dx);*/
    sycl::float3 omega_211 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i + 1, j, k, dx);
/* DPCT_ORIG     float3 omega_011 = computeVorticity(d_mT, d_mU, d_mV, d_mW,
 * i-1, j,   k,   dx);*/
    sycl::float3 omega_011 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i - 1, j, k, dx);
/* DPCT_ORIG     float3 omega_121 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,
 * j+1, k,   dx);*/
    sycl::float3 omega_121 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i, j + 1, k, dx);
/* DPCT_ORIG     float3 omega_101 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,
 * j-1, k,   dx);*/
    sycl::float3 omega_101 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i, j - 1, k, dx);
/* DPCT_ORIG     float3 omega_112 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,
 * j,   k+1, dx);*/
    sycl::float3 omega_112 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i, j, k + 1, dx);
/* DPCT_ORIG     float3 omega_110 = computeVorticity(d_mT, d_mU, d_mV, d_mW, i,
 * j,   k-1, dx);*/
    sycl::float3 omega_110 =
        computeVorticity(d_mT, d_mU, d_mV, d_mW, i, j, k - 1, dx);

/* DPCT_ORIG     float3 gradNormOmega = make_float3( length(omega_211) -
   length(omega_011), length(omega_121) - length(omega_101), length(omega_112) -
   length(omega_110) ) / (2.0f*dx);*/
    sycl::float3 gradNormOmega = dpct_operator_overloading::operator/(
        sycl::float3(length(omega_211) - length(omega_011),
                     length(omega_121) - length(omega_101),
                     length(omega_112) - length(omega_110)),
        (2.0f * dx));

/* DPCT_ORIG     float3 N = gradNormOmega / ( length(gradNormOmega) +
 * 1e-20f*(1.0f/(dx*dt)) );*/
    sycl::float3 N = dpct_operator_overloading::operator/(
        gradNormOmega, (length(gradNormOmega) + 1e-20f * (1.0f / (dx * dt))));

    // enforce stability by controlling the vorticity magnitude based on local speed
/* DPCT_ORIG     float3 pos = (*d_mT)->gridToObj(i, j, k);*/
    sycl::float3 pos = (*d_mT)->gridToObj(i, j, k);
/* DPCT_ORIG     float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                              (*d_mV)->sampleO(pos, linear),
                              (*d_mW)->sampleO(pos, linear) );*/
    sycl::float3 vel = sycl::float3((*d_mU)->sampleO(pos, linear),
                                    (*d_mV)->sampleO(pos, linear),
                                    (*d_mW)->sampleO(pos, linear));
    float speed = length(vel);

    float omegaNorm = length(omega) + 1e-20f*(1.0f/(dx*dt));
    if (omegaNorm > speed) {
/* DPCT_ORIG         omega = speed * omega / omegaNorm;*/
        omega = dpct_operator_overloading::operator/(
            dpct_operator_overloading::operator*(speed, omega), omegaNorm);
    }

/* DPCT_ORIG     float3 vortConf = conf * dx * cross(gradNormOmega, omega);*/
    sycl::float3 vortConf = dpct_operator_overloading::operator*(
        conf *dx, cross(gradNormOmega, omega));

/* DPCT_ORIG     atomicAdd(&((*d_mU)->atBack(i,  j  ,k  )),
 * 0.5f*dt*vortConf.x);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mU)->atBack(i, j, k)), 0.5f * dt * vortConf.x());
/* DPCT_ORIG     atomicAdd(&((*d_mU)->atBack(i+1,j  ,k  )),
 * 0.5f*dt*vortConf.x);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mU)->atBack(i + 1, j, k)), 0.5f * dt * vortConf.x());
/* DPCT_ORIG     atomicAdd(&((*d_mV)->atBack(i,  j  ,k  )),
 * 0.5f*dt*vortConf.y);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mV)->atBack(i, j, k)), 0.5f * dt * vortConf.y());
/* DPCT_ORIG     atomicAdd(&((*d_mV)->atBack(i,  j+1,k  )),
 * 0.5f*dt*vortConf.y);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mV)->atBack(i, j + 1, k)), 0.5f * dt * vortConf.y());
/* DPCT_ORIG     atomicAdd(&((*d_mW)->atBack(i,  j  ,k  )),
 * 0.5f*dt*vortConf.z);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mW)->atBack(i, j, k)), 0.5f * dt * vortConf.z());
/* DPCT_ORIG     atomicAdd(&((*d_mW)->atBack(i,  j,  k+1)),
 * 0.5f*dt*vortConf.z);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mW)->atBack(i, j, k + 1)), 0.5f * dt * vortConf.z());
}

/* DPCT_ORIG __device__ inline float fit(float v, float minIn, float maxIn,
 * float minOut, float maxOut) {*/
inline float fit(float v, float minIn, float maxIn, float minOut,
                 float maxOut) {
/* DPCT_ORIG     return fmaxf( fminf( (v-minIn)/(maxIn-minIn), 1.0f),
 * 0.0f)*(maxOut-minOut) + minOut;*/
    return sycl::fmax(sycl::fmin((v - minIn) / (maxIn - minIn), 1.0f), 0.0f) *
               (maxOut - minOut) +
           minOut;
}

/* DPCT_ORIG __device__ float computeTempMask(Grid **grid, Grid **d_mT, int i,
 * int j, int k, float2 maskTempRamp) {*/
float computeTempMask(Grid **grid, Grid **d_mT, int i, int j, int k,
                      sycl::float2 maskTempRamp) {
/* DPCT_ORIG     float3 pos = (*grid)->gridToObj(i,j,k);*/
    sycl::float3 pos = (*grid)->gridToObj(i, j, k);

/* DPCT_ORIG     return fit((*d_mT)->sampleO(pos, linear), maskTempRamp.x,
 * maskTempRamp.y, 1.0f, 0.0f);*/
    return fit((*d_mT)->sampleO(pos, linear), maskTempRamp.x(),
               maskTempRamp.y(), 1.0f, 0.0f);
}

/* DPCT_ORIG __device__ float computeVelMask(Grid **grid, Grid **d_mU, Grid
   **d_mV, Grid **d_mW, int i, int j, int k, float2 maskVelRamp) {*/
float computeVelMask(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW, int i,
                     int j, int k, sycl::float2 maskVelRamp) {
/* DPCT_ORIG     float3 pos = (*grid)->gridToObj(i,j,k);*/
    sycl::float3 pos = (*grid)->gridToObj(i, j, k);

/* DPCT_ORIG     float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                              (*d_mV)->sampleO(pos, linear),
                              (*d_mW)->sampleO(pos, linear) );*/
    sycl::float3 vel = sycl::float3((*d_mU)->sampleO(pos, linear),
                                    (*d_mV)->sampleO(pos, linear),
                                    (*d_mW)->sampleO(pos, linear));

/* DPCT_ORIG     return fit(length(vel), maskVelRamp.x, maskVelRamp.y,
 * 0.0f, 1.0f);*/
    return fit(length(vel), maskVelRamp.x(), maskVelRamp.y(), 0.0f, 1.0f);
}

/* DPCT_ORIG __global__ void add_curl_noise_kernel(Grid **d_mU, Grid **d_mV,
 * Grid **d_mW, float dx, float dt) {*/
void add_curl_noise_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, float dx,
                           float dt, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    float dudx = ( (*d_mU)->atBack(i+1, j,   k  ) - (*d_mV)->atBack(i, j, k) ) / dx;
    float dvdy = ( (*d_mV)->atBack(i,   j+1, k  ) - (*d_mV)->atBack(i, j, k) ) / dx;
    float dwdz = ( (*d_mW)->atBack(i,   j,   k+1) - (*d_mV)->atBack(i, j, k) ) / dx;

/* DPCT_ORIG     float3 curlNoise = make_float3(dvdy-dwdz, dwdz-dudx,
 * dudx-dvdy);*/
    sycl::float3 curlNoise =
        sycl::float3(dvdy - dwdz, dwdz - dudx, dudx - dvdy);

/* DPCT_ORIG     float avgSpeed = length( make_float3((*d_mU)->atBack(i, j, k) +
   (*d_mU)->atBack(i+1, j,   k  ),
                                         (*d_mV)->atBack(i, j, k) +
   (*d_mV)->atBack(i,   j+1, k  ),
                                         (*d_mW)->atBack(i, j, k) +
   (*d_mW)->atBack(i,   j,   k+1) )/2.0f );*/
    float avgSpeed = length(dpct_operator_overloading::operator/(
        sycl::float3((*d_mU)->atBack(i, j, k) + (*d_mU)->atBack(i + 1, j, k),
                     (*d_mV)->atBack(i, j, k) + (*d_mV)->atBack(i, j + 1, k),
                     (*d_mW)->atBack(i, j, k) + (*d_mW)->atBack(i, j, k + 1)),
        2.0f));
/* DPCT_ORIG     curlNoise = avgSpeed*curlNoise / (length(curlNoise)+
 * 1e-20f*(1.0f/(dx*dt)));*/
    curlNoise = dpct_operator_overloading::operator/(
        dpct_operator_overloading::operator*(avgSpeed, curlNoise),
        (length(curlNoise) + 1e-20f * (1.0f / (dx * dt))));

/* DPCT_ORIG     atomicAdd(&((*d_mU)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.x);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mU)->at(i, j, k)), 0.5f * dt * curlNoise.x());
/* DPCT_ORIG     atomicAdd(&((*d_mU)->at(i+1,j  ,k  )), 0.5f*dt*curlNoise.x);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mU)->at(i + 1, j, k)), 0.5f * dt * curlNoise.x());
/* DPCT_ORIG     atomicAdd(&((*d_mV)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.y);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mV)->at(i, j, k)), 0.5f * dt * curlNoise.y());
/* DPCT_ORIG     atomicAdd(&((*d_mV)->at(i,  j+1,k  )), 0.5f*dt*curlNoise.y);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mV)->at(i, j + 1, k)), 0.5f * dt * curlNoise.y());
/* DPCT_ORIG     atomicAdd(&((*d_mW)->at(i,  j  ,k  )), 0.5f*dt*curlNoise.z);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mW)->at(i, j, k)), 0.5f * dt * curlNoise.z());
/* DPCT_ORIG     atomicAdd(&((*d_mW)->at(i,  j,  k+1)), 0.5f*dt*curlNoise.z);*/
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &((*d_mW)->at(i, j, k + 1)), 0.5f * dt * curlNoise.z());
}

/* DPCT_ORIG __global__ void compute_turbulence_kernel(Grid **d_mT, Grid **d_mU,
   Grid **d_mV, Grid **d_mW, float amp, float scale, float2 maskTempRamp, float2
   maskVelRamp, float dt, float time)*/
void compute_turbulence_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV,
                               Grid **d_mW, float amp, float scale,
                               sycl::float2 maskTempRamp,
                               sycl::float2 maskVelRamp, float dt, float time,
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

    float freq = 1.0f / scale;
    int seed = 123;
    float avgGridSize = ((*d_mT)->mWidth + (*d_mT)->mHeight + (*d_mT)->mDepth) / 3.0f;
/* DPCT_ORIG     float offset = fmod(time*avgGridSize, 25000.0f); */
    float offset = sycl::fmod(
        time * avgGridSize, 25000.0f); // fmod to avoid floating point roundoff

    if (i < (*d_mU)->mWidth && j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) {
        float tempMaskU = computeTempMask(d_mU, d_mT, i, j, k, maskTempRamp);
        float velMaskU = computeVelMask(d_mU, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskU = tempMaskU * velMaskU;

/* DPCT_ORIG         float3 posU = make_float3((float)i+offset, (float)j,
 * (float)k);*/
        sycl::float3 posU = sycl::float3((float)i + offset, (float)j, (float)k);
        float noiseU = cudaNoise::perlinNoise(posU, freq, seed);
        (*d_mU)->atBack(i,j,k) = maskU*amp*noiseU;
    }
    if (i < (*d_mV)->mWidth && j < (*d_mV)->mHeight && k < (*d_mV)->mDepth) {
        float tempMaskV = computeTempMask(d_mV, d_mT, i, j, k, maskTempRamp);
        float velMaskV = computeVelMask(d_mV, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskV = tempMaskV * velMaskV;

/* DPCT_ORIG         float3 posV = make_float3((float)i, (float)j+offset,
 * (float)k);*/
        sycl::float3 posV = sycl::float3((float)i, (float)j + offset, (float)k);
        float noiseV = cudaNoise::perlinNoise(posV, freq, seed);
        (*d_mV)->atBack(i,j,k) = maskV*amp*noiseV;
    }
    if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight && k < (*d_mW)->mDepth) {
        float tempMaskW = computeTempMask(d_mW, d_mT, i, j, k, maskTempRamp);
        float velMaskW = computeVelMask(d_mW, d_mU, d_mV, d_mW, i, j, k, maskVelRamp);
        float maskW = tempMaskW * velMaskW;

/* DPCT_ORIG         float3 posW = make_float3((float)i, (float)j,
 * (float)k+offset);*/
        sycl::float3 posW = sycl::float3((float)i, (float)j, (float)k + offset);
        float noiseW = cudaNoise::perlinNoise(posW, freq, seed);
        (*d_mW)->atBack(i,j,k) = maskW*amp*noiseW;
    }
}

/* DPCT_ORIG __global__ void add_wind_kernel(Grid **d_mU, Grid **d_mV, Grid
   **d_mW, int windDir, float windAmp, float windSpeed, float windTurbAmp, float
   windTurbScale, float dt, float time)*/
void add_wind_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW, int windDir,
                     float windAmp, float windSpeed, float windTurbAmp,
                     float windTurbScale, float dt, float time,
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

    if (i >= (*d_mU)->mWidth || j >= (*d_mV)->mHeight || k >= (*d_mW)->mDepth) return;

    int voxelPadding = 2;

    if (windDir <= 1) { // wind along X axis
        if (i >= voxelPadding && i < (*d_mU)->mWidth-voxelPadding) return;

        float freq = 1.0f / windTurbScale;
/* DPCT_ORIG         float avgDomSize = ((*d_mU)->mDomSize.x +
 * (*d_mU)->mDomSize.y + (*d_mU)->mDomSize.z) / 3.0f;*/
        float avgDomSize = ((*d_mU)->mDomSize.x() + (*d_mU)->mDomSize.y() +
                            (*d_mU)->mDomSize.z()) /
                           3.0f;
/* DPCT_ORIG         float offset = fmod(time*avgDomSize*windSpeed, 25000.0f);
 */
        float offset =
            sycl::fmod(time * avgDomSize * windSpeed,
                       25000.0f); // fmod to avoid floating point roundoff

/* DPCT_ORIG         float3 pos = make_float3((*d_mU)->iGridToObj(i),
 * (*d_mV)->jGridToObj(j), (*d_mW)->kGridToObj(k)+offset);*/
        sycl::float3 pos =
            sycl::float3((*d_mU)->iGridToObj(i), (*d_mV)->jGridToObj(j),
                         (*d_mW)->kGridToObj(k) + offset);
        float noiseU = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 123);
        float noiseV = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 456);
        float noiseW = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 789);
        float windDirVel = windDir == 0 ? 1.0f : -1.0f;

        // blend in wind vel over 1 second
        if (j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) // U
            (*d_mU)->at(i,j,k) = (*d_mU)->at(i,j,k)*(1.0f-dt) + (windAmp*(windDirVel+noiseU))*dt;
        if (i < (*d_mV)->mWidth && k < (*d_mV)->mDepth)  // V
            (*d_mV)->at(i,j,k) = (*d_mV)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseV)*dt;
        if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight) // W
            (*d_mW)->at(i,j,k) = (*d_mW)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseW)*dt;
    } else {            // wind along Z axis
        if (k >= voxelPadding && k < (*d_mW)->mDepth-voxelPadding) return;

        float freq = 1.0f / windTurbScale;
/* DPCT_ORIG         float avgDomSize = ((*d_mU)->mDomSize.x +
 * (*d_mU)->mDomSize.y + (*d_mU)->mDomSize.z) / 3.0f;*/
        float avgDomSize = ((*d_mU)->mDomSize.x() + (*d_mU)->mDomSize.y() +
                            (*d_mU)->mDomSize.z()) /
                           3.0f;
/* DPCT_ORIG         float offset = fmod(time*avgDomSize*windSpeed, 25000.0f);
 */
        float offset =
            sycl::fmod(time * avgDomSize * windSpeed,
                       25000.0f); // fmod to avoid floating point roundoff

/* DPCT_ORIG         float3 pos = make_float3((*d_mU)->iGridToObj(i),
 * (*d_mV)->jGridToObj(j), (*d_mW)->kGridToObj(k)+offset);*/
        sycl::float3 pos =
            sycl::float3((*d_mU)->iGridToObj(i), (*d_mV)->jGridToObj(j),
                         (*d_mW)->kGridToObj(k) + offset);
        float noiseU = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 123);
        float noiseV = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 456);
        float noiseW = windTurbAmp * cudaNoise::perlinNoise(pos, freq, 789);
        float windDirVel = windDir == 2 ? 1.0f : -1.0f;

        // blend in wind vel over 1 second
        if (j < (*d_mU)->mHeight && k < (*d_mU)->mDepth) // U
            (*d_mU)->at(i,j,k) = (*d_mU)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseU)*dt;
        if (i < (*d_mV)->mWidth && k < (*d_mV)->mDepth)  // V
            (*d_mV)->at(i,j,k) = (*d_mV)->at(i,j,k)*(1.0f-dt) + (windAmp*noiseV)*dt;
        if (i < (*d_mW)->mWidth && j < (*d_mW)->mHeight) // W
            (*d_mW)->at(i,j,k) = (*d_mW)->at(i,j,k)*(1.0f-dt) + (windAmp*(windDirVel+noiseW))*dt;
    }
}

/* DPCT_ORIG __global__ void advect_forward_euler_kernel(Grid **grid, Grid
 * **d_mU, Grid **d_mV, Grid **d_mW, float dt, bool clamp) {*/
void advect_forward_euler_kernel(Grid **grid, Grid **d_mU, Grid **d_mV,
                                 Grid **d_mW, float dt, bool clamp,
                                 const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
/* DPCT_ORIG         float3 pos = (*grid)->gridToObj(i,j,k);*/
        sycl::float3 pos = (*grid)->gridToObj(i, j, k);
/* DPCT_ORIG         float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                                  (*d_mV)->sampleO(pos, linear),
                                  (*d_mW)->sampleO(pos, linear) );*/
        sycl::float3 vel = sycl::float3((*d_mU)->sampleO(pos, linear),
                                        (*d_mV)->sampleO(pos, linear),
                                        (*d_mW)->sampleO(pos, linear));

/* DPCT_ORIG         pos -= dt*vel;*/
        dpct_operator_overloading::operator-=(
            pos, dpct_operator_overloading::operator*(dt, vel));

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

/* DPCT_ORIG __global__ void advect_RK2_kernel(Grid **grid, Grid **d_mU, Grid
 * **d_mV, Grid **d_mW, float dt, bool clamp) {*/
void advect_RK2_kernel(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                       float dt, bool clamp, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
/* DPCT_ORIG         float3 pos = (*grid)->gridToObj(i,j,k);*/
        sycl::float3 pos = (*grid)->gridToObj(i, j, k);
/* DPCT_ORIG         float3 vel = make_float3( (*d_mU)->sampleO(pos, linear),
                                  (*d_mV)->sampleO(pos, linear),
                                  (*d_mW)->sampleO(pos, linear) );*/
        sycl::float3 vel = sycl::float3((*d_mU)->sampleO(pos, linear),
                                        (*d_mV)->sampleO(pos, linear),
                                        (*d_mW)->sampleO(pos, linear));

/* DPCT_ORIG         float3 posMid = pos - 0.5f*dt*vel;*/
        sycl::float3 posMid = dpct_operator_overloading::operator-(
            pos, dpct_operator_overloading::operator*(0.5f * dt, vel));
/* DPCT_ORIG         float3 velMid = make_float3( (*d_mU)->sampleO(posMid,
   linear),
                                     (*d_mV)->sampleO(posMid, linear),
                                     (*d_mW)->sampleO(posMid, linear) );*/
        sycl::float3 velMid = sycl::float3((*d_mU)->sampleO(posMid, linear),
                                           (*d_mV)->sampleO(posMid, linear),
                                           (*d_mW)->sampleO(posMid, linear));

/* DPCT_ORIG         pos -= dt*velMid;*/
        dpct_operator_overloading::operator-=(
            pos, dpct_operator_overloading::operator*(dt, velMid));

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

/* DPCT_ORIG __global__ void advect_RK3_kernel(Grid **grid, Grid **d_mU, Grid
 * **d_mV, Grid **d_mW, float dt, bool clamp) {*/
void advect_RK3_kernel(Grid **grid, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                       float dt, bool clamp, const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);
/* DPCT_ORIG     int k = threadIdx.z + blockIdx.z * blockDim.z;*/
    int k = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);

    if (i < (*grid)->mWidth && j < (*grid)->mHeight && k < (*grid)->mDepth) {
/* DPCT_ORIG         float3 pos = (*grid)->gridToObj(i,j,k);*/
        sycl::float3 pos = (*grid)->gridToObj(i, j, k);
/* DPCT_ORIG         float3 k1 = make_float3( (*d_mU)->sampleO(pos, linear),
                                 (*d_mV)->sampleO(pos, linear),
                                 (*d_mW)->sampleO(pos, linear) );*/
        sycl::float3 k1 = sycl::float3((*d_mU)->sampleO(pos, linear),
                                       (*d_mV)->sampleO(pos, linear),
                                       (*d_mW)->sampleO(pos, linear));

/* DPCT_ORIG         float3 pos1 = pos - 0.5f*dt*k1;*/
        sycl::float3 pos1 = dpct_operator_overloading::operator-(
            pos, dpct_operator_overloading::operator*(0.5f * dt, k1));
/* DPCT_ORIG         float3 k2 = make_float3( (*d_mU)->sampleO(pos1, linear),
                                 (*d_mV)->sampleO(pos1, linear),
                                 (*d_mW)->sampleO(pos1, linear) );*/
        sycl::float3 k2 = sycl::float3((*d_mU)->sampleO(pos1, linear),
                                       (*d_mV)->sampleO(pos1, linear),
                                       (*d_mW)->sampleO(pos1, linear));

/* DPCT_ORIG         float3 pos2 = pos - 0.75f*dt*k2;*/
        sycl::float3 pos2 = dpct_operator_overloading::operator-(
            pos, dpct_operator_overloading::operator*(0.75f * dt, k2));
/* DPCT_ORIG         float3 k3 = make_float3( (*d_mU)->sampleO(pos2, linear),
                                 (*d_mV)->sampleO(pos2, linear),
                                 (*d_mW)->sampleO(pos2, linear) );*/
        sycl::float3 k3 = sycl::float3((*d_mU)->sampleO(pos2, linear),
                                       (*d_mV)->sampleO(pos2, linear),
                                       (*d_mW)->sampleO(pos2, linear));

/* DPCT_ORIG         pos -= dt * ( (2.0f/9.0f)*k1 + (3.0f/9.0f)*k2 +
 * (4.0f/9.0f)*k3 );*/
        dpct_operator_overloading::operator-=(
            pos,
            dpct_operator_overloading::operator*(
                dt,
                (dpct_operator_overloading::operator+(
                    dpct_operator_overloading::operator+(
                        dpct_operator_overloading::operator*((2.0f / 9.0f), k1),
                        dpct_operator_overloading::operator*((3.0f / 9.0f),
                                                             k2)),
                    dpct_operator_overloading::operator*((4.0f / 9.0f), k3)))));

        (*grid)->atBack(i,j,k) = (*grid)->sampleO(pos, cubic, clamp);
    }
}

/* DPCT_ORIG __global__ void swap_grids_kernel(Grid **d_mT, Grid **d_mU, Grid
 * **d_mV, Grid **d_mW)  {*/
void swap_grids_kernel(Grid **d_mT, Grid **d_mU, Grid **d_mV, Grid **d_mW,
                       const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
        (*d_mT)->swap();
        (*d_mU)->swap();
        (*d_mV)->swap();
        (*d_mW)->swap();
    }
}

/* DPCT_ORIG __global__ void swap_vel_grids_kernel(Grid **d_mU, Grid **d_mV,
 * Grid **d_mW)  {*/
void swap_vel_grids_kernel(Grid **d_mU, Grid **d_mV, Grid **d_mW,
                           const sycl::nd_item<3> &item_ct1) {
/* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
    if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
        (*d_mU)->swap();
        (*d_mV)->swap();
        (*d_mW)->swap();
    }
}

/* DPCT_ORIG __global__ void compute_divergence_kernel(float *d_mRhs, Grid
   **d_mU, Grid **d_mV, Grid **d_mW, int width, int height, int depth, float
   density, float dx)*/
void compute_divergence_kernel(float *d_mRhs, Grid **d_mU, Grid **d_mV,
                               Grid **d_mW, int width, int height, int depth,
                               float density, float dx,
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

//    assert(i < width);
//    assert(j < height);
//    assert(k < depth);

    int idx = i + j*width + k*width*height;
    float scale = 1.0f / dx;

    d_mRhs[idx] = -scale*( (*d_mU)->at(i+1,j,k) - (*d_mU)->at(i,j,k)
                         + (*d_mV)->at(i,j+1,k) - (*d_mV)->at(i,j,k)
                         + (*d_mW)->at(i,j,k+1) - (*d_mW)->at(i,j,k));

    // handle boundary condition (closed or opened bounds)
    if (i == 0) {               // xMin
        if ((*d_mU)->mClosedBounds[0])
            d_mRhs[idx] -= scale*(*d_mU)->at(i,j,k);
    } else if (i == width-1) {  // xMax
        if ((*d_mU)->mClosedBounds[1])
            d_mRhs[idx] += scale*(*d_mU)->at(i+1,j,k);
    }
    if (j == 0) {               // yMax (don't forget the y is flipped)
        if ((*d_mV)->mClosedBounds[3])
            d_mRhs[idx] -= scale*(*d_mV)->at(i,j,k);
    } else if (j == height-1) { // yMin (don't forget the y is flipped)
        if ((*d_mV)->mClosedBounds[2])
            d_mRhs[idx] += scale*(*d_mV)->at(i,j+1,k);
    }
    if (k == 0) {               // zMin
        if ((*d_mW)->mClosedBounds[4])
            d_mRhs[idx] -= scale*(*d_mW)->at(i,j,k);
    } else if (k == depth-1) {  // zMax
        if ((*d_mW)->mClosedBounds[5])
            d_mRhs[idx] += scale*(*d_mW)->at(i,j,k+1);
    }
}

/* DPCT_ORIG __global__ void gs_solve_kernel(float* rhs, float* p, float
   density, float dx, float dt, int width, int height, int depth, int maxIter)*/
void gs_solve_kernel(float *rhs, float *p, float density, float dx, float dt,
                     int width, int height, int depth, int maxIter,
                     const sycl::nd_item<3> &item_ct1)
{
/* DPCT_ORIG     int i = threadIdx.x + blockIdx.x * blockDim.x;*/
    int i = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range(2);
/* DPCT_ORIG     int j = threadIdx.y + blockIdx.y * blockDim.y;*/
    int j = item_ct1.get_local_id(1) +
            item_ct1.get_group(1) * item_ct1.get_local_range(1);

//    assert(i < width);
//    assert(j < height);

    for (int l=0; l<maxIter; l++) {
        for (int k=0; k<depth; k++) {

            int idx = i + j*width + k*width*height;
            float scale = (density*dx*dx)/dt; // Bridson p.75

            if( (i + j)%2 == 0 ) {
                float denom = 0.0f;
                float num = scale*rhs[idx];

                if (i > 0) {
                    num += p[idx - 1];
                    denom += 1.0f;
                }
                if (i < width-1) {
                    num += p[idx + 1];
                    denom += 1.0f;
                }
                if (j > 0) {
                    num += p[idx - width];
                    denom += 1.0f;
                }
                if (j < height-1) {
                    num += p[idx + width];
                    denom += 1.0f;
                }
                if (k > 0) {
                    num += p[idx - width*height];
                    denom += 1.0f;
                }
                if (k < depth-1) {
                    num += p[idx + width*height];
                    denom += 1.0f;
                }

                p[idx] = num / denom;
            }

/* DPCT_ORIG             __syncthreads();*/
            /*
            DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            if( (i + j)%2 != 0 ) {
                float denom = 0.0f;
                float num = scale*rhs[idx];

                if (i > 0) {
                    num += p[idx - 1];
                    denom += 1.0f;
                }
                if (i < width-1) {
                    num += p[idx + 1];
                    denom += 1.0f;
                }
                if (j > 0) {
                    num += p[idx - width];
                    denom += 1.0f;
                }
                if (j < height-1) {
                    num += p[idx + width];
                    denom += 1.0f;
                }
                if (k > 0) {
                    num += p[idx - width*height];
                    denom += 1.0f;
                }
                if (k < depth-1) {
                    num += p[idx + width*height];
                    denom += 1.0f;
                }

                p[idx] = num / denom;
            }
        }
    }
}

/* DPCT_ORIG __global__ void pressure_gradient_update_kernel(float *d_mP, Grid
   **d_mU, Grid **d_mV, Grid **d_mW, int width, int height, int depth, float
   density, float dx, float dt)*/
void pressure_gradient_update_kernel(float *d_mP, Grid **d_mU, Grid **d_mV,
                                     Grid **d_mW, int width, int height,
                                     int depth, float density, float dx,
                                     float dt, const sycl::nd_item<3> &item_ct1)
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

    if (i >= width+1 || j >= height+1 || k >= depth+1) return;

    int idx = i + j*width + k*width*height;
    float scale = dt / (density*dx);

    if (i < width && j < height && k < depth) {
        if (i > 0 ) {
            (*d_mU)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - 1]);
        }
        if (j > 0 ) {
            (*d_mV)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - width]);
        }
        if (k > 0 ) {
            (*d_mW)->at(i, j, k) -= scale*(d_mP[idx] - d_mP[idx - width*height]);
        }
    }

    // set the boundary velocities (closed or opened bounds)
    // The lines marked with a "?" indicates that it should be deleted according
    // to p.71 of Bridson's book. Our tests shows it is needed, though...
    // Free surface implementation might be containing a bug.
    if ( (i == 0 || i == width) && j < height && k < depth) { // X
        if (i == 0) { // xMin
            if ((*d_mU)->mClosedBounds[0])
                (*d_mU)->at(i, j, k) = 0.0f;
            else
                (*d_mU)->at(i, j, k) = (*d_mU)->at(i+1, j, k); // ?
        } else {      // xMax
             if ((*d_mU)->mClosedBounds[1])
                 (*d_mU)->at(i, j, k) = 0.0f;
             else
                 (*d_mU)->at(i, j, k) = (*d_mU)->at(i-1, j, k); // ?
        }
    }
    if ( (j == 0 || j == height) && i < width && k < depth) { // Y
        if (j == 0) { // yMax (don't forget the y is flipped)
            if ((*d_mV)->mClosedBounds[3])
                (*d_mV)->at(i, j, k) = 0.0f;
            else
                (*d_mV)->at(i, j, k) = (*d_mV)->at(i, j+1, k); // ?
        } else {      // yMin (don't forget the y is flipped)
             if ((*d_mV)->mClosedBounds[2])
                 (*d_mV)->at(i, j, k) = 0.0f;
             else
                 (*d_mV)->at(i, j, k) = (*d_mV)->at(i, j-1, k); // ?
        }
    }
    if ( (k == 0 || k == depth) && i < width && j < height) { // Z
        if (k == 0) { // zMin
            if ((*d_mW)->mClosedBounds[4])
                (*d_mW)->at(i, j, k) = 0.0f;
            else
                (*d_mW)->at(i, j, k) = (*d_mW)->at(i, j, k+1); // ?
        } else {      // zMax
             if ((*d_mW)->mClosedBounds[5])
                 (*d_mW)->at(i, j, k) = 0.0f;
             else
                 (*d_mW)->at(i, j, k) = (*d_mW)->at(i, j, k-1); // ?
        }
    }
}

FluidSolver::FluidSolver(Timer *tmr, SceneSettings *scn)
    /* DPCT_ORIG     : mWidth(scn->gridRes.x)*/
    : mWidth(scn->gridRes.x())
      /* DPCT_ORIG     , mHeight(scn->gridRes.y)*/
      ,
      mHeight(scn->gridRes.y())
      /* DPCT_ORIG     , mDepth(scn->gridRes.z)*/
      ,
      mDepth(scn->gridRes.z()), mDensity(scn->density),
      mMaxIter(scn->maxIterSolve), mDt(scn->dt), mDx(scn->dx), mTmr(tmr),
      mBuoyancy(scn->buoyancy), mCoolingRate(scn->coolingRate),
      mGravity(scn->gravity), mVorticeConf(scn->vorticityConf),
      mDrag(scn->drag), mTurbulenceAmp(scn->turbulence_amp),
      mTurbulenceScale(scn->turbulence_scale),
      mTurbMaskTempRamp(scn->turbMaskTempRamp),
      mTurbMaskVelRamp(scn->turbMaskVelRamp), mScn(scn),
      mSingleFrameSourceInit(false), mParticleCount(0)
      /* DPCT_ORIG     , mByteSize ( scn->gridRes.x    *  scn->gridRes.y    *
         scn->gridRes.z    * sizeof(float))*/
      ,
      mByteSize(scn->gridRes.x() * scn->gridRes.y() * scn->gridRes.z() *
                sizeof(float))
      /* DPCT_ORIG     , mByteSizeU((scn->gridRes.x+1) *  scn->gridRes.y    *
         scn->gridRes.z    * sizeof(float))*/
      ,
      mByteSizeU((scn->gridRes.x() + 1) * scn->gridRes.y() * scn->gridRes.z() *
                 sizeof(float))
      /* DPCT_ORIG     , mByteSizeV( scn->gridRes.x    * (scn->gridRes.y+1) *
         scn->gridRes.z    * sizeof(float))*/
      ,
      mByteSizeV(scn->gridRes.x() * (scn->gridRes.y() + 1) * scn->gridRes.z() *
                 sizeof(float))
      /* DPCT_ORIG     , mByteSizeW( scn->gridRes.x    *  scn->gridRes.y    *
         (scn->gridRes.z+1) * sizeof(float))*/
      ,
      mByteSizeW(scn->gridRes.x() * scn->gridRes.y() * (scn->gridRes.z() + 1) *
                 sizeof(float))
{
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mTFront, mByteSize));*/
    /*
    DPCT1003:245: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mTFront = (float *)sycl::malloc_device(
                         mByteSize, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mTBack, mByteSize));*/
    /*
    DPCT1003:246: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mTBack = (float *)sycl::malloc_device(
                         mByteSize, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mUFront, mByteSizeU));*/
    /*
    DPCT1003:247: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mUFront = (float *)sycl::malloc_device(
                         mByteSizeU, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mUBack, mByteSizeU));*/
    /*
    DPCT1003:248: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mUBack = (float *)sycl::malloc_device(
                         mByteSizeU, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mVFront, mByteSizeV));*/
    /*
    DPCT1003:249: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mVFront = (float *)sycl::malloc_device(
                         mByteSizeV, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mVBack, mByteSizeV));*/
    /*
    DPCT1003:250: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mVBack = (float *)sycl::malloc_device(
                         mByteSizeV, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mWFront, mByteSizeW));*/
    /*
    DPCT1003:251: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mWFront = (float *)sycl::malloc_device(
                         mByteSizeW, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mWBack, mByteSizeW));*/
    /*
    DPCT1003:252: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mWBack = (float *)sycl::malloc_device(
                         mByteSizeW, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mTFront, 0, mByteSize));*/
    /*
    DPCT1003:253: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mTFront, 0, mByteSize).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mTBack, 0, mByteSize));*/
    /*
    DPCT1003:254: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mTBack, 0, mByteSize).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mUFront, 0, mByteSizeU));*/
    /*
    DPCT1003:255: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mUFront, 0, mByteSizeU).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mUBack, 0, mByteSizeU));*/
    /*
    DPCT1003:256: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mUBack, 0, mByteSizeU).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mVFront, 0, mByteSizeV));*/
    /*
    DPCT1003:257: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mVFront, 0, mByteSizeV).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mVBack, 0, mByteSizeV));*/
    /*
    DPCT1003:258: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mVBack, 0, mByteSizeV).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mWFront, 0, mByteSizeW));*/
    /*
    DPCT1003:259: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mWFront, 0, mByteSizeW).wait(), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mWBack, 0, mByteSizeW));*/
    /*
    DPCT1003:260: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mWBack, 0, mByteSizeW).wait(), 0));

/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mRhs, mByteSize));*/
    /*
    DPCT1003:261: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mRhs = (float *)sycl::malloc_device(
                         mByteSize, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mP, mByteSize));*/
    /*
    DPCT1003:262: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mP = (float *)sycl::malloc_device(
                         mByteSize, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMemset(d_mP, 0, mByteSize));*/
    /*
    DPCT1003:263: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::get_default_queue().memset(d_mP, 0, mByteSize).wait(), 0));

    // create grids object on device
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mT, sizeof(Grid *)));*/
    /*
    DPCT1003:264: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (d_mT = sycl::malloc_device<Grid>(1, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mU, sizeof(Grid *)));*/
    /*
    DPCT1003:265: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (d_mU = sycl::malloc_device<Grid>(1, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mV, sizeof(Grid *)));*/
    /*
    DPCT1003:266: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (d_mV = sycl::malloc_device<Grid>(1, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void **)&d_mW, sizeof(Grid *)));*/
    /*
    DPCT1003:267: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (d_mW = sycl::malloc_device<Grid>(1, dpct::get_default_queue()), 0));
/* DPCT_ORIG     create_grids_kernel<<<1,1>>>(d_mT, d_mU, d_mV, d_mW, d_mTFront,
   d_mTBack, d_mUFront, d_mUBack, d_mVFront, d_mVBack, d_mWFront, d_mWBack,
   mWidth, mHeight, mDepth, mDx, *scn);*/
    create_grids_kernel(
        d_mT, d_mU, d_mV, d_mW, d_mTFront,
        d_mTBack, d_mUFront, d_mUBack, d_mVFront,
        d_mVBack, d_mWFront, d_mWBack, mWidth,
        mHeight, mDepth, mDx, *scn);
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:268: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:269: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    // allocate particle arrays on host and device
/* DPCT_ORIG     int byteSizePartFloat3 =
 * scn->sourceMaxParticleCount*sizeof(float3);*/
    /*
    DPCT1083:270: The size of float3 in the migrated code may be different from
    the original code. Check that the allocated memory size in the migrated code
    is correct.
    */
    int byteSizePartFloat3 = scn->sourceMaxParticleCount * sizeof(sycl::float3);
    int byteSizePartFloat = scn->sourceMaxParticleCount*sizeof(float);

/* DPCT_ORIG     h_mPartPos = (float3*)malloc(byteSizePartFloat3);*/
    h_mPartPos = (sycl::float3 *)malloc(byteSizePartFloat3);
/* DPCT_ORIG     h_mPartVel = (float3*)malloc(byteSizePartFloat3);*/
    h_mPartVel = (sycl::float3 *)malloc(byteSizePartFloat3);
    h_mPartPscale = (float*)malloc(byteSizePartFloat);
    h_mPartTemp = (float*)malloc(byteSizePartFloat);

/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mPartPos,
 * byteSizePartFloat3));*/
    /*
    DPCT1003:271: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mPartPos = (sycl::float3 *)sycl::malloc_device(
                         byteSizePartFloat3, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mPartVel,
 * byteSizePartFloat3));*/
    /*
    DPCT1003:272: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mPartVel = (sycl::float3 *)sycl::malloc_device(
                         byteSizePartFloat3, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mPartPscale,
 * byteSizePartFloat));*/
    /*
    DPCT1003:273: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mPartPscale = (float *)sycl::malloc_device(
                         byteSizePartFloat, dpct::get_default_queue()),
                     0));
/* DPCT_ORIG     checkCudaErrors(cudaMalloc((void**)&d_mPartTemp,
 * byteSizePartFloat));*/
    /*
    DPCT1003:274: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((d_mPartTemp = (float *)sycl::malloc_device(
                         byteSizePartFloat, dpct::get_default_queue()),
                     0));
}

FluidSolver::~FluidSolver() {
    // grids arrays
/* DPCT_ORIG     free_grids_kernel<<<1,1>>>(d_mT, d_mU, d_mV, d_mW);*/
    free_grids_kernel(d_mT, d_mU, d_mV, d_mW);
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:275: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mTFront));*/
    /*
    DPCT1003:276: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mTFront, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mTBack));*/
    /*
    DPCT1003:277: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mTBack, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mUFront));*/
    /*
    DPCT1003:278: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mUFront, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mUBack));*/
    /*
    DPCT1003:279: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mUBack, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mVFront));*/
    /*
    DPCT1003:280: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mVFront, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mVBack));*/
    /*
    DPCT1003:281: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mVBack, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mWFront));*/
    /*
    DPCT1003:282: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mWFront, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mWBack));*/
    /*
    DPCT1003:283: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mWBack, dpct::get_default_queue()), 0));

/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mRhs));*/
    /*
    DPCT1003:284: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mRhs, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mP));*/
    /*
    DPCT1003:285: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mP, dpct::get_default_queue()), 0));

    // particle arrays
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mPartPos));*/
    /*
    DPCT1003:286: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mPartPos, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mPartVel));*/
    /*
    DPCT1003:287: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mPartVel, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mPartPscale));*/
    /*
    DPCT1003:288: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mPartPscale, dpct::get_default_queue()), 0));
/* DPCT_ORIG     checkCudaErrors(cudaFree(d_mPartTemp));*/
    /*
    DPCT1003:289: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((sycl::free(d_mPartTemp, dpct::get_default_queue()), 0));
    free(h_mPartPos);
    free(h_mPartVel);
    free(h_mPartPscale);
    free(h_mPartTemp);
}

void FluidSolver::addSource() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  auto info = dev_ct1.get_device_info();

  std::cout << "Currunt device: " << info.get_name() << std::endl;

  sycl::queue &q_ct1 = dev_ct1.default_queue();

    /*
    DPCT1008:290: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->source_in = clock();

    // if sourcing is not animated it will happen on every frame
    if (!mScn->sourceAnimated || (mScn->sourceAnimated &&
                                  /* DPCT_ORIG mTmr->iter >= mScn->sourceRange.x
                                     &&*/
                                  mTmr->iter >= mScn->sourceRange.x() &&
                                  /* DPCT_ORIG mTmr->iter <=
                                     mScn->sourceRange.y))*/
                                  mTmr->iter <= mScn->sourceRange.y()))
    {
        if (!mSingleFrameSourceInit || mScn->sourceAnimated) {
            // parse custom particles file and fill host particle array
            Parser::sourceParticleParse(mScn, h_mPartPos, h_mPartVel, h_mPartPscale, h_mPartTemp, mParticleCount, mTmr->iter);

            if (mParticleCount > 0) {
                // copy particle arrays to device
/* DPCT_ORIG                 int byteSizePartFloat3 =
 * mParticleCount*sizeof(float3);*/
                /*
                DPCT1083:291: The size of float3 in the migrated code may be
                different from the original code. Check that the allocated
                memory size in the migrated code is correct.
                */
                int byteSizePartFloat3 = mParticleCount * sizeof(sycl::float3);
                int byteSizePartFloat = mParticleCount*sizeof(float);
/* DPCT_ORIG                 checkCudaErrors(cudaMemcpy(d_mPartPos, h_mPartPos,
 * byteSizePartFloat3, cudaMemcpyHostToDevice));*/
                /*
                DPCT1003:292: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(d_mPartPos, h_mPartPos, byteSizePartFloat3)
                         .wait(),
                     0));
/* DPCT_ORIG                 checkCudaErrors(cudaMemcpy(d_mPartVel, h_mPartVel,
 * byteSizePartFloat3, cudaMemcpyHostToDevice));*/
                /*
                DPCT1003:293: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(d_mPartVel, h_mPartVel, byteSizePartFloat3)
                         .wait(),
                     0));
/* DPCT_ORIG                 checkCudaErrors(cudaMemcpy(d_mPartPscale,
 * h_mPartPscale, byteSizePartFloat, cudaMemcpyHostToDevice));*/
                /*
                DPCT1003:294: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors((
                    dpct::get_default_queue()
                        .memcpy(d_mPartPscale, h_mPartPscale, byteSizePartFloat)
                        .wait(),
                    0));
/* DPCT_ORIG                 checkCudaErrors(cudaMemcpy(d_mPartTemp,
 * h_mPartTemp, byteSizePartFloat, cudaMemcpyHostToDevice));*/
                /*
                DPCT1003:295: Migrated API does not return error code. (*, 0) is
                inserted. You may need to rewrite this code.
                */
                checkCudaErrors(
                    (dpct::get_default_queue()
                         .memcpy(d_mPartTemp, h_mPartTemp, byteSizePartFloat)
                         .wait(),
                     0));
            }

            mSingleFrameSourceInit = true;
        }

        if (mParticleCount > 0) {
            // clear back buffers
/* DPCT_ORIG             dim3 block(8, 8, 8);*/
            sycl::range<3> block(8, 8, 8);
/* DPCT_ORIG             dim3 grid((mWidth+1)/block.x+1, (mHeight+1)/block.y+1,
 * (mDepth+1)/block.z+1);*/
            sycl::range<3> grid((mDepth + 1) / block[0] + 1,
                                (mHeight + 1) / block[1] + 1,
                                (mWidth + 1) / block[2] + 1);
/* DPCT_ORIG             clear_back_buffer_kernel <<< grid, block >>> (d_mT,
 * d_mU, d_mV, d_mW);*/
            /*
            DPCT1049:8: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto d_mT_ct0 = &d_mT;
                auto d_mU_ct1 = &d_mU;
                auto d_mV_ct2 = &d_mV;
                auto d_mW_ct3 = &d_mW;

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     clear_back_buffer_kernel(
                                         d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                         item_ct1);
                                 });
            });
/* DPCT_ORIG             checkCudaErrors(cudaGetLastError());*/
            /*
            DPCT1010:296: SYCL uses exceptions to report errors and does not use
            the error codes. The call was replaced with 0. You need to rewrite
            this code.
            */
            checkCudaErrors(0);
/* DPCT_ORIG             checkCudaErrors(cudaDeviceSynchronize());*/
            /*
            DPCT1003:297: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_current_device().queues_wait_and_throw(), 0));

            // scatter particle-to-grid to back buffer
/* DPCT_ORIG             dim3 blockPart(32, 1, 1);*/
            sycl::range<3> blockPart(1, 1, 32);
/* DPCT_ORIG             dim3 gridPart(mParticleCount/blockPart.x+1, 1, 1);*/
            sycl::range<3> gridPart(1, 1, mParticleCount / blockPart[2] + 1);
/* DPCT_ORIG             add_source_to_back_buffer_kernel <<< gridPart,
   blockPart >>> (d_mT, d_mU, d_mV, d_mW, d_mPartPos, d_mPartVel, d_mPartPscale,
   d_mPartTemp, mDx, mParticleCount);*/
            /*
            DPCT1049:9: The work-group size passed to the SYCL kernel may exceed
            the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto d_mT_ct0 = &d_mT;
                auto d_mU_ct1 = &d_mU;
                auto d_mV_ct2 = &d_mV;
                auto d_mW_ct3 = &d_mW;
                auto d_mPartPos_ct4 = d_mPartPos;
                auto d_mPartVel_ct5 = d_mPartVel;
                auto d_mPartPscale_ct6 = d_mPartPscale;
                auto d_mPartTemp_ct7 = d_mPartTemp;
                auto mDx_ct8 = mDx;
                auto mParticleCount_ct9 = mParticleCount;

                cgh.parallel_for(
                    sycl::nd_range<3>(gridPart * blockPart, blockPart),
                    [=](sycl::nd_item<3> item_ct1) {
                        add_source_to_back_buffer_kernel(
                            d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                            d_mPartPos_ct4, d_mPartVel_ct5, d_mPartPscale_ct6,
                            d_mPartTemp_ct7, mDx_ct8, mParticleCount_ct9,
                            item_ct1);
                    });
            });
/* DPCT_ORIG             checkCudaErrors(cudaGetLastError());*/
            /*
            DPCT1010:298: SYCL uses exceptions to report errors and does not use
            the error codes. The call was replaced with 0. You need to rewrite
            this code.
            */
            checkCudaErrors(0);
/* DPCT_ORIG             checkCudaErrors(cudaDeviceSynchronize());*/
            /*
            DPCT1003:299: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_current_device().queues_wait_and_throw(), 0));

            // copy value from back buffer to front buffer
/* DPCT_ORIG             set_source_from_back_to_front_kernel <<< grid, block
 * >>> (d_mT, d_mU, d_mV, d_mW);*/
            /*
            DPCT1049:10: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto d_mT_ct0 = &d_mT;
                auto d_mU_ct1 = &d_mU;
                auto d_mV_ct2 = &d_mV;
                auto d_mW_ct3 = &d_mW;

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     set_source_from_back_to_front_kernel(
                                         d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                         item_ct1);
                                 });
            });
/* DPCT_ORIG             checkCudaErrors(cudaGetLastError());*/
            /*
            DPCT1010:300: SYCL uses exceptions to report errors and does not use
            the error codes. The call was replaced with 0. You need to rewrite
            this code.
            */
            checkCudaErrors(0);
/* DPCT_ORIG             checkCudaErrors(cudaDeviceSynchronize());*/
            /*
            DPCT1003:301: Migrated API does not return error code. (*, 0) is
            inserted. You may need to rewrite this code.
            */
            checkCudaErrors(
                (dpct::get_current_device().queues_wait_and_throw(), 0));
        }
    }

    /*
    DPCT1008:302: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->source_out = clock();
}

void FluidSolver::project() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

/* DPCT_ORIG     dim3 block(8, 8, 8);*/
    sycl::range<3> block(8, 8, 8);
/* DPCT_ORIG     dim3 grid(mWidth/block.x, mHeight/block.y, mDepth/block.z);*/
    sycl::range<3> grid(mDepth / block[0], mHeight / block[1],
                        mWidth / block[2]);

    /*
    DPCT1008:303: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->computeDivergence_in = clock();

/* DPCT_ORIG     compute_divergence_kernel <<< grid, block >>> (d_mRhs, d_mU,
   d_mV, d_mW, mWidth, mHeight, mDepth, mDensity, mDx);*/
    /*
    DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mRhs_ct0 = d_mRhs;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mWidth_ct4 = mWidth;
        auto mHeight_ct5 = mHeight;
        auto mDepth_ct6 = mDepth;
        auto mDensity_ct7 = mDensity;
        auto mDx_ct8 = mDx;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             compute_divergence_kernel(
                                 d_mRhs_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                 mWidth_ct4, mHeight_ct5, mDepth_ct6,
                                 mDensity_ct7, mDx_ct8, item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:304: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:305: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:306: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->computeDivergence_out = clock();

    /*
    DPCT1008:307: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->gsSolve_in = clock();

/* DPCT_ORIG     dim3 blockGS(16, 16, 1);*/
    sycl::range<3> blockGS(1, 16, 16);
/* DPCT_ORIG     dim3 gridGS(mWidth/blockGS.x, mHeight/blockGS.y, 1);*/
    sycl::range<3> gridGS(1, mHeight / blockGS[1], mWidth / blockGS[2]);
/* DPCT_ORIG     gs_solve_kernel <<< gridGS, blockGS >>> (d_mRhs, d_mP,
 * mDensity, mDx, mDt, mWidth, mHeight, mDepth, mMaxIter);*/
    /*
    DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mRhs_ct0 = d_mRhs;
        auto d_mP_ct1 = d_mP;
        auto mDensity_ct2 = mDensity;
        auto mDx_ct3 = mDx;
        auto mDt_ct4 = mDt;
        auto mWidth_ct5 = mWidth;
        auto mHeight_ct6 = mHeight;
        auto mDepth_ct7 = mDepth;
        auto mMaxIter_ct8 = mMaxIter;

        cgh.parallel_for(sycl::nd_range<3>(gridGS * blockGS, blockGS),
                         [=](sycl::nd_item<3> item_ct1) {
                             gs_solve_kernel(d_mRhs_ct0, d_mP_ct1, mDensity_ct2,
                                             mDx_ct3, mDt_ct4, mWidth_ct5,
                                             mHeight_ct6, mDepth_ct7,
                                             mMaxIter_ct8, item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:308: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:309: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:310: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->gsSolve_out = clock();

    /*
    DPCT1008:311: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->pressureGradientUpdate_in = clock();

/* DPCT_ORIG     dim3 gridP((mWidth+1)/block.x+1, (mHeight+1)/block.y+1,
 * (mDepth+1)/block.z+1); */
    sycl::range<3> gridP((mDepth + 1) / block[0] + 1,
                         (mHeight + 1) / block[1] + 1,
                         (mWidth + 1) / block[2] +
                             1); // padding for the staggered velocity bounds
/* DPCT_ORIG     pressure_gradient_update_kernel <<< gridP, block >>> (d_mP,
   d_mU, d_mV, d_mW, mWidth, mHeight, mDepth, mDensity, mDx, mDt);*/
    /*
    DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mP_ct0 = d_mP;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mWidth_ct4 = mWidth;
        auto mHeight_ct5 = mHeight;
        auto mDepth_ct6 = mDepth;
        auto mDensity_ct7 = mDensity;
        auto mDx_ct8 = mDx;
        auto mDt_ct9 = mDt;

        cgh.parallel_for(sycl::nd_range<3>(gridP * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             pressure_gradient_update_kernel(
                                 d_mP_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                 mWidth_ct4, mHeight_ct5, mDepth_ct6,
                                 mDensity_ct7, mDx_ct8, mDt_ct9, item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:312: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:313: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:314: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->pressureGradientUpdate_out = clock();
}

void FluidSolver::step() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
/* DPCT_ORIG     dim3 block(8, 8, 8);*/
    sycl::range<3> block(8, 8, 8);
/* DPCT_ORIG     dim3 gridT(mWidth     /block.x,     mHeight   /block.y, mDepth
 * /block.z    );*/
    sycl::range<3> gridT(mDepth / block[0], mHeight / block[1],
                         mWidth / block[2]);
/* DPCT_ORIG     dim3 gridU((mWidth+1) /block.x + 1, mHeight   /block.y, mDepth
 * /block.z    );*/
    sycl::range<3> gridU(mDepth / block[0], mHeight / block[1],
                         (mWidth + 1) / block[2] + 1);
/* DPCT_ORIG     dim3 gridV(mWidth     /block.x,    (mHeight+1)/block.y + 1,
 * mDepth   /block.z    );*/
    sycl::range<3> gridV(mDepth / block[0], (mHeight + 1) / block[1] + 1,
                         mWidth / block[2]);
/* DPCT_ORIG     dim3 gridW(mWidth     /block.x,     mHeight   /block.y,
 * (mDepth+1)/block.z + 1);*/
    sycl::range<3> gridW((mDepth + 1) / block[0] + 1, mHeight / block[1],
                         mWidth / block[2]);
/* DPCT_ORIG     dim3 gridUVW(mWidth+1 /block.x + 1,(mHeight+1)/block.y + 1,
 * (mDepth+1)/block.z + 1);*/
    sycl::range<3> gridUVW((mDepth + 1) / block[0] + 1,
                           (mHeight + 1) / block[1] + 1,
                           mWidth + 1 / block[2] + 1);

    /*
    DPCT1008:315: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->cooldown_in = clock();

    if (abs(mCoolingRate) > 0.0f) {
/* DPCT_ORIG         temperature_cooldown_kernel <<< gridT, block >>> (d_mT,
 * mCoolingRate, mDt);*/
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mT_ct0 = &d_mT;
            auto mCoolingRate_ct1 = mCoolingRate;
            auto mDt_ct2 = mDt;

            cgh.parallel_for(sycl::nd_range<3>(gridT * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 temperature_cooldown_kernel(d_mT_ct0,
                                                             mCoolingRate_ct1,
                                                             mDt_ct2, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:316: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:317: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:318: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->cooldown_out = clock();

    /*
    DPCT1008:319: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->drag_in = clock();

    if (mDrag > 0.0f) {
/* DPCT_ORIG         apply_drag_kernel <<< gridUVW, block >>> (d_mU, d_mV, d_mW,
 * mDrag, mDt);*/
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mU_ct0 = &d_mU;
            auto d_mV_ct1 = &d_mV;
            auto d_mW_ct2 = &d_mW;
            auto mDrag_ct3 = mDrag;
            auto mDt_ct4 = mDt;

            cgh.parallel_for(sycl::nd_range<3>(gridUVW * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 apply_drag_kernel(d_mU_ct0, d_mV_ct1, d_mW_ct2,
                                                   mDrag_ct3, mDt_ct4,
                                                   item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:320: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:321: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:322: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->drag_out = clock();

    /*
    DPCT1008:323: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->vorticity_in = clock();

    if (abs(mVorticeConf) > 0.0f) {
/* DPCT_ORIG         copy_velocity_to_back_buffer_kernel <<< gridUVW, block >>>
 * (d_mU, d_mV, d_mW);*/
        /*
        DPCT1049:20: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mU_ct0 = &d_mU;
            auto d_mV_ct1 = &d_mV;
            auto d_mW_ct2 = &d_mW;

            cgh.parallel_for(sycl::nd_range<3>(gridUVW * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 copy_velocity_to_back_buffer_kernel(
                                     d_mU_ct0, d_mV_ct1, d_mW_ct2, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:324: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:325: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
/* DPCT_ORIG         vorticity_confinement_kernel <<< gridT, block >>> (d_mT,
 * d_mU, d_mV, d_mW, mVorticeConf, mDx, mDt);*/
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mT_ct0 = &d_mT;
            auto d_mU_ct1 = &d_mU;
            auto d_mV_ct2 = &d_mV;
            auto d_mW_ct3 = &d_mW;
            auto mVorticeConf_ct4 = mVorticeConf;
            auto mDx_ct5 = mDx;
            auto mDt_ct6 = mDt;

            cgh.parallel_for(sycl::nd_range<3>(gridT * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 vorticity_confinement_kernel(
                                     d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                     mVorticeConf_ct4, mDx_ct5, mDt_ct6,
                                     item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:326: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:327: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
/* DPCT_ORIG         swap_vel_grids_kernel <<< 1, 1 >>> (d_mU, d_mV, d_mW);*/
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mU_ct0 = &d_mU;
            auto d_mV_ct1 = &d_mV;
            auto d_mW_ct2 = &d_mW;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 1),
                                               sycl::range<3>(1, 1, 1)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 swap_vel_grids_kernel(d_mU_ct0, d_mV_ct1,
                                                       d_mW_ct2, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:328: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:329: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:330: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->vorticity_out = clock();

    /*
    DPCT1008:331: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->buoyancy_in = clock();

    if (abs(mBuoyancy) > 0.0f) {
/* DPCT_ORIG         add_buoyancy_kernel <<< gridV, block >>> (d_mT, d_mV,
 * mBuoyancy, mGravity, mDt);*/
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mT_ct0 = &d_mT;
            auto d_mV_ct1 = &d_mV;
            auto mBuoyancy_ct2 = mBuoyancy;
            auto mGravity_ct3 = mGravity;
            auto mDt_ct4 = mDt;

            cgh.parallel_for(sycl::nd_range<3>(gridV * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 add_buoyancy_kernel(
                                     d_mT_ct0, d_mV_ct1, mBuoyancy_ct2,
                                     mGravity_ct3, mDt_ct4, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:332: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:333: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:334: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->buoyancy_out = clock();

    /*
    DPCT1008:335: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->wind_in = clock();

    if (mScn->windAmp > 0.0f) {
/* DPCT_ORIG         add_wind_kernel <<< gridUVW, block >>> (d_mU, d_mV, d_mW,
   mScn->windDir, mScn->windAmp, mScn->windSpeed, mScn->windTurbAmp,
   mScn->windTurbScale, mDt, mTmr->time);*/
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mU_ct0 = &d_mU;
            auto d_mV_ct1 = &d_mV;
            auto d_mW_ct2 = &d_mW;
            auto mScn_windDir_ct3 = mScn->windDir;
            auto mScn_windAmp_ct4 = mScn->windAmp;
            auto mScn_windSpeed_ct5 = mScn->windSpeed;
            auto mScn_windTurbAmp_ct6 = mScn->windTurbAmp;
            auto mScn_windTurbScale_ct7 = mScn->windTurbScale;
            auto mDt_ct8 = mDt;
            auto mTmr_time_ct9 = mTmr->time;

            cgh.parallel_for(sycl::nd_range<3>(gridUVW * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 add_wind_kernel(
                                     d_mU_ct0, d_mV_ct1, d_mW_ct2,
                                     mScn_windDir_ct3, mScn_windAmp_ct4,
                                     mScn_windSpeed_ct5, mScn_windTurbAmp_ct6,
                                     mScn_windTurbScale_ct7, mDt_ct8,
                                     mTmr_time_ct9, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:336: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:337: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:338: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->wind_out = clock();

    /*
    DPCT1008:339: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->turbulence_in = clock();

    if (mTurbulenceAmp > 0.0f) {
/* DPCT_ORIG         compute_turbulence_kernel <<< gridUVW, block >>> (d_mT,
   d_mU, d_mV, d_mW, mTurbulenceAmp, mTurbulenceScale, mTurbMaskTempRamp,
   mTurbMaskVelRamp, mDt, mTmr->time);*/
        /*
        DPCT1049:24: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mT_ct0 = &d_mT;
            auto d_mU_ct1 = &d_mU;
            auto d_mV_ct2 = &d_mV;
            auto d_mW_ct3 = &d_mW;
            auto mTurbulenceAmp_ct4 = mTurbulenceAmp;
            auto mTurbulenceScale_ct5 = mTurbulenceScale;
            auto mTurbMaskTempRamp_ct6 = mTurbMaskTempRamp;
            auto mTurbMaskVelRamp_ct7 = mTurbMaskVelRamp;
            auto mDt_ct8 = mDt;
            auto mTmr_time_ct9 = mTmr->time;

            cgh.parallel_for(sycl::nd_range<3>(gridUVW * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 compute_turbulence_kernel(
                                     d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                     mTurbulenceAmp_ct4, mTurbulenceScale_ct5,
                                     mTurbMaskTempRamp_ct6,
                                     mTurbMaskVelRamp_ct7, mDt_ct8,
                                     mTmr_time_ct9, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:340: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:341: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));

/* DPCT_ORIG         add_curl_noise_kernel <<< gridT, block >>> (d_mU, d_mV,
 * d_mW, mDx, mDt);*/
        /*
        DPCT1049:25: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto d_mU_ct0 = &d_mU;
            auto d_mV_ct1 = &d_mV;
            auto d_mW_ct2 = &d_mW;
            auto mDx_ct3 = mDx;
            auto mDt_ct4 = mDt;

            cgh.parallel_for(sycl::nd_range<3>(gridT * block, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 add_curl_noise_kernel(d_mU_ct0, d_mV_ct1,
                                                       d_mW_ct2, mDx_ct3,
                                                       mDt_ct4, item_ct1);
                             });
        });
/* DPCT_ORIG         checkCudaErrors(cudaGetLastError());*/
        /*
        DPCT1010:342: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        checkCudaErrors(0);
/* DPCT_ORIG         checkCudaErrors(cudaDeviceSynchronize());*/
        /*
        DPCT1003:343: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::get_current_device().queues_wait_and_throw(), 0));
    }

    /*
    DPCT1008:344: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->turbulence_out = clock();

    project();

    /*
    DPCT1008:345: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->advect_in = clock();

/* DPCT_ORIG     advect_RK3_kernel <<< gridT, block >>> (d_mT , d_mU, d_mV,
 * d_mW, mDt, true);*/
    /*
    DPCT1049:14: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mT_ct0 = &d_mT;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mDt_ct4 = mDt;

        cgh.parallel_for(sycl::nd_range<3>(gridT * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             advect_RK3_kernel(d_mT_ct0, d_mU_ct1, d_mV_ct2,
                                               d_mW_ct3, mDt_ct4, true,
                                               item_ct1);
                         });
    });
/* DPCT_ORIG     advect_RK3_kernel <<< gridU, block >>> (d_mU , d_mU, d_mV,
 * d_mW, mDt, false);*/
    /*
    DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mU_ct0 = &d_mU;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mDt_ct4 = mDt;

        cgh.parallel_for(sycl::nd_range<3>(gridU * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             advect_RK3_kernel(d_mU_ct0, d_mU_ct1, d_mV_ct2,
                                               d_mW_ct3, mDt_ct4, false,
                                               item_ct1);
                         });
    });
/* DPCT_ORIG     advect_RK3_kernel <<< gridV, block >>> (d_mV , d_mU, d_mV,
 * d_mW, mDt, false);*/
    /*
    DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mV_ct0 = &d_mV;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mDt_ct4 = mDt;

        cgh.parallel_for(sycl::nd_range<3>(gridV * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             advect_RK3_kernel(d_mV_ct0, d_mU_ct1, d_mV_ct2,
                                               d_mW_ct3, mDt_ct4, false,
                                               item_ct1);
                         });
    });
/* DPCT_ORIG     advect_RK3_kernel <<< gridW, block >>> (d_mW , d_mU, d_mV,
 * d_mW, mDt, false);*/
    /*
    DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mW_ct0 = &d_mW;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;
        auto mDt_ct4 = mDt;

        cgh.parallel_for(sycl::nd_range<3>(gridW * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             advect_RK3_kernel(d_mW_ct0, d_mU_ct1, d_mV_ct2,
                                               d_mW_ct3, mDt_ct4, false,
                                               item_ct1);
                         });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:346: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:347: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));

    /*
    DPCT1008:348: clock function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    mTmr->advect_out = clock();

/* DPCT_ORIG     swap_grids_kernel <<< 1, 1 >>> (d_mT, d_mU, d_mV, d_mW);*/
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto d_mT_ct0 = &d_mT;
        auto d_mU_ct1 = &d_mU;
        auto d_mV_ct2 = &d_mV;
        auto d_mW_ct3 = &d_mW;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
            [=](sycl::nd_item<3> item_ct1) {
                swap_grids_kernel(d_mT_ct0, d_mU_ct1, d_mV_ct2, d_mW_ct3,
                                  item_ct1);
            });
    });
/* DPCT_ORIG     checkCudaErrors(cudaGetLastError());*/
    /*
    DPCT1010:349: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    checkCudaErrors(0);
/* DPCT_ORIG     checkCudaErrors(cudaDeviceSynchronize());*/
    /*
    DPCT1003:350: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors((dpct::get_current_device().queues_wait_and_throw(), 0));
}
