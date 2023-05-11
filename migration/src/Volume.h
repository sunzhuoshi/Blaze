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


#ifndef VOLUME_H
#define VOLUME_H

/* DPCT_ORIG #include "cuda_runtime.h"*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils/helper_cuda.h"
#include "utils/helper_math.h"
#include "dpcx/dpcx.hpp"

class Volume {

public:
/* DPCT_ORIG     cudaArray             *content;*/
    dpct::image_matrix *content;
/* DPCT_ORIG     cudaExtent            size;*/
    sycl::range<3> size{0, 0, 0};
/* DPCT_ORIG     cudaChannelFormatDesc channelDesc;*/
    dpct::image_channel channelDesc;
/* DPCT_ORIG     cudaTextureObject_t   volumeTex;*/
    dpct::image_wrapper_base_p volumeTex;
    dpct::image_wrapper_base_p volumeSurf;

    Volume(int w, int h, int d){
        // create 3D array
/* DPCT_ORIG         cudaExtent dataSize = {(size_t)w, (size_t)h, (size_t)d};*/
        sycl::range<3> dataSize = {(size_t)w, (size_t)h, (size_t)d};
/* DPCT_ORIG         channelDesc = cudaCreateChannelDesc<float>();*/
        /*
        DPCT1059:0: SYCL only supports 4-channel image format. Adjust the code.
        */
        channelDesc = dpct::image_channel::create<float>();
/* DPCT_ORIG         checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc,
 * dataSize, cudaArraySurfaceLoadStore));*/
        /*
        DPCT1003:217: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (content = new dpct::image_matrix(channelDesc, dataSize), 0));
        size = dataSize;

/* DPCT_ORIG         cudaResourceDesc surfRes;*/
        dpct::image_data surfRes;
/* DPCT_ORIG         memset(&surfRes, 0, sizeof(cudaResourceDesc));*/
        memset(&surfRes, 0, sizeof(dpct::image_data));
/* DPCT_ORIG         surfRes.resType = cudaResourceTypeArray;*/

/* DPCT_ORIG         surfRes.res.array.array = content;*/

        surfRes.set_data(content);

        /*
        DPCT1007:218: Migration of cudaCreateSurfaceObject is not supported.
        */
        checkCudaErrors(dpcx::CreateSurfaceObject(&volumeSurf, &surfRes));

/* DPCT_ORIG         cudaResourceDesc texRes;*/
        dpct::image_data texRes;
/* DPCT_ORIG         memset(&texRes, 0, sizeof(cudaResourceDesc));*/
        memset(&texRes, 0, sizeof(dpct::image_data));

/* DPCT_ORIG         texRes.resType = cudaResourceTypeArray;*/
        texRes.set_data_type(dpct::image_data_type::matrix);
/* DPCT_ORIG         texRes.res.array.array = content;*/
        texRes.set_data_ptr(content);

/* DPCT_ORIG         cudaTextureDesc texDescr;*/
        dpct::sampling_info texDescr;
/* DPCT_ORIG         memset(&texDescr, 0, sizeof(cudaTextureDesc));*/
        memset(&texDescr, 0, sizeof(dpct::sampling_info));

/* DPCT_ORIG         texDescr.filterMode     = cudaFilterModeLinear;*/
        texDescr.set(sycl::filtering_mode::linear);
/* DPCT_ORIG         texDescr.addressMode[0] = cudaAddressModeClamp;*/
/* DPCT_ORIG         texDescr.addressMode[1] = cudaAddressModeClamp;*/
/* DPCT_ORIG         texDescr.addressMode[2] = cudaAddressModeClamp;*/
        texDescr.set(sycl::addressing_mode::clamp_to_edge);

/* DPCT_ORIG         checkCudaErrors(cudaCreateTextureObject(&volumeTex,
 * &texRes, &texDescr, NULL));*/
        /*
        DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        checkCudaErrors(
            (volumeTex = dpct::create_image_wrapper(texRes, texDescr), 0));
    }

    ~Volume() {
/* DPCT_ORIG         checkCudaErrors(cudaDestroyTextureObject(volumeTex));*/
        /*
        DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        checkCudaErrors((delete volumeTex, 0));
        /*
        DPCT1007:219: Migration of cudaDestroySurfaceObject is not supported.
        */
        checkCudaErrors(dpcx::DestroySurfaceObject(volumeSurf));
/* DPCT_ORIG         checkCudaErrors(cudaFreeArray(content));*/
        /*
        DPCT1003:220: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((delete content, 0));
        content = 0;
    }
};


#endif
