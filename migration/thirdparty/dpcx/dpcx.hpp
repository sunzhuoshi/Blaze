#pragma once
/*
* typedef __device_builtin__ unsigned long long cudaTextureObject_t;
* typedef __device_builtin__ unsigned long long cudaSurfaceObject_t;
* 
* cudaTextureObject_t ==> dpct::image_wrapper_base_p or dpct::image_accessor_ext<Type, Dimensions>
* cudaSurfaceObject_t ==> ?
*/

#include <dpct/image.hpp>

namespace dpcx {
    inline int CreateSurfaceObject(dpct::image_wrapper_base_p* SurfaceObject, dpct::image_data* ResourceDesc) {
        return 0;
    }
    
    inline int DestroySurfaceObject(dpct::image_wrapper_base_p SurfaceObject) {
        return 0;
    }

    inline void surf3Dwrite(float Val, dpct::image_wrapper_base_p SurfaceObject, int X, int Y, int Z) {
    }

    template <class T, int dimensions>
    dpct::image_accessor_ext<T, dimensions> ImageToAccessorExt(dpct::image_wrapper_base_p Image, sycl::handler& cgh) {
        dpct::image_wrapper<T, dimensions, false>* RealImage = dynamic_cast<dpct::image_wrapper<T, dimensions, false> *>(Image);
        return dpct::image_accessor_ext<T, dimensions, false>(RealImage->get_sampler(), RealImage->get_access(cgh));
    }
}
