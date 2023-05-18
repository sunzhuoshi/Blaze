#pragma once
/*
* typedef __device_builtin__ unsigned long long cudaTextureObject_t;
* typedef __device_builtin__ unsigned long long cudaSurfaceObject_t;
* 
* cudaTextureObject_t ==> dpct::image_wrapper_base_p or dpct::image_accessor_ext<Type, Dimensions>
* cudaSurfaceObject_t ==> dpct::image_wrapper_base_p?
*/

#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <sycl/usm.hpp>
#include <dpct/dpct.hpp>

namespace experimental = sycl::ext::oneapi::experimental;

namespace dpcx {
    inline int CreateSurfaceObject(dpct::image_wrapper_base_p* surface_object, dpct::image_data* resource_desc) {
        return 0;
    }
    
    inline int DestroySurfaceObject(dpct::image_wrapper_base_p surface_object) {
        return 0;
    }

    inline void surf3Dwrite(float val, dpct::image_wrapper_base_p surface_object, int x, int y, int z) {
        experimental::printf("test");
    }

    template <class T, int dimensions>
    dpct::image_accessor_ext<T, dimensions> ImageToAccessorExt(dpct::image_wrapper_base_p image, sycl::handler& cgh) {
        dpct::image_wrapper<T, dimensions, false>* RealImage = dynamic_cast<dpct::image_wrapper<T, dimensions, false> *>(image);
        return dpct::image_accessor_ext<T, dimensions, false>(RealImage->get_sampler(), RealImage->get_access(cgh));
    }
    inline void free(void* ptr) {
        sycl::free(ptr, dpct::get_default_queue());
    }
    template <class T>
    inline void init_device_mem(void* device_mem, T* host_mem) {
        dpct::get_default_queue().memcpy(device_mem, host_mem, sizeof(T)).wait();
        delete host_mem;
    }
}
