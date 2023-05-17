# Issue list
##  Language extension limitation
1. SYCL kernel cannot allocate storage

    new/delete is supported in CUDA kernels, but not in DPC++

    Sample code:
    ```
    /* DPCT_ORIG __global__ void create_scene_kernel(Camera **d_mCamera, Shader
   **d_mShader, Light **d_mLights, SceneSettings scn, float dx)*/
    void create_scene_kernel(Camera **d_mCamera, Shader **d_mShader,
                            Light **d_mLights, SceneSettings scn, float dx,
                            const sycl::nd_item<3> &item_ct1)
    {
    /* DPCT_ORIG     if (threadIdx.x == 0 && blockIdx.x == 0) {*/
        if (item_ct1.get_local_id(2) == 0 && item_ct1.get_group(2) == 0) {
            *d_mCamera =
                new Camera(scn.camTrans, scn.camU, scn.camV, scn.camW, scn.camFocal,
                        /* DPCT_ORIG scn.camAperture, scn.renderRes.x,
                            scn.renderRes.y);*/
                        scn.camAperture, scn.renderRes.x(), scn.renderRes.y());
            *d_mShader = new Shader(scn);
            for (int i=0; i<scn.lightCount; i++) {
                d_mLights[i] = new Light(scn, i);
            }
        }
    }
    ```
    Solution:

    Rewrite code manually
## Inadequate CUDA runtime API migration support

Around 50% of CUDA runtime API migration supported

Source: https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/docs/dev_guide/api-mapping-status/Runtime_and_Driver_API_migration_status.csv 

Unsupported API list in current project:
```
cudaSurfaceObject_t
cudaCreateSurfaceObject()
cudaDestroySurfaceObject()
surf3Dwrite()
```
Solution:

Migrate APIs manually without DPCT support

## Other issues
https://github.com/oneapi-src/SYCLomatic/issues/903
## Reference
https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html
https://www.codingame.com/playgrounds/53666/dpc/unified-shared-memory