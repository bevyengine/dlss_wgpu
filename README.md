# dlss_wgpu - Deep Learning Super Sampling for wgpu

A wrapper for using [DLSS](https://www.nvidia.com/en-us/geforce/technologies/dlss) with [wgpu](https://github.com/gfx-rs/wgpu) when targeting Vulkan.


## Version Chart

| dlss_wgpu |   dlss   | wgpu |
|:---------:|:--------:|:----:|
|    v1.0   | v310.1.0 |  v25 |

## Downloading The DLSS SDK
The DLSS SDK cannot be redistributed by this crate. You will need to download the SDK as follows:
* Clone the NVIDIA DLSS Super Resolution SDK v310.1.0 from https://github.com/NVIDIA/DLSS
* Set the environment variable `DLSS_SDK = /path/to/DLSS`
* Ensure you comply with the DLSS SDK license located at https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt

## Build Dependencies
* Install the DLSS SDK
* Install the Vulkan SDK https://vulkan.lunarg.com/sdk/home and set the `VULKAN_SDK` environment variable
* Install clang https://rust-lang.github.io/rust-bindgen/requirements.html#clang

## Debug Overlay
When `dlss_wgpu` is compiled with the `debug_overlay` cargo feature, and the `DLSS_SDK` environment variable is set, the development version of the DLSS DLL will be linked.

The development version of the DLSS SDK comes with an in-app overlay to help debug usage of DLSS. See section `8.2` of `$DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf` for details.

## Distributing Your App
Once your app is compiled, you do not need to distribute the entire DLSS SDK, or set the `DLSS_SDK` environment variable. You only need to distribute the DLSS DLL and license text as follows:

* On Windows, copy `$DLSS_SDK/lib/Windows_x86_64/rel/nvngx_dlss.dll` to the same directory as your app
* On Linux, copy `$DLSS_SDK/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.310.1.0` to the same directory as your app
* Include the full copyright and license blurb texts from section `9.5` of `$DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf` with your app
