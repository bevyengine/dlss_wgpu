# dlss_wgpu - Deep Learning Super Sampling for wgpu

## Downloading The DLSS SDK
The DLSS SDK cannot be redistributed by this crate. You will need to download the SDK as follows:
* Clone the NVIDIA DLSS Super Resolution SDK v310.1.0 from https://github.com/NVIDIA/DLSS
* Set the environment variable `DLSS_SDK = /path/to/dlss_sdk/DLSS`
* Ensure you comply with the DLSS SDK license located at https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt

## Build Dependencies
* On Windows, install `clang.exe` (following https://rust-lang.github.io/rust-bindgen/requirements.html#windows) and `lib.exe` (provided by Visual Studio)
* On Linux, install `clang` and `ar`

## Distributing Your App
* On Windows, copy `DLSS_SDK/lib/Windows_x86_64/rel/nvngx_dlss.dll` to the same directory as your app
* On Linux, copy `DLSS_SDK/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.310.1.0` to the same directory as your app
* Include the full copyright and license blurb texts from section `9.5` of `DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf` with your app
