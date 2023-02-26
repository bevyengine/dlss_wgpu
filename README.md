# dlss_wgpu
### Deep Learning Super Sampling for wgpu

## Downloading The DLSS SDK
The DLSS SDK cannot be redistributed by this crate. You will need to download the SDK as follows:
* On Linux, install `clang` and `ar`
* On Windows, install `clang` and `lib` (provided by Visual Studio)
* Clone the NVIDIA DLSS Super Resolution SDK v3.1 from https://github.com/NVIDIA/DLSS/tree/v3.1.0
* Set the environment variable `DLSS_SDK = /path/to/dlss_sdk/DLSS`
* Ensure you comply with the DLSS SDK license located at https://github.com/NVIDIA/DLSS/blob/v3.1.0/LICENSE.txt
