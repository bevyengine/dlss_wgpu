# dlss_wgpu
### Deep Learning Super Sampling for wgpu

## Downloading The DLSS SDK
The DLSS SDK cannot be redistributed by this crate.
You will need to create a NVIDIA developer account and download the SDK as follows:
* Install Clang
* Download the NVIDIA DLSS Super Resolution SDK v3.1 from https://developer.nvidia.com/rtx/dlss/get-started#sdk-version
* Set the environment variable `DLSS_SDK = /path/to/dlss_sdk/dlssv3.1.0/nvngx_dlss_sdk`
* Ensure you comply with the DLSS SDK license located at `$DLSS_SDK/LICENSE.txt`
