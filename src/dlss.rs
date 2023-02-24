#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

type PFN_vkGetDeviceProcAddr = ash::vk::PFN_vkGetDeviceProcAddr;
type PFN_vkGetInstanceProcAddr = ash::vk::PFN_vkGetInstanceProcAddr;
type VkBuffer = ash::vk::Buffer;
type VkCommandBuffer = ash::vk::CommandBuffer;
type VkDevice = ash::vk::Device;
type VkExtensionProperties = ash::vk::ExtensionProperties;
type VkFormat = ash::vk::Format;
type VkImage = ash::vk::Image;
type VkImageSubresourceRange = ash::vk::ImageSubresourceRange;
type VkImageView = ash::vk::ImageView;
type VkInstance = ash::vk::Instance;
type VkPhysicalDevice = ash::vk::PhysicalDevice;

use std::ffi::OsStr;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum DLSSPreset {
    #[default]
    Auto,
    Native,
    UltraQuality,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

bitflags::bitflags! {
    pub struct DLSSFeatureFlags: NVSDK_NGX_DLSS_Feature_Flags {
        const HighDynamicRange = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        const LowResolutionMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        const JitteredMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
        const InvertedDepth = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        const AutoExposure = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        const PartialTextureInputs = 1 << 7;
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DLSSError {
    #[error("TODO")]
    FeatureNotSupported,
    #[error("TODO")]
    PlatformError,
    #[error("TODO")]
    FeatureAlreadyExists,
    #[error("TODO")]
    FeatureNotFound,
    #[error("TODO")]
    InvalidParameters,
    #[error("TODO")]
    ScratchBufferTooSmall,
    #[error("TODO")]
    NotInitialized,
    #[error("TODO")]
    UnsupportedInputFormat,
    #[error("TODO")]
    RWFlagMissing,
    #[error("TODO")]
    MissingInput,
    #[error("TODO")]
    UnableToInitializeFeature,
    #[error("TODO")]
    OutOfDate,
    #[error("TODO")]
    OutOfGPUMemory,
    #[error("TODO")]
    UnsupportedFormat,
    #[error("TODO")]
    UnableToWriteToAppDataPath,
    #[error("TODO")]
    UnsupportedParameter,
    #[error("TODO")]
    Denied,
    #[error("TODO")]
    NotImplemented,
}

pub fn check_ngx_result(result: NVSDK_NGX_Result) -> Result<(), DLSSError> {
    match result {
        NVSDK_NGX_Result_Success => Ok(()),
        NVSDK_NGX_Result_FAIL_FeatureNotSupported => Err(DLSSError::FeatureNotSupported),
        NVSDK_NGX_RESULT_FAIL_PlatformError => Err(DLSSError::PlatformError),
        NVSDK_NGX_RESULT_FAIL_FeatureAlreadyExists => Err(DLSSError::FeatureAlreadyExists),
        NVSDK_NGX_RESULT_FAIL_FeatureNotFound => Err(DLSSError::FeatureNotFound),
        NVSDK_NGX_RESULT_FAIL_InvalidParameters => Err(DLSSError::InvalidParameters),
        NVSDK_NGX_RESULT_FAIL_ScratchBufferTooSmall => Err(DLSSError::ScratchBufferTooSmall),
        NVSDK_NGX_RESULT_FAIL_NotInitialized => Err(DLSSError::NotInitialized),
        NVSDK_NGX_RESULT_FAIL_UnsupportedInputFormat => Err(DLSSError::UnsupportedInputFormat),
        NVSDK_NGX_RESULT_FAIL_RWFlagMissing => Err(DLSSError::RWFlagMissing),
        NVSDK_NGX_RESULT_FAIL_MissingInput => Err(DLSSError::MissingInput),
        NVSDK_NGX_RESULT_FAIL_UnableToInitializeFeature => {
            Err(DLSSError::UnableToInitializeFeature)
        }
        NVSDK_NGX_RESULT_FAIL_OutOfDate => Err(DLSSError::OutOfDate),
        NVSDK_NGX_RESULT_FAIL_OutOfGPUMemory => Err(DLSSError::OutOfGPUMemory),
        NVSDK_NGX_RESULT_FAIL_UnsupportedFormat => Err(DLSSError::UnsupportedFormat),
        NVSDK_NGX_RESULT_FAIL_UnableToWriteToAppDataPath => {
            Err(DLSSError::UnableToWriteToAppDataPath)
        }
        NVSDK_NGX_RESULT_FAIL_UnsupportedParameter => Err(DLSSError::UnsupportedParameter),
        NVSDK_NGX_RESULT_FAIL_Denied => Err(DLSSError::Denied),
        NVSDK_NGX_RESULT_FAIL_NotImplemented => Err(DLSSError::NotImplemented),
        _ => unreachable!(),
    }
}

#[cfg(target_os = "windows")]
pub fn os_str_to_wchar(s: &OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;

    s.encode_wide().collect()
}

#[cfg(not(target_os = "windows"))]
pub fn os_str_to_wchar(s: &OsStr) -> Vec<u32> {
    s.to_str().unwrap_or("").chars().map(|c| c as u32).collect()
}
