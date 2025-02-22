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

use glam::{UVec2, Vec2};
use std::ffi::OsStr;
use wgpu::{ImageSubresourceRange, Texture, TextureUsages, TextureView};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub enum DlssPreset {
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
    pub struct DlssFeatureFlags: NVSDK_NGX_DLSS_Feature_Flags {
        const HighDynamicRange = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        const LowResolutionMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        const JitteredMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
        const InvertedDepth = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        const AutoExposure = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        const PartialTextureInputs = 1 << 7;
    }
}

// TODO: Allow configuring partial texture origins
pub struct DlssRenderParameters<'a> {
    pub color: DlssTexture<'a>,
    pub depth: DlssTexture<'a>,
    pub motion_vectors: DlssTexture<'a>,
    pub exposure: DlssExposure<'a>,
    pub transparency_mask: Option<DlssTexture<'a>>,
    pub bias: Option<DlssTexture<'a>>,
    pub dlss_output: DlssTexture<'a>,

    pub reset: bool,
    pub jitter_offset: Vec2,
    pub partial_texture_size: Option<UVec2>,
    pub motion_vector_scale: Option<Vec2>,
}

pub enum DlssExposure<'a> {
    Manual {
        exposure: DlssTexture<'a>,
        exposure_scale: Option<f32>,
        pre_exposure: Option<f32>,
    },
    Automatic,
}

pub struct DlssTexture<'a> {
    pub texture: &'a Texture,
    pub view: &'a TextureView,
    pub subresource_range: ImageSubresourceRange,
    pub usages: TextureUsages,
}

#[derive(thiserror::Error, Debug)]
pub enum DlssRequestDeviceError {
    #[error(transparent)]
    WgpuRequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    DeviceError(#[from] wgpu::hal::DeviceError),
    #[error(transparent)]
    VulkanError(#[from] ash::vk::Result),
    #[error(transparent)]
    DlssError(#[from] DlssError),
}

#[derive(thiserror::Error, Debug)]
pub enum DlssError {
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

pub fn check_ngx_result(result: NVSDK_NGX_Result) -> Result<(), DlssError> {
    match result {
        NVSDK_NGX_Result_Success => Ok(()),
        NVSDK_NGX_Result_FAIL_FeatureNotSupported => Err(DlssError::FeatureNotSupported),
        NVSDK_NGX_RESULT_FAIL_PlatformError => Err(DlssError::PlatformError),
        NVSDK_NGX_RESULT_FAIL_FeatureAlreadyExists => Err(DlssError::FeatureAlreadyExists),
        NVSDK_NGX_RESULT_FAIL_FeatureNotFound => Err(DlssError::FeatureNotFound),
        NVSDK_NGX_RESULT_FAIL_InvalidParameters => Err(DlssError::InvalidParameters),
        NVSDK_NGX_RESULT_FAIL_ScratchBufferTooSmall => Err(DlssError::ScratchBufferTooSmall),
        NVSDK_NGX_RESULT_FAIL_NotInitialized => Err(DlssError::NotInitialized),
        NVSDK_NGX_RESULT_FAIL_UnsupportedInputFormat => Err(DlssError::UnsupportedInputFormat),
        NVSDK_NGX_RESULT_FAIL_RWFlagMissing => Err(DlssError::RWFlagMissing),
        NVSDK_NGX_RESULT_FAIL_MissingInput => Err(DlssError::MissingInput),
        NVSDK_NGX_RESULT_FAIL_UnableToInitializeFeature => {
            Err(DlssError::UnableToInitializeFeature)
        }
        NVSDK_NGX_RESULT_FAIL_OutOfDate => Err(DlssError::OutOfDate),
        NVSDK_NGX_RESULT_FAIL_OutOfGPUMemory => Err(DlssError::OutOfGPUMemory),
        NVSDK_NGX_RESULT_FAIL_UnsupportedFormat => Err(DlssError::UnsupportedFormat),
        NVSDK_NGX_RESULT_FAIL_UnableToWriteToAppDataPath => {
            Err(DlssError::UnableToWriteToAppDataPath)
        }
        NVSDK_NGX_RESULT_FAIL_UnsupportedParameter => Err(DlssError::UnsupportedParameter),
        NVSDK_NGX_RESULT_FAIL_Denied => Err(DlssError::Denied),
        NVSDK_NGX_RESULT_FAIL_NotImplemented => Err(DlssError::NotImplemented),
        _ => unreachable!(),
    }
}
