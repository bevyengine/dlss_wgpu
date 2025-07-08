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

use glam::UVec2;

/// How much DLSS should upscale by.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub enum DlssPerfQualityMode {
    /// Let DLSS decide.
    #[default]
    Auto,
    /// Anti-aliasing only, no upscaling.
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl DlssPerfQualityMode {
    pub(crate) fn as_perf_quality_value(
        &self,
        upscaled_resolution: UVec2,
    ) -> NVSDK_NGX_PerfQuality_Value {
        match self {
            Self::Auto => {
                let mega_pixels =
                    (upscaled_resolution.x * upscaled_resolution.y) as f32 / 1_000_000.0;

                if mega_pixels < 2.03 {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA
                } else if mega_pixels < 3.68 {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
                } else if mega_pixels < 8.29 {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
                } else {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
                }
            }
            Self::Dlaa => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA,
            Self::Quality => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality,
            Self::Balanced => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced,
            Self::Performance => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf,
            Self::UltraPerformance => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
            }
        }
    }
}

bitflags::bitflags! {
    /// Flags for creating a [`crate::DlssContext`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DlssFeatureFlags: NVSDK_NGX_DLSS_Feature_Flags {
        /// Use an HDR texture for [`crate::DlssRenderParameters::color`] instead of an SDR texture.
        const HighDynamicRange = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        /// Motion vector values in [`crate::DlssRenderParameters::motion_vectors`] are at the upscaled resolution,
        /// instead of render resolution.
        const LowResolutionMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
        /// Motion vector values in [`crate::DlssRenderParameters::motion_vectors`] contain jitter.
        const JitteredMotionVectors = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
        /// Camera is using a reverse depth buffer for [`crate::DlssRenderParameters::depth`].
        const InvertedDepth = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        /// Have DLSS apply auto-exposure.
        const AutoExposure = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        /// Use a 4 channel RGBA texture for [`crate::DlssRenderParameters::color`] instead of a 3 channel RGB texture.
        const AlphaUpscaling = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling;
        /// Allow DLSS to write to a subrect of [`crate::DlssRenderParameters::dlss_output`].
        const OutputSubrect = 256; // Not part of NVSDK_NGX_DLSS_Feature_Flags
    }
}

impl DlssFeatureFlags {
    pub(crate) fn as_flags(&self) -> NVSDK_NGX_DLSS_Feature_Flags {
        let mut flags = self.clone();
        flags.remove(DlssFeatureFlags::OutputSubrect);
        flags.bits()
    }
}

/// Errors thrown by DLSS.
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
