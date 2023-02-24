mod dlss;

pub use dlss::{DLSSError, DLSSFeatureFlags, DLSSPreset};

use dlss::*;
use glam::UVec2;
use std::env;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::os::raw::c_int;
use std::ptr;
use wgpu::{CommandEncoder, Device};
use wgpu_core::api::Vulkan;

pub struct DLSSSDK<D: Deref<Target = Device> + Clone> {
    parameters: *mut NVSDK_NGX_Parameter,
    device: D,
}

impl<D: Deref<Target = Device> + Clone> DLSSSDK<D> {
    pub fn new(application_id: Option<u64>, device: D) -> Result<Self, DLSSError> {
        let sdk_info = NVSDK_NGX_FeatureCommonInfo {
            // TODO: Allow passing list of extra DLSS shared library paths
            PathListInfo: NVSDK_NGX_PathListInfo {
                Path: ptr::null(),
                Length: 0,
            },
            InternalData: ptr::null_mut(),
            LoggingInfo: NVSDK_NGX_LoggingInfo {
                LoggingCallback: None,
                MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_OFF,
                DisableOtherLoggingSinks: false,
            },
        };

        unsafe {
            let (vk_instance, vk_physical_device, vk_device, vk_gipa, vk_gdpa) = device
                .as_hal::<Vulkan, _, _>(|device| {
                    let device = device.unwrap();
                    let shared_instance = device.shared_instance();
                    let raw_instance = shared_instance.raw_instance();
                    (
                        raw_instance.handle(),
                        device.raw_physical_device(),
                        device.raw_device().handle(),
                        shared_instance.entry().static_fn().get_instance_proc_addr,
                        raw_instance.fp_v1_0().get_device_proc_addr,
                    )
                });

            check_ngx_result(NVSDK_NGX_VULKAN_Init(
                application_id.unwrap_or(0),
                os_str_to_wchar(env::temp_dir().as_os_str()).as_ptr() as *const _,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_gipa,
                vk_gdpa,
                &sdk_info as *const _,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            ))?;

            let mut parameters = MaybeUninit::<*mut NVSDK_NGX_Parameter>::uninit();
            check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(
                parameters.as_mut_ptr(),
            ))?;
            let parameters = parameters.assume_init();

            let mut dlss_supported = MaybeUninit::<c_int>::uninit();
            NVSDK_NGX_Parameter_GetI(
                parameters,
                NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                dlss_supported.as_mut_ptr(),
            );
            let dlss_supported = dlss_supported.assume_init();
            if dlss_supported == 0 {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                return Err(DLSSError::FeatureNotSupported);
            }

            Ok(Self { parameters, device })
        }
    }
}

impl<D: Deref<Target = Device> + Clone> Drop for DLSSSDK<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap().raw_device();

                device
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DLSSSDK");

                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(self.parameters))
                    .expect("Failed to destroy DLSSSDK parameters");
                check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(device.handle()))
                    .expect("Failed to destroy DLSSSDK");
            });
        }
    }
}

// TODO: Rather than clone device, ensure context does not live longer than sdk?
pub struct DLSSContext<D: Deref<Target = Device> + Clone> {
    upscaled_resolution: UVec2,
    min_render_resolution: UVec2,
    max_render_resolution: UVec2,
    feature: *mut NVSDK_NGX_Handle,
    device: D,
}

impl<D: Deref<Target = Device> + Clone> DLSSContext<D> {
    pub fn new(
        upscaled_resolution: UVec2,
        preset: DLSSPreset,
        feature_flags: DLSSFeatureFlags,
        sdk: &mut DLSSSDK<D>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<Self, DLSSError> {
        let enable_output_subrects = feature_flags.contains(DLSSFeatureFlags::PartialTextureInputs);
        feature_flags.remove(DLSSFeatureFlags::PartialTextureInputs);

        let perf_quality_value = match preset {
            DLSSPreset::Auto => {
                let mega_pixels =
                    (upscaled_resolution.x * upscaled_resolution.y) as f32 / 1_000_000.0;

                if mega_pixels < 3.68 {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
                } else if mega_pixels < 8.29 {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
                } else {
                    NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
                }
            }
            DLSSPreset::Native => {
                // Doesn't really matter
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraQuality
            }
            DLSSPreset::UltraQuality => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraQuality
            }
            DLSSPreset::Quality => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
            }
            DLSSPreset::Balanced => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced
            }
            DLSSPreset::Performance => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
            }
            DLSSPreset::UltraPerformance => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
            }
        };

        let mut optimal_render_resolution = UVec2::ZERO;
        let mut min_render_resolution = UVec2::ZERO;
        let mut max_render_resolution = UVec2::ZERO;
        unsafe {
            let mut deprecated_sharpness = 0.0f32;
            check_ngx_result(NGX_DLSS_GET_OPTIMAL_SETTINGS(
                sdk.parameters,
                upscaled_resolution.x,
                upscaled_resolution.y,
                perf_quality_value,
                &mut optimal_render_resolution.x as *mut _,
                &mut optimal_render_resolution.y as *mut _,
                &mut max_render_resolution.x as *mut _,
                &mut max_render_resolution.y as *mut _,
                &mut min_render_resolution.x as *mut _,
                &mut min_render_resolution.y as *mut _,
                &mut deprecated_sharpness as *mut _,
            ))?;
        }
        if preset == DLSSPreset::Native {
            optimal_render_resolution = upscaled_resolution;
            min_render_resolution = upscaled_resolution;
            max_render_resolution = upscaled_resolution;
        }

        let mut dlss_create_params = NVSDK_NGX_DLSS_Create_Params {
            Feature: NVSDK_NGX_Feature_Create_Params {
                InWidth: optimal_render_resolution.x,
                InHeight: optimal_render_resolution.y,
                InTargetWidth: upscaled_resolution.x,
                InTargetHeight: upscaled_resolution.y,
                InPerfQualityValue: perf_quality_value,
            },
            InFeatureCreateFlags: feature_flags.bits(),
            InEnableOutputSubrects: enable_output_subrects,
        };

        unsafe {
            let mut feature = MaybeUninit::<*mut NVSDK_NGX_Handle>::uninit();
            check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                todo!("Command buffer"),
                1,
                1,
                feature.as_mut_ptr(),
                sdk.parameters,
                &mut dlss_create_params as *mut _,
            ))?;
            let feature = feature.assume_init();

            Ok(Self {
                upscaled_resolution,
                min_render_resolution,
                max_render_resolution,
                feature,
                device: sdk.device.clone(),
            })
        }
    }

    pub fn render(&mut self) {
        todo!()
    }

    pub fn upscaled_resolution(&self) -> UVec2 {
        self.upscaled_resolution
    }

    pub fn min_render_resolution(&self) -> UVec2 {
        self.min_render_resolution
    }

    pub fn max_render_resolution(&self) -> UVec2 {
        self.max_render_resolution
    }
}

impl<D: Deref<Target = Device> + Clone> Drop for DLSSContext<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                device
                    .unwrap()
                    .raw_device()
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DLSSContext");

                check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature))
                    .expect("Failed to destroy DLSSContext feature");
            });
        }
    }
}
