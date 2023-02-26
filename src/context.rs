use crate::nvsdk_ngx::*;
use crate::DlssSdk;
use glam::UVec2;
use std::ops::Deref;
use std::ptr;
use wgpu::{CommandEncoder, Device};
use wgpu_core::api::Vulkan;

// TODO: Rather than clone device, ensure context does not live longer than sdk?
pub struct DlssContext<D: Deref<Target = Device> + Clone> {
    upscaled_resolution: UVec2,
    min_render_resolution: UVec2,
    max_render_resolution: UVec2,
    feature: *mut NVSDK_NGX_Handle,
    device: D,
}

impl<D: Deref<Target = Device> + Clone> DlssContext<D> {
    pub fn new(
        upscaled_resolution: UVec2,
        preset: DlssPreset,
        mut feature_flags: DlssFeatureFlags,
        sdk: &mut DlssSdk<D>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<Self, DlssError> {
        let enable_output_subrects = feature_flags.contains(DlssFeatureFlags::PartialTextureInputs);
        feature_flags.remove(DlssFeatureFlags::PartialTextureInputs);

        let perf_quality_value = match preset {
            DlssPreset::Auto => {
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
            DlssPreset::Native => {
                // Doesn't really matter
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraQuality
            }
            DlssPreset::UltraQuality => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraQuality
            }
            DlssPreset::Quality => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality
            }
            DlssPreset::Balanced => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced
            }
            DlssPreset::Performance => {
                NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf
            }
            DlssPreset::UltraPerformance => {
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
        if preset == DlssPreset::Native {
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
            let feature = ptr::null_mut();
            check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                todo!("Command buffer"),
                1,
                1,
                &mut feature,
                sdk.parameters,
                &mut dlss_create_params,
            ))?;

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

impl<D: Deref<Target = Device> + Clone> Drop for DlssContext<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                device
                    .unwrap()
                    .raw_device()
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DlssContext");

                check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature))
                    .expect("Failed to destroy DlssContext feature");
            });
        }
    }
}
