use crate::nvsdk_ngx::*;
use crate::DlssSdk;
use glam::UVec2;
use std::ops::Deref;
use std::ptr;
use wgpu::{CommandEncoder, Device};
use wgpu_core::api::Vulkan;

pub struct DlssContext<'a, D: Deref<Target = Device> + Clone> {
    pub upscaled_resolution: UVec2,
    pub min_render_resolution: UVec2,
    pub max_render_resolution: UVec2,
    sdk: &'a DlssSdk<D>,
    feature: *mut NVSDK_NGX_Handle,
}

impl<'a, D: Deref<Target = Device> + Clone> DlssContext<'a, D> {
    pub fn new<'f>(
        upscaled_resolution: UVec2,
        preset: DlssPreset,
        mut feature_flags: DlssFeatureFlags,
        sdk: &'a mut DlssSdk<D>,
        command_encoder: &'f mut CommandEncoder,
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
                &mut optimal_render_resolution.x,
                &mut optimal_render_resolution.y,
                &mut max_render_resolution.x,
                &mut max_render_resolution.y,
                &mut min_render_resolution.x,
                &mut min_render_resolution.y,
                &mut deprecated_sharpness,
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
                sdk,
                feature,
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

impl<D: Deref<Target = Device> + Clone> Drop for DlssContext<'_, D> {
    fn drop(&mut self) {
        unsafe {
            self.sdk.device.as_hal::<Vulkan, _, _>(|device| {
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
