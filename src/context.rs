use crate::nvsdk_ngx::*;
use crate::DlssSdk;
use glam::UVec2;
use std::ops::Deref;
use std::ptr;
use wgpu::{Adapter, CommandEncoder, Device, TextureUsages};
use wgpu_core::api::Vulkan;
use wgpu_hal::vulkan::conv::map_subresource_range;

pub struct DlssContext<'a, D: Deref<Target = Device> + Clone> {
    upscaled_resolution: UVec2,
    min_render_resolution: UVec2,
    max_render_resolution: UVec2,
    sdk: &'a DlssSdk<D>,
    feature: *mut NVSDK_NGX_Handle,
}

impl<'a, D: Deref<Target = Device> + Clone> DlssContext<'a, D> {
    pub fn new<'b>(
        upscaled_resolution: UVec2,
        preset: DlssPreset,
        mut feature_flags: DlssFeatureFlags,
        sdk: &'a mut DlssSdk<D>,
        command_encoder: &'b mut CommandEncoder,
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
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                let mut feature = ptr::null_mut();
                check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                    command_encoder.unwrap().raw_handle(),
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
            })
        }
    }

    pub fn render(
        &mut self,
        render_parameters: DlssRenderParameters,
        command_encoder: &mut CommandEncoder,
    ) -> Result<(), DlssError> {
        let mut dlss_eval_params = NVSDK_NGX_VK_DLSS_Eval_Params {
            Feature: NVSDK_NGX_VK_Feature_Eval_Params {
                pInColor: todo!(),
                pInOutput: todo!(),
                InSharpness: todo!(),
            },
            pInDepth: todo!(),
            pInMotionVectors: todo!(),
            InJitterOffsetX: todo!(),
            InJitterOffsetY: todo!(),
            InRenderSubrectDimensions: todo!(),
            InReset: todo!(),
            InMVScaleX: todo!(),
            InMVScaleY: todo!(),
            pInTransparencyMask: todo!(),
            pInExposureTexture: todo!(),
            pInBiasCurrentColorMask: todo!(),
            InColorSubrectBase: todo!(),
            InDepthSubrectBase: todo!(),
            InMVSubrectBase: todo!(),
            InTranslucencySubrectBase: todo!(),
            InBiasCurrentColorSubrectBase: todo!(),
            InOutputSubrectBase: todo!(),
            InPreExposure: todo!(),
            InExposureScale: todo!(),
            InIndicatorInvertXAxis: todo!(),
            InIndicatorInvertYAxis: todo!(),
            GBufferSurface: todo!(),
            InToneMapperType: todo!(),
            pInMotionVectors3D: todo!(),
            pInIsParticleMask: todo!(),
            pInAnimatedTextureMask: todo!(),
            pInDepthHighRes: todo!(),
            pInPositionViewSpace: todo!(),
            InFrameTimeDeltaInMsec: todo!(),
            pInRayTracingHitDistance: todo!(),
            pInMotionVectorsReflections: todo!(),
        };

        unsafe {
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_EVALUATE_DLSS_EXT(
                    command_encoder.unwrap().raw_handle(),
                    self.feature,
                    self.sdk.parameters,
                    &mut dlss_eval_params,
                ))
            })
        }
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

fn dlss_resource(texture: &DlssTexture, adapter: &Adapter) -> NVSDK_NGX_Resource_VK {
    unsafe {
        NVSDK_NGX_Create_ImageView_Resource_VK(
            texture
                .view
                .as_hal::<Vulkan, _, _>(|v| v.unwrap().raw_handle()),
            texture
                .texture
                .as_hal::<Vulkan, _, _>(|t| t.unwrap().raw_handle()),
            map_subresource_range(&texture.subresource_range, texture.texture.format()),
            adapter
                .texture_format_as_hal::<Vulkan>(texture.texture.format())
                .unwrap(),
            texture.texture.width(),
            texture.texture.height(),
            texture.usages == TextureUsages::STORAGE_BINDING,
        )
    }
}
