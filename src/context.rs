use crate::nvsdk_ngx::*;
use crate::DlssSdk;
use glam::{UVec2, Vec2};
use std::ops::{Deref, RangeInclusive};
use std::ptr;
use std::rc::Rc;
use wgpu::{Adapter, CommandEncoder, Device, TextureUsages};
use wgpu_core::api::Vulkan;
use wgpu_hal::vulkan::conv::map_subresource_range;

pub struct DlssContext<D: Deref<Target = Device>> {
    upscaled_resolution: UVec2,
    min_render_resolution: UVec2,
    max_render_resolution: UVec2,
    sdk: Rc<DlssSdk<D>>,
    feature: *mut NVSDK_NGX_Handle,
}

impl<D: Deref<Target = Device>> DlssContext<D> {
    pub fn new(
        upscaled_resolution: UVec2,
        preset: DlssPreset,
        mut feature_flags: DlssFeatureFlags,
        sdk: &Rc<DlssSdk<D>>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<Self, DlssError> {
        let sdk = Rc::clone(sdk);

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
        adapter: &Adapter,
    ) -> Result<(), DlssError> {
        // TODO: Validate render_parameters

        let partial_texture_size = render_parameters
            .partial_texture_size
            .unwrap_or(self.max_render_resolution);

        let (exposure, exposure_scale, pre_exposure) = match render_parameters.exposure {
            DlssExposure::Manual {
                exposure,
                exposure_scale,
                pre_exposure,
            } => (
                &mut dlss_resource(&exposure, adapter) as *mut _,
                exposure_scale.unwrap_or(1.0),
                pre_exposure.unwrap_or(0.0),
            ),
            DlssExposure::Automatic => (ptr::null_mut(), 0.0, 0.0),
        };

        let mut dlss_eval_params = NVSDK_NGX_VK_DLSS_Eval_Params {
            Feature: NVSDK_NGX_VK_Feature_Eval_Params {
                pInColor: &mut dlss_resource(&render_parameters.color, adapter),
                pInOutput: &mut dlss_resource(&render_parameters.dlss_output, adapter),
                InSharpness: 0.0,
            },
            pInDepth: &mut dlss_resource(&render_parameters.depth, adapter),
            pInMotionVectors: &mut dlss_resource(&render_parameters.motion_vectors, adapter),
            InJitterOffsetX: render_parameters.jitter_offset.x,
            InJitterOffsetY: render_parameters.jitter_offset.y,
            InRenderSubrectDimensions: NVSDK_NGX_Dimensions {
                Width: partial_texture_size.x,
                Height: partial_texture_size.y,
            },
            InReset: render_parameters.reset as _,
            InMVScaleX: render_parameters.motion_vector_scale.unwrap_or(Vec2::ONE).x,
            InMVScaleY: render_parameters.motion_vector_scale.unwrap_or(Vec2::ONE).y,
            pInTransparencyMask: match render_parameters.transparency_mask {
                Some(transparency_mask) => &mut dlss_resource(&transparency_mask, adapter),
                None => ptr::null_mut(),
            },
            pInExposureTexture: exposure,
            pInBiasCurrentColorMask: match render_parameters.bias {
                Some(bias) => &mut dlss_resource(&bias, adapter),
                None => ptr::null_mut(),
            },
            InColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDepthSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InMVSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InTranslucencySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InBiasCurrentColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InOutputSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InPreExposure: pre_exposure,
            InExposureScale: exposure_scale,
            InIndicatorInvertXAxis: 0,
            InIndicatorInvertYAxis: 0,
            GBufferSurface: NVSDK_NGX_VK_GBuffer {
                pInAttrib: [ptr::null_mut(); 16],
            },
            InToneMapperType: NVSDK_NGX_ToneMapperType_NVSDK_NGX_TONEMAPPER_STRING,
            pInMotionVectors3D: ptr::null_mut(),
            pInIsParticleMask: ptr::null_mut(),
            pInAnimatedTextureMask: ptr::null_mut(),
            pInDepthHighRes: ptr::null_mut(),
            pInPositionViewSpace: ptr::null_mut(),
            InFrameTimeDeltaInMsec: 0.0,
            pInRayTracingHitDistance: ptr::null_mut(),
            pInMotionVectorsReflections: ptr::null_mut(),
        };

        command_encoder.push_debug_group("dlss");
        let result = unsafe {
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_EVALUATE_DLSS_EXT(
                    command_encoder.unwrap().raw_handle(),
                    self.feature,
                    self.sdk.parameters,
                    &mut dlss_eval_params,
                ))
            })
        };
        command_encoder.pop_debug_group();
        result
    }

    pub fn suggested_jitter(&self, frame_count: u32, render_resolution: UVec2) -> Vec2 {
        let ratio = self.upscaled_resolution.x as f32 / render_resolution.x as f32;
        let phase_count = (8.0 * ratio * ratio) as u32;
        let i = (frame_count % phase_count) + 1;

        Vec2 {
            x: halton_sequence(i, 2),
            y: halton_sequence(i, 3),
        } - 0.5
    }

    pub fn upscaled_resolution(&self) -> UVec2 {
        self.upscaled_resolution
    }

    pub fn render_resolution(&self) -> UVec2 {
        self.min_render_resolution
    }

    pub fn render_resolution_range(&self) -> RangeInclusive<UVec2> {
        self.min_render_resolution..=self.max_render_resolution
    }
}

impl<D: Deref<Target = Device>> Drop for DlssContext<D> {
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
            texture.usages.contains(TextureUsages::STORAGE_BINDING),
        )
    }
}

fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut result = 0.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index = (index as f32 / base as f32).floor() as u32;
    }
    result
}
