use crate::{
    DlssExposure, DlssRenderParameters, DlssSdk, nvsdk_ngx::*,
    render_parameters::texture_to_ngx_resource,
};
use glam::{UVec2, Vec2};
use std::{
    iter,
    ops::RangeInclusive,
    ptr,
    sync::{Arc, Mutex},
};
use wgpu::{Adapter, CommandEncoder, CommandEncoderDescriptor, Device, Queue, hal::api::Vulkan};

/// Camera-specific object for using DLSS.
pub struct DlssContext {
    upscaled_resolution: UVec2,
    min_render_resolution: UVec2,
    max_render_resolution: UVec2,
    device: Device,
    sdk: Arc<Mutex<DlssSdk>>,
    feature: *mut NVSDK_NGX_Handle,
}

impl DlssContext {
    /// Create a new [`DlssContext`].
    ///
    /// This is an expensive operation. The resulting context should be cached, and only recreated when settings change.
    pub fn new(
        upscaled_resolution: UVec2,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: DlssFeatureFlags,
        sdk: Arc<Mutex<DlssSdk>>,
        device: &Device,
        queue: &Queue,
    ) -> Result<Self, DlssError> {
        let locked_sdk = sdk.lock().unwrap();

        let perf_quality_value = perf_quality_mode.as_perf_quality_value(upscaled_resolution);

        let mut optimal_render_resolution = UVec2::ZERO;
        let mut min_render_resolution = UVec2::ZERO;
        let mut max_render_resolution = UVec2::ZERO;
        unsafe {
            let mut deprecated_sharpness = 0.0f32;
            check_ngx_result(NGX_DLSS_GET_OPTIMAL_SETTINGS(
                locked_sdk.parameters,
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
        if perf_quality_mode == DlssPerfQualityMode::Dlaa {
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
            InFeatureCreateFlags: feature_flags.as_flags(),
            InEnableOutputSubrects: feature_flags.contains(DlssFeatureFlags::OutputSubrect),
        };

        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlss_context_creation"),
        });

        let mut feature = ptr::null_mut();
        unsafe {
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                    command_encoder.unwrap().raw_handle(),
                    1,
                    1,
                    &mut feature,
                    locked_sdk.parameters,
                    &mut dlss_create_params,
                ))
            })?
        }

        queue.submit([command_encoder.finish()]);

        Ok(Self {
            upscaled_resolution,
            min_render_resolution,
            max_render_resolution,
            device: device.clone(),
            sdk: Arc::clone(&sdk),
            feature,
        })
    }

    /// Encode commands to render DLSS.
    pub fn render(
        &mut self,
        render_parameters: DlssRenderParameters,
        command_encoder: &mut CommandEncoder,
        adapter: &Adapter,
    ) -> Result<(), DlssError> {
        render_parameters.validate()?;

        let sdk = self.sdk.lock().unwrap();

        let partial_texture_size = render_parameters
            .partial_texture_size
            .unwrap_or(self.max_render_resolution);

        let (exposure, exposure_scale, pre_exposure) = match &render_parameters.exposure {
            DlssExposure::Manual {
                exposure,
                exposure_scale,
                pre_exposure,
            } => (
                &mut texture_to_ngx_resource(exposure, adapter) as *mut _,
                exposure_scale.unwrap_or(1.0),
                pre_exposure.unwrap_or(0.0),
            ),
            DlssExposure::Automatic => (ptr::null_mut(), 0.0, 0.0),
        };

        let mut dlss_eval_params = NVSDK_NGX_VK_DLSS_Eval_Params {
            Feature: NVSDK_NGX_VK_Feature_Eval_Params {
                pInColor: &mut texture_to_ngx_resource(render_parameters.color, adapter) as *mut _,
                pInOutput: &mut texture_to_ngx_resource(render_parameters.dlss_output, adapter)
                    as *mut _,
                InSharpness: 0.0,
            },
            pInDepth: &mut texture_to_ngx_resource(render_parameters.depth, adapter) as *mut _,
            pInMotionVectors: &mut texture_to_ngx_resource(
                render_parameters.motion_vectors,
                adapter,
            ) as *mut _,
            InJitterOffsetX: render_parameters.jitter_offset.x,
            InJitterOffsetY: render_parameters.jitter_offset.y,
            InRenderSubrectDimensions: NVSDK_NGX_Dimensions {
                Width: partial_texture_size.x,
                Height: partial_texture_size.y,
            },
            InReset: render_parameters.reset as _,
            InMVScaleX: render_parameters.motion_vector_scale.unwrap_or(Vec2::ONE).x,
            InMVScaleY: render_parameters.motion_vector_scale.unwrap_or(Vec2::ONE).y,
            pInTransparencyMask: ptr::null_mut(),
            pInExposureTexture: exposure,
            pInBiasCurrentColorMask: match &render_parameters.bias {
                Some(bias) => &mut texture_to_ngx_resource(bias, adapter) as *mut _,
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

        command_encoder.transition_resources(iter::empty(), render_parameters.barrier_list());
        unsafe {
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_EVALUATE_DLSS_EXT(
                    command_encoder.unwrap().raw_handle(),
                    self.feature,
                    sdk.parameters,
                    &mut dlss_eval_params,
                ))
            })
        }
    }

    /// Suggested subpixel camera jitter for a given frame.
    pub fn suggested_jitter(&self, frame_number: u32, render_resolution: UVec2) -> Vec2 {
        let ratio = self.upscaled_resolution.x as f32 / render_resolution.x as f32;
        let phase_count = (8.0 * ratio * ratio) as u32;
        let i = frame_number % phase_count;

        Vec2 {
            x: halton_sequence(i, 2),
            y: halton_sequence(i, 3),
        } - 0.5
    }

    /// Suggested mip bias for sampling textures.
    pub fn suggested_mip_bias(&self, render_resolution: UVec2) -> f32 {
        (render_resolution.x as f32 / self.upscaled_resolution.x as f32).log2() - 1.0
    }

    /// The upscaled resolution DLSS will output at.
    pub fn upscaled_resolution(&self) -> UVec2 {
        self.upscaled_resolution
    }

    /// The resolution the camera should render at, pre-upscaling.
    pub fn render_resolution(&self) -> UVec2 {
        self.min_render_resolution
    }

    /// Like [`Self::render_resolution`], but returns a range of values for use with dynamic resolution scaling.
    pub fn render_resolution_range(&self) -> RangeInclusive<UVec2> {
        self.min_render_resolution..=self.max_render_resolution
    }
}

impl Drop for DlssContext {
    fn drop(&mut self) {
        unsafe {
            let hal_device = self.device.as_hal::<Vulkan>().unwrap();
            hal_device
                .raw_device()
                .device_wait_idle()
                .expect("Failed to wait for idle device when destroying DlssContext");

            check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature))
                .expect("Failed to destroy DlssContext feature");
        }
    }
}

unsafe impl Send for DlssContext {}
unsafe impl Sync for DlssContext {}

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
