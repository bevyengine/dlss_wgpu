use crate::{DlssSdk, nvsdk_ngx::*};
use glam::{Mat4, UVec2, Vec2};
use std::{
    iter, ptr,
    sync::{Arc, Mutex},
};
use wgpu::{
    Adapter, CommandEncoder, CommandEncoderDescriptor, Device, Queue, Texture, TextureTransition,
    TextureUses, TextureView, hal::api::Vulkan,
};

/// Camera-specific object for using DLSS Ray Reconstruction.
pub struct DlssRayReconstruction {
    upscaled_resolution: UVec2,
    render_resolution: UVec2,
    device: Device,
    sdk: Arc<Mutex<DlssSdk>>,
    feature: *mut NVSDK_NGX_Handle,
}

impl DlssRayReconstruction {
    /// Create a new [`DlssRayReconstruction`] object.
    ///
    /// This is an expensive operation. The resulting object should be cached, and only recreated when settings change.
    ///
    /// This should only be called if [`crate::FeatureSupport::ray_reconstruction_supported`] is true.
    pub fn new(
        upscaled_resolution: UVec2,
        perf_quality_mode: DlssPerfQualityMode,
        feature_flags: DlssFeatureFlags,
        roughness_mode: DlssRayReconstructionRoughnessMode,
        depth_mode: DlssRayReconstructionDepthMode,
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
            check_ngx_result(NGX_DLSSD_GET_OPTIMAL_SETTINGS(
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
        }

        let mut create_params = NVSDK_NGX_DLSSD_Create_Params {
            InDenoiseMode: NVSDK_NGX_DLSS_Denoise_Mode_NVSDK_NGX_DLSS_Denoise_Mode_DLUnified,
            InRoughnessMode: match roughness_mode {
                DlssRayReconstructionRoughnessMode::Unpacked => {
                    NVSDK_NGX_DLSS_Roughness_Mode_NVSDK_NGX_DLSS_Roughness_Mode_Unpacked
                }
                DlssRayReconstructionRoughnessMode::Packed => {
                    NVSDK_NGX_DLSS_Roughness_Mode_NVSDK_NGX_DLSS_Roughness_Mode_Packed
                }
            },
            InUseHWDepth: match depth_mode {
                DlssRayReconstructionDepthMode::Linear => {
                    NVSDK_NGX_DLSS_Depth_Type_NVSDK_NGX_DLSS_Depth_Type_Linear
                }
                DlssRayReconstructionDepthMode::Hardware => {
                    NVSDK_NGX_DLSS_Depth_Type_NVSDK_NGX_DLSS_Depth_Type_HW
                }
            },
            InWidth: optimal_render_resolution.x,
            InHeight: optimal_render_resolution.y,
            InTargetWidth: upscaled_resolution.x,
            InTargetHeight: upscaled_resolution.y,
            InPerfQualityValue: perf_quality_value,
            InFeatureCreateFlags: feature_flags.as_flags(),
            InEnableOutputSubrects: feature_flags.contains(DlssFeatureFlags::OutputSubrect),
        };

        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlss_ray_reconstruction_context_creation"),
        });

        let mut feature = ptr::null_mut();
        unsafe {
            let hal_device = device.as_hal::<Vulkan>().unwrap();
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_CREATE_DLSSD_EXT1(
                    hal_device.raw_device().handle(),
                    command_encoder.unwrap().raw_handle(),
                    1,
                    1,
                    &mut feature,
                    locked_sdk.parameters,
                    &mut create_params,
                ))
            })?
        }

        queue.submit([command_encoder.finish()]);

        Ok(Self {
            upscaled_resolution,
            render_resolution: optimal_render_resolution,
            device: device.clone(),
            sdk: Arc::clone(&sdk),
            feature,
        })
    }

    /// Encode rendering commands for DLSS Ray Reconstruction.
    pub fn render(
        &mut self,
        render_parameters: DlssRayReconstructionRenderParameters,
        command_encoder: &mut CommandEncoder,
        adapter: &Adapter,
    ) -> Result<(), DlssError> {
        render_parameters.validate()?;

        let sdk = self.sdk.lock().unwrap();

        let partial_texture_size = render_parameters
            .partial_texture_size
            .unwrap_or(self.render_resolution);

        // TODO: We may want to expose some more of these
        let mut eval_params = NVSDK_NGX_VK_DLSSD_Eval_Params {
            pInDiffuseAlbedo: &mut texture_to_ngx(render_parameters.diffuse_albedo, adapter)
                as *mut _,
            pInSpecularAlbedo: &mut texture_to_ngx(render_parameters.specular_albedo, adapter)
                as *mut _,
            pInNormals: &mut texture_to_ngx(render_parameters.normals, adapter) as *mut _,
            pInRoughness: match render_parameters.roughness {
                Some(roughness) => &mut texture_to_ngx(roughness, adapter) as *mut _,
                None => ptr::null_mut(),
            },
            pInColor: &mut texture_to_ngx(render_parameters.color, adapter) as *mut _,
            pInAlpha: ptr::null_mut(),
            pInOutput: &mut texture_to_ngx(render_parameters.dlss_output, adapter) as *mut _,
            pInOutputAlpha: ptr::null_mut(),
            pInDepth: &mut texture_to_ngx(render_parameters.depth, adapter) as *mut _,
            pInMotionVectors: &mut texture_to_ngx(render_parameters.motion_vectors, adapter)
                as *mut _,
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
            pInExposureTexture: ptr::null_mut(),
            pInBiasCurrentColorMask: match &render_parameters.bias {
                Some(bias) => &mut texture_to_ngx(bias, adapter) as *mut _,
                None => ptr::null_mut(),
            },
            InAlphaSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InOutputAlphaSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDiffuseAlbedoSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InSpecularAlbedoSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InNormalsSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InRoughnessSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDepthSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InMVSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InTranslucencySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InBiasCurrentColorSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InOutputSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InPreExposure: 0.0,
            InExposureScale: 0.0,
            InIndicatorInvertXAxis: 0,
            InIndicatorInvertYAxis: 0,
            pInReflectedAlbedo: ptr::null_mut(),
            pInColorBeforeParticles: ptr::null_mut(),
            pInColorAfterParticles: ptr::null_mut(),
            pInColorBeforeTransparency: ptr::null_mut(),
            pInColorAfterTransparency: ptr::null_mut(),
            pInColorBeforeFog: ptr::null_mut(),
            pInColorAfterFog: ptr::null_mut(),
            pInScreenSpaceSubsurfaceScatteringGuide: match &render_parameters
                .screen_space_subsurface_scattering_guide
            {
                Some(screen_space_subsurface_scattering_guide) => {
                    &mut texture_to_ngx(screen_space_subsurface_scattering_guide, adapter) as *mut _
                }
                None => ptr::null_mut(),
            },
            pInColorBeforeScreenSpaceSubsurfaceScattering: ptr::null_mut(),
            pInColorAfterScreenSpaceSubsurfaceScattering: ptr::null_mut(),
            pInScreenSpaceRefractionGuide: ptr::null_mut(),
            pInColorBeforeScreenSpaceRefraction: ptr::null_mut(),
            pInColorAfterScreenSpaceRefraction: ptr::null_mut(),
            pInDepthOfFieldGuide: ptr::null_mut(),
            pInColorBeforeDepthOfField: ptr::null_mut(),
            pInColorAfterDepthOfField: ptr::null_mut(),
            pInDiffuseHitDistance: ptr::null_mut(),
            pInSpecularHitDistance: match render_parameters.specular_guide {
                DlssRayReconstructionSpecularGuide::SpecularMotionVectors(_) => ptr::null_mut(),
                DlssRayReconstructionSpecularGuide::SpecularHitDistance {
                    texture_view, ..
                } => &mut texture_to_ngx(texture_view, adapter) as *mut _,
            },
            pInDiffuseRayDirection: ptr::null_mut(),
            pInSpecularRayDirection: ptr::null_mut(),
            pInDiffuseRayDirectionHitDistance: ptr::null_mut(),
            pInSpecularRayDirectionHitDistance: ptr::null_mut(),
            InReflectedAlbedoSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeParticlesSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorAfterParticlesSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeTransparencySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorAfterTransparencySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeFogSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorAfterFogSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InScreenSpaceSubsurfaceScatteringGuideSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeScreenSpaceSubsurfaceScatteringSubrectBase: NVSDK_NGX_Coordinates {
                X: 0,
                Y: 0,
            },
            InColorAfterScreenSpaceSubsurfaceScatteringSubrectBase: NVSDK_NGX_Coordinates {
                X: 0,
                Y: 0,
            },
            InScreenSpaceRefractionGuideSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeScreenSpaceRefractionSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorAfterScreenSpaceRefractionSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDepthOfFieldGuideSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorBeforeDepthOfFieldSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InColorAfterDepthOfFieldSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDiffuseHitDistanceSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InSpecularHitDistanceSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDiffuseRayDirectionSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InSpecularRayDirectionSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InDiffuseRayDirectionHitDistanceSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            InSpecularRayDirectionHitDistanceSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            pInWorldToViewMatrix: match render_parameters.specular_guide {
                DlssRayReconstructionSpecularGuide::SpecularMotionVectors(_) => ptr::null_mut(),
                DlssRayReconstructionSpecularGuide::SpecularHitDistance {
                    world_to_view_matrix,
                    ..
                } => &mut world_to_view_matrix.transpose().to_cols_array() as *mut _,
            },
            pInViewToClipMatrix: match render_parameters.specular_guide {
                DlssRayReconstructionSpecularGuide::SpecularMotionVectors(_) => ptr::null_mut(),
                DlssRayReconstructionSpecularGuide::SpecularHitDistance {
                    view_to_clip_matrix,
                    ..
                } => &mut view_to_clip_matrix.transpose().to_cols_array() as *mut _,
            },
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
            pInTransparencyLayer: ptr::null_mut(),
            InTransparencyLayerSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            pInTransparencyLayerOpacity: ptr::null_mut(),
            InTransparencyLayerOpacitySubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            pInTransparencyLayerMvecs: ptr::null_mut(),
            InTransparencyLayerMvecsSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
            pInDisocclusionMask: ptr::null_mut(),
            InDisocclusionMaskSubrectBase: NVSDK_NGX_Coordinates { X: 0, Y: 0 },
        };

        command_encoder.transition_resources(iter::empty(), render_parameters.barrier_list());
        unsafe {
            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NGX_VULKAN_EVALUATE_DLSSD_EXT(
                    command_encoder.unwrap().raw_handle(),
                    self.feature,
                    sdk.parameters,
                    &mut eval_params,
                ))
            })
        }
    }

    /// Suggested subpixel camera jitter for a given frame.
    pub fn suggested_jitter(&self, frame_number: u32, render_resolution: UVec2) -> Vec2 {
        let ratio = self.upscaled_resolution.x as f32 / render_resolution.x as f32;
        let phase_count = ((8.0 * ratio * ratio) as u32).max(32);
        let i = frame_number % phase_count;

        Vec2 {
            x: halton_sequence(i, 2),
            y: halton_sequence(i, 3),
        } - 0.5
    }

    /// Suggested mip bias to apply when sampling textures.
    pub fn suggested_mip_bias(&self, render_resolution: UVec2) -> f32 {
        (render_resolution.x as f32 / self.upscaled_resolution.x as f32).log2() - 1.0
    }

    /// The upscaled resolution DLSS will output at.
    pub fn upscaled_resolution(&self) -> UVec2 {
        self.upscaled_resolution
    }

    /// The resolution the camera should render at, pre-upscaling.
    pub fn render_resolution(&self) -> UVec2 {
        self.render_resolution
    }
}

impl Drop for DlssRayReconstruction {
    fn drop(&mut self) {
        unsafe {
            let hal_device = self.device.as_hal::<Vulkan>().unwrap();
            hal_device
                .raw_device()
                .device_wait_idle()
                .expect("Failed to wait for idle device when destroying DlssRayReconstruction");

            check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature))
                .expect("Failed to destroy DlssRayReconstruction feature");
        }
    }
}

unsafe impl Send for DlssRayReconstruction {}
unsafe impl Sync for DlssRayReconstruction {}

/// How roughness will be provided to [`DlssRayReconstruction`].
pub enum DlssRayReconstructionRoughnessMode {
    /// Roughness is provided as a standalone texture in [`DlssRayReconstructionRenderParameters::roughness`].
    Unpacked,
    /// Roughness is packed into the alpha channel of the normal texture in [`DlssRayReconstructionRenderParameters::normals`].
    Packed,
}

/// How depth will be provided to [`DlssRayReconstruction`].
pub enum DlssRayReconstructionDepthMode {
    /// Depth will be linear in view-space.
    Linear,
    /// Depth is a hardware depth buffer.
    Hardware,
}

/// Inputs and output resources needed for rendering [`DlssRayReconstruction`].
pub struct DlssRayReconstructionRenderParameters<'a> {
    /// Diffuse albedo.
    pub diffuse_albedo: &'a TextureView,
    /// Specular albedo.
    ///
    /// See section 3.4.2 of `$DLSS_SDK/doc/DLSS-RR Integration Guide.pdf` for how to calculate this texture.
    pub specular_albedo: &'a TextureView,
    /// Normals.
    ///
    /// Can be view-space or world-space.
    ///
    /// Must have linear material roughness in the alpha channel when using [`DlssRayReconstructionRoughnessMode::Packed`].
    pub normals: &'a TextureView,
    /// Linear material roughness.
    ///
    /// Must be provided when using [`DlssRayReconstructionRoughnessMode::Unpacked`].
    pub roughness: Option<&'a TextureView>,
    /// Main color view of your camera.
    pub color: &'a TextureView,
    /// Depth buffer.
    ///
    /// See [`DlssRayReconstructionDepthMode`] for format.
    pub depth: &'a TextureView,
    /// Motion vectors.
    pub motion_vectors: &'a TextureView,
    /// Specular material guide.
    pub specular_guide: DlssRayReconstructionSpecularGuide<'a>,
    /// Screen-space subsurface scattering guide.
    ///
    /// See section 3.4.12 of `$DLSS_SDK/doc/DLSS-RR Integration Guide.pdf` for how to calculate this texture
    pub screen_space_subsurface_scattering_guide: Option<&'a TextureView>,
    /// Optional per-pixel bias to make DLSS more reactive.
    pub bias: Option<&'a TextureView>,
    /// The texture DLSS outputs to.
    pub dlss_output: &'a TextureView,
    /// Whether DLSS should reset temporal history, useful for camera cuts.
    pub reset: bool,
    /// Subpixel jitter that was applied to your camera.
    pub jitter_offset: Vec2,
    /// Optionally use only a specific subrect of the input textures, rather than the whole textures.
    // TODO: Allow configuring partial texture origins
    pub partial_texture_size: Option<UVec2>,
    /// Optional scaling factor to apply to the values contained within [`Self::motion_vectors`].
    pub motion_vector_scale: Option<Vec2>,
}

/// Guide buffer for specular material handling.
pub enum DlssRayReconstructionSpecularGuide<'a> {
    /// Motion vectors for objects reflected in specular material pixels.
    SpecularMotionVectors(&'a TextureView),
    /// World-space distance between primary vertex and hit point from tracing specular material pixels.
    SpecularHitDistance {
        /// Specular hit distance texture.
        texture_view: &'a TextureView,
        /// World-space to view-space camera matrix.
        world_to_view_matrix: Mat4,
        /// View-space to clip-space camera matrix.
        view_to_clip_matrix: Mat4,
    },
}

impl<'a> DlssRayReconstructionRenderParameters<'a> {
    fn validate(&self) -> Result<(), DlssError> {
        // TODO
        Ok(())
    }

    fn barrier_list(&self) -> impl Iterator<Item = TextureTransition<&'a Texture>> {
        fn resource_barrier<'a>(texture_view: &'a TextureView) -> TextureTransition<&'a Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::RESOURCE,
            }
        }

        [
            Some(resource_barrier(&self.diffuse_albedo)),
            Some(resource_barrier(&self.specular_albedo)),
            Some(resource_barrier(&self.normals)),
            self.roughness.map(resource_barrier),
            Some(resource_barrier(&self.color)),
            Some(resource_barrier(&self.depth)),
            Some(resource_barrier(&self.motion_vectors)),
            match &self.specular_guide {
                DlssRayReconstructionSpecularGuide::SpecularMotionVectors(
                    specular_motion_vectors,
                ) => Some(resource_barrier(specular_motion_vectors)),
                DlssRayReconstructionSpecularGuide::SpecularHitDistance {
                    texture_view: specular_hit_distance,
                    ..
                } => Some(resource_barrier(specular_hit_distance)),
            },
            self.screen_space_subsurface_scattering_guide
                .map(resource_barrier),
            self.bias.map(resource_barrier),
            Some(TextureTransition {
                texture: self.dlss_output.texture(),
                selector: None,
                state: TextureUses::STORAGE_READ_WRITE,
            }),
        ]
        .into_iter()
        .flatten()
    }
}
