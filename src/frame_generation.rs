use crate::{DlssSdk, nvsdk_ngx::*};
use std::{
    iter, ptr,
    sync::{Arc, Mutex},
};
use wgpu::{
    Adapter, CommandBuffer, CommandEncoder, CommandEncoderDescriptor, Device, Queue, Texture,
    TextureFormat, TextureTransition, TextureUses, TextureView, hal::api::Vulkan,
};

pub struct DlssFrameGeneration {
    output_resolution: [u32; 2],
    device: Device,
    sdk: Arc<Mutex<DlssSdk>>,
    feature: *mut NVSDK_NGX_Handle,
}

impl DlssFrameGeneration {
    /// Creates a DLSS Frame Generation context.
    ///
    /// All color inputs must be display-ready, with tone mapping and the display transfer
    /// function already applied. Set `hdr` when the backbuffer holds HDR10 values, PQ
    /// encoded in a 10-bit format such as [`TextureFormat::Rgb10a2Unorm`]. NGX does not
    /// accept scRGB or scene-referred linear inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_resolution: [u32; 2],
        render_resolution: [u32; 2],
        output_format: TextureFormat,
        hdr: bool,
        dynamic_resolution_scaling: bool,
        sdk: Arc<Mutex<DlssSdk>>,
        adapter: &Adapter,
        device: &Device,
        queue: &Queue,
    ) -> Result<Self, DlssError> {
        let locked_sdk = sdk.lock().unwrap();
        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlss_frame_generation_context_creation"),
        });
        let mut feature = ptr::null_mut();

        unsafe {
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_Parameter_CreationNodeMask,
                1,
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_Parameter_VisibilityNodeMask,
                1,
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_Parameter_Width,
                output_resolution[0],
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_Parameter_Height,
                output_resolution[1],
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_BackbufferFormat,
                adapter
                    .as_hal::<Vulkan>()
                    .unwrap()
                    .texture_format_as_raw(output_format)
                    .as_raw() as u32,
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                render_resolution[0],
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                render_resolution[1],
            );
            set_u32(
                locked_sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_DynamicResolution,
                dynamic_resolution_scaling as u32,
            );

            command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                check_ngx_result(NVSDK_NGX_VULKAN_CreateFeature(
                    command_encoder.unwrap().raw_handle(),
                    NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration,
                    locked_sdk.parameters,
                    &mut feature,
                ))
            })?;

            set_constant_frame_parameters(locked_sdk.parameters, output_resolution, hdr);
        }

        queue.submit([command_encoder.finish()]);

        drop(locked_sdk);
        Ok(Self {
            output_resolution,
            device: device.clone(),
            sdk,
            feature,
        })
    }

    /// Encode rendering commands for DLSS Frame Generation.
    ///
    /// One frame is generated per entry of [`DlssFrameGenerationRenderParameters::outputs_interpolated`],
    /// evaluated in temporal order. The resulting command buffer should be submitted to a [`Queue`]
    /// in the same submit as the finished `command_encoder`, ordered immediately afterwards.
    pub fn render(
        &mut self,
        render_parameters: DlssFrameGenerationRenderParameters<'_>,
        command_encoder: &mut CommandEncoder,
        adapter: &Adapter,
    ) -> Result<CommandBuffer, DlssError> {
        let sdk = self.sdk.lock().unwrap();
        render_parameters.validate(self.output_resolution, sdk.multi_frame_count_max())?;
        command_encoder.transition_resources(iter::empty(), render_parameters.barrier_list());

        // NGX reads these through raw pointers during EvaluateFeature. The bindings
        // must stay alive until the last evaluate call below.
        let backbuffer = texture_to_ngx(render_parameters.backbuffer, adapter);
        let depth = texture_to_ngx(render_parameters.depth, adapter);
        let motion_vectors = texture_to_ngx(render_parameters.motion_vectors, adapter);
        let hudless = render_parameters
            .hudless
            .map(|view| texture_to_ngx(view, adapter));
        let ui = render_parameters
            .ui
            .map(|view| texture_to_ngx(view, adapter));
        let outputs_interpolated = render_parameters
            .outputs_interpolated
            .iter()
            .map(|view| texture_to_ngx(view, adapter))
            .collect::<Vec<_>>();
        let output_real = render_parameters
            .output_real
            .map(|view| texture_to_ngx(view, adapter));

        let mut dlss_command_encoder =
            self.device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("dlss_frame_generation"),
                });
        unsafe {
            set_ptr(
                sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_Backbuffer,
                &backbuffer,
            );
            set_ptr(sdk.parameters, NVSDK_NGX_DLSSG_Parameter_Depth, &depth);
            set_ptr(
                sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_MVecs,
                &motion_vectors,
            );
            set_optional_ptr(
                sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_HUDLess,
                hudless.as_ref(),
            );
            set_optional_ptr(sdk.parameters, NVSDK_NGX_DLSSG_Parameter_UI, ui.as_ref());
            set_frame_parameters(sdk.parameters, &render_parameters);
            set_u32(
                sdk.parameters,
                NVSDK_NGX_DLSSG_Parameter_MultiFrameCount,
                outputs_interpolated.len() as u32,
            );

            // NGX requires one evaluate call per generated frame, with MultiFrameIndex
            // counting up from 1. Skipped or reordered indices are undefined behavior.
            for (i, output_interpolated) in outputs_interpolated.iter().enumerate() {
                set_u32(
                    sdk.parameters,
                    NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex,
                    i as u32 + 1,
                );
                // NGX only honors reset when MultiFrameIndex is 1
                set_u32(
                    sdk.parameters,
                    NVSDK_NGX_DLSSG_Parameter_Reset,
                    (render_parameters.reset && i == 0) as u32,
                );
                set_ptr(
                    sdk.parameters,
                    NVSDK_NGX_DLSSG_Parameter_OutputInterpolated,
                    output_interpolated,
                );
                // The retained real frame only needs to be written by one evaluation
                set_optional_ptr(
                    sdk.parameters,
                    NVSDK_NGX_DLSSG_Parameter_OutputReal,
                    if i == 0 { output_real.as_ref() } else { None },
                );

                dlss_command_encoder.as_hal_mut::<Vulkan, _, _>(|command_encoder| {
                    check_ngx_result(NVSDK_NGX_VULKAN_EvaluateFeature_C(
                        command_encoder.unwrap().raw_handle(),
                        self.feature,
                        sdk.parameters,
                        None,
                    ))
                })?;
            }
        }

        Ok(dlss_command_encoder.finish())
    }

    pub fn output_resolution(&self) -> [u32; 2] {
        self.output_resolution
    }
}

impl Drop for DlssFrameGeneration {
    fn drop(&mut self) {
        unsafe {
            let hal_device = self.device.as_hal::<Vulkan>().unwrap();
            hal_device
                .raw_device()
                .device_wait_idle()
                .expect("Failed to wait for idle device when destroying DlssFrameGeneration");
            check_ngx_result(NVSDK_NGX_VULKAN_ReleaseFeature(self.feature))
                .expect("Failed to destroy DlssFrameGeneration feature");
        }
    }
}

unsafe impl Send for DlssFrameGeneration {}
unsafe impl Sync for DlssFrameGeneration {}

pub struct DlssFrameGenerationRenderParameters<'a> {
    /// Final display-ready frame, including UI.
    pub backbuffer: &'a TextureView,
    /// Depth buffer.
    pub depth: &'a TextureView,
    /// Motion vectors.
    pub motion_vectors: &'a TextureView,
    /// Optional color buffer without UI drawn on top, for higher quality UI handling.
    pub hudless: Option<&'a TextureView>,
    /// Optional premultiplied UI color and alpha.
    pub ui: Option<&'a TextureView>,
    /// One output texture per generated frame, in temporal order.
    ///
    /// The length sets the generated frame count and must not exceed
    /// [`DlssSdk::multi_frame_count_max`]. One output gives 2x output, two gives 3x, three
    /// gives 4x. Output `i` is the frame interpolated at
    /// `(i + 1) / (len + 1)` between the previous and current rendered frames.
    pub outputs_interpolated: &'a [&'a TextureView],
    /// Optional copy of the real frame, for presenting after the generated frames.
    ///
    /// Alternatively the [`Self::backbuffer`] texture can be retained and presented directly, if
    /// your rendering architecture guarantees it is not overwritten before presentation.
    pub output_real: Option<&'a TextureView>,
    /// Camera data for the rendered frame.
    pub camera: DlssFrameGenerationCamera,
    /// Whether to reset temporal history, e.g. after a camera cut.
    pub reset: bool,
    /// Whether the application is presenting frames outside of gameplay, e.g. a menu or loading screen.
    pub not_rendering_game_frames: bool,
    /// Optionally use only a specific subrect of the depth and motion vector textures, rather
    /// than the whole textures.
    ///
    /// This should match the render resolution the frame generation context was created with,
    /// e.g. when an upscaler renders into a viewport of a larger texture.
    pub partial_texture_size: Option<[u32; 2]>,
    /// A counter that increments by exactly one for each fully rendered frame, including while
    /// frame generation is disabled.
    pub backbuffer_frame_id: u64,
}

impl DlssFrameGenerationRenderParameters<'_> {
    fn validate(
        &self,
        output_resolution: [u32; 2],
        multi_frame_count_max: u32,
    ) -> Result<(), DlssError> {
        let backbuffer = self.backbuffer.texture();
        if [backbuffer.width(), backbuffer.height()] != output_resolution
            || self.outputs_interpolated.is_empty()
            || self.outputs_interpolated.len() as u32 > multi_frame_count_max
        {
            return Err(DlssError::InvalidParameters);
        }
        for output in self
            .outputs_interpolated
            .iter()
            .copied()
            .chain(self.output_real)
        {
            let output = output.texture();
            if [output.width(), output.height()] != output_resolution
                || output.format() != backbuffer.format()
            {
                return Err(DlssError::InvalidParameters);
            }
        }
        Ok(())
    }

    fn barrier_list(&self) -> impl Iterator<Item = TextureTransition<&Texture>> {
        fn input(texture_view: &TextureView) -> TextureTransition<&Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::RESOURCE,
            }
        }
        fn output(texture_view: &TextureView) -> TextureTransition<&Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::STORAGE_READ_WRITE,
            }
        }

        [
            Some(input(self.backbuffer)),
            Some(input(self.depth)),
            Some(input(self.motion_vectors)),
            self.hudless.map(input),
            self.ui.map(input),
            self.output_real.map(output),
        ]
        .into_iter()
        .flatten()
        .chain(self.outputs_interpolated.iter().map(|view| output(view)))
    }
}

/// Camera data for DLSS Frame Generation.
///
/// Matrices are `float[4][4]` in row-major order assuming post-multiplication, `v' = v * M`,
/// and must not contain temporal jitter. For column-vector matrix libraries like glam this is
/// the memory layout of the untransposed matrix, `Mat4::to_cols_array_2d()`.
#[derive(Clone, Copy)]
pub struct DlssFrameGenerationCamera {
    pub camera_view_to_clip: [[f32; 4]; 4],
    pub clip_to_camera_view: [[f32; 4]; 4],
    pub clip_to_previous_clip: [[f32; 4]; 4],
    pub previous_clip_to_clip: [[f32; 4]; 4],
    /// Clip space jitter offset applied to the camera this frame.
    pub jitter_offset: [f32; 2],
    /// Scale to convert motion vector values into screen-space pixel offsets, at the resolution
    /// of the motion vector texture (or its subrect).
    pub motion_vector_scale: [f32; 2],
    pub position: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub near: f32,
    pub far: f32,
    /// Vertical field of view, in radians.
    pub vertical_fov: f32,
    pub aspect_ratio: f32,
    pub depth_inverted: bool,
    pub camera_motion_included: bool,
    pub motion_vectors_dilated: bool,
}

unsafe fn set_frame_parameters(
    parameters: *mut NVSDK_NGX_Parameter,
    render: &DlssFrameGenerationRenderParameters<'_>,
) {
    let camera = &render.camera;
    unsafe {
        set_u64(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID,
            render.backbuffer_frame_id,
        );
        set_ptr(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraViewToClip,
            &camera.camera_view_to_clip,
        );
        set_ptr(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_ClipToCameraView,
            &camera.clip_to_camera_view,
        );
        set_ptr(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip,
            &camera.clip_to_previous_clip,
        );
        set_ptr(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_PrevClipToClip,
            &camera.previous_clip_to_clip,
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_JitterOffsetX,
            camera.jitter_offset[0],
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_JitterOffsetY,
            camera.jitter_offset[1],
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MvecScaleX,
            camera.motion_vector_scale[0],
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MvecScaleY,
            camera.motion_vector_scale[1],
        );
        let camera_vectors: [([&[u8]; 3], [f32; 3]); 4] = [
            (
                [
                    NVSDK_NGX_DLSSG_Parameter_CameraPosX,
                    NVSDK_NGX_DLSSG_Parameter_CameraPosY,
                    NVSDK_NGX_DLSSG_Parameter_CameraPosZ,
                ],
                camera.position,
            ),
            (
                [
                    NVSDK_NGX_DLSSG_Parameter_CameraUpX,
                    NVSDK_NGX_DLSSG_Parameter_CameraUpY,
                    NVSDK_NGX_DLSSG_Parameter_CameraUpZ,
                ],
                camera.up,
            ),
            (
                [
                    NVSDK_NGX_DLSSG_Parameter_CameraRightX,
                    NVSDK_NGX_DLSSG_Parameter_CameraRightY,
                    NVSDK_NGX_DLSSG_Parameter_CameraRightZ,
                ],
                camera.right,
            ),
            (
                [
                    NVSDK_NGX_DLSSG_Parameter_CameraFwdX,
                    NVSDK_NGX_DLSSG_Parameter_CameraFwdY,
                    NVSDK_NGX_DLSSG_Parameter_CameraFwdZ,
                ],
                camera.forward,
            ),
        ];
        for (names, vector) in camera_vectors {
            for (name, value) in names.into_iter().zip(vector) {
                set_f32(parameters, name, value);
            }
        }
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraNear,
            camera.near,
        );
        set_f32(parameters, NVSDK_NGX_DLSSG_Parameter_CameraFar, camera.far);
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraFOV,
            camera.vertical_fov,
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio,
            camera.aspect_ratio,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_DepthInverted,
            camera.depth_inverted as u32,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded,
            camera.camera_motion_included as u32,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames,
            render.not_rendering_game_frames as u32,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MvecDilated,
            camera.motion_vectors_dilated as u32,
        );
        // Depth and motion vectors may only be valid in a subrect matching the render resolution
        let [partial_width, partial_height] = render.partial_texture_size.unwrap_or([
            render.motion_vectors.texture().width(),
            render.motion_vectors.texture().height(),
        ]);
        let partial_size_textures: [(&[u8], &[u8]); 2] = [
            (
                NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight,
            ),
        ];
        for (width, height) in partial_size_textures {
            set_u32(parameters, width, partial_width);
            set_u32(parameters, height, partial_height);
        }
        if let Some(hudless) = render.hudless {
            set_texture_size(
                parameters,
                NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectHeight,
                hudless,
            );
        }
        if let Some(ui) = render.ui {
            set_texture_size(
                parameters,
                NVSDK_NGX_DLSSG_Parameter_UISubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_UISubrectHeight,
                ui,
            );
        }
    }
}

/// Sets the parameters that never change for the lifetime of the feature. The NGX
/// parameter map persists across frames, so setting them once at creation is enough.
unsafe fn set_constant_frame_parameters(
    parameters: *mut NVSDK_NGX_Parameter,
    output_resolution: [u32; 2],
    hdr: bool,
) {
    unsafe {
        // NGX reads matrix pointers during EvaluateFeature, so the identity must stay
        // alive past this function. A static lives long enough.
        static IDENTITY: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        set_ptr(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_ClipToLensClip,
            &IDENTITY,
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetX,
            0.0,
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetY,
            0.0,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR,
            hdr as u32,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_AutomodeOverrideReset,
            0,
        );
        set_u32(parameters, NVSDK_NGX_DLSSG_Parameter_OrthoProjection, 0);
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MvecInvalidValue,
            f32::NAN,
        );
        set_u32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MenuDetectionEnabled,
            0,
        );
        set_f32(
            parameters,
            NVSDK_NGX_DLSSG_Parameter_MinRelativeLinearDepthObjectSeparation,
            40.0,
        );
        // This crate never provides these optional resources, so they stay null
        let unused_resources: [&[u8]; 3] = [
            NVSDK_NGX_DLSSG_Parameter_UIAlpha,
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField,
            NVSDK_NGX_DLSSG_Parameter_OutputDisableInterpolation,
        ];
        for name in unused_resources {
            set_ptr(parameters, name, ptr::null::<NVSDK_NGX_Resource_VK>());
        }
        // Subrects are unused, resources are always read from the origin
        let subrect_bases: [(&[u8], &[u8]); 7] = [
            (
                NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_UISubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_UISubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseY,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseX,
                NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseY,
            ),
        ];
        for (base_x, base_y) in subrect_bases {
            set_u32(parameters, base_x, 0);
            set_u32(parameters, base_y, 0);
        }
        // validate() guarantees the backbuffer and every output match the output resolution
        let full_size_textures: [(&[u8], &[u8]); 3] = [
            (
                NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectHeight,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectHeight,
            ),
            (
                NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectWidth,
                NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectHeight,
            ),
        ];
        for (width, height) in full_size_textures {
            set_u32(parameters, width, output_resolution[0]);
            set_u32(parameters, height, output_resolution[1]);
        }
    }
}
