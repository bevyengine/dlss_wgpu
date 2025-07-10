use crate::{
    DlssError,
    nvsdk_ngx::{NVSDK_NGX_Create_ImageView_Resource_VK, NVSDK_NGX_Resource_VK},
};
use ash::vk::{
    ImageAspectFlags, ImageSubresourceRange, REMAINING_ARRAY_LAYERS, REMAINING_MIP_LEVELS,
};
use glam::{UVec2, Vec2};
use wgpu::{
    Adapter, Texture, TextureTransition, TextureUsages, TextureUses, TextureView, hal::api::Vulkan,
};

/// Inputs and output resources needed for rendering DLSS.
pub struct DlssRenderParameters<'a> {
    /// Main color view of your camera.
    pub color: &'a TextureView,
    /// Depth buffer.
    pub depth: &'a TextureView,
    // Motion vectors.
    pub motion_vectors: &'a TextureView,
    /// Camera exposure settings.
    pub exposure: DlssExposure<'a>,
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

impl<'a> DlssRenderParameters<'a> {
    pub(crate) fn validate(&self) -> Result<(), DlssError> {
        // TODO
        Ok(())
    }

    pub(crate) fn barrier_list(&self) -> impl Iterator<Item = TextureTransition<&'a Texture>> {
        fn resource_barrier<'a>(texture_view: &'a TextureView) -> TextureTransition<&'a Texture> {
            TextureTransition {
                texture: texture_view.texture(),
                selector: None,
                state: TextureUses::RESOURCE,
            }
        }

        [
            Some(resource_barrier(&self.color)),
            Some(resource_barrier(&self.depth)),
            Some(resource_barrier(&self.motion_vectors)),
            match &self.exposure {
                DlssExposure::Manual { exposure, .. } => Some(resource_barrier(exposure)),
                DlssExposure::Automatic => None,
            },
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

/// Camera exposure used by DLSS.
pub enum DlssExposure<'a> {
    /// Exposure controlled by the application.
    Manual {
        exposure: &'a TextureView,
        exposure_scale: Option<f32>,
        pre_exposure: Option<f32>,
    },
    /// Auto-exposure handled by DLSS.
    Automatic,
}

pub(crate) fn texture_to_ngx_resource(
    texture_view: &TextureView,
    adapter: &Adapter,
) -> NVSDK_NGX_Resource_VK {
    let texture = texture_view.texture();
    unsafe {
        NVSDK_NGX_Create_ImageView_Resource_VK(
            texture_view.as_hal::<Vulkan>().unwrap().raw_handle(),
            texture.as_hal::<Vulkan>().unwrap().raw_handle(),
            ImageSubresourceRange {
                aspect_mask: if texture.format().has_color_aspect() {
                    ImageAspectFlags::COLOR
                } else {
                    ImageAspectFlags::DEPTH
                },
                base_mip_level: 0,
                level_count: REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: REMAINING_ARRAY_LAYERS,
            },
            adapter
                .as_hal::<Vulkan>()
                .unwrap()
                .texture_format_as_raw(texture.format()),
            texture.width(),
            texture.height(),
            texture.usage().contains(TextureUsages::STORAGE_BINDING),
        )
    }
}
