//! Presentation pacing via the [VK_NV_present_metering] Vulkan extension, which lets the
//! driver evenly meter the display timing of a batch of presented frames, e.g. the
//! generated and real frames produced by frame generation.
//!
//! The extension's structs are not yet available in ash, so they are defined manually here.
//!
//! Usage:
//! 1. Enable the extension at device creation. Both `crate::request_device` and
//!    `crate::register_device_extensions` enable it automatically when supported. A fully
//!    manual [`open_with_callback`](wgpu::hal::vulkan::Adapter::open_with_callback) callback
//!    can call `register_present_metering` instead.
//! 2. Before presenting the first frame of each batch, chain a `SetPresentConfigNV`
//!    onto the surface's next present with
//!    `wgpu::hal::vulkan::Surface::set_next_present_chain`, which requires wgpu 31.
//!    Keep the struct alive until that present completes.
//!
//! [VK_NV_present_metering]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_NV_present_metering.html

use ash::vk;
use std::ffi::{CStr, c_void};
use wgpu::hal::vulkan::CreateDeviceCallbackArgs;

pub const NAME: &CStr = c"VK_NV_present_metering";

pub const STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV: vk::StructureType =
    vk::StructureType::from_raw(1000613000);
pub const STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_METERING_FEATURES_NV: vk::StructureType =
    vk::StructureType::from_raw(1000613001);

/// `VkSetPresentConfigNV`. Chained onto `VkPresentInfoKHR` for the first present of a
/// batch of `num_frames_per_batch` frames to have the driver meter their display timing.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SetPresentConfigNV {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub num_frames_per_batch: u32,
    pub present_config_feedback: u32,
}

impl Default for SetPresentConfigNV {
    fn default() -> Self {
        Self {
            s_type: STRUCTURE_TYPE_SET_PRESENT_CONFIG_NV,
            p_next: std::ptr::null(),
            num_frames_per_batch: 0,
            present_config_feedback: 0,
        }
    }
}

unsafe impl vk::ExtendsPresentInfoKHR for SetPresentConfigNV {}

// SAFETY: Plain data. `p_next` is caller-managed, and anything chained through it is
// covered by the safety contract of whatever consumes the struct.
unsafe impl Send for SetPresentConfigNV {}
unsafe impl Sync for SetPresentConfigNV {}

/// `VkPhysicalDevicePresentMeteringFeaturesNV`
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PhysicalDevicePresentMeteringFeaturesNV {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub present_metering: vk::Bool32,
}

impl Default for PhysicalDevicePresentMeteringFeaturesNV {
    fn default() -> Self {
        Self {
            s_type: STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_METERING_FEATURES_NV,
            p_next: std::ptr::null_mut(),
            present_metering: vk::FALSE,
        }
    }
}

unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDevicePresentMeteringFeaturesNV {}
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhysicalDevicePresentMeteringFeaturesNV {}

// SAFETY: The `p_next` pointer is only ever null or pointing at another feature struct
// with the same constraints while chained during device creation on a single thread.
unsafe impl Send for PhysicalDevicePresentMeteringFeaturesNV {}
unsafe impl Sync for PhysicalDevicePresentMeteringFeaturesNV {}

/// Call this inside of [`wgpu::hal::vulkan::Adapter::open_with_callback`] to enable
/// VK_NV_present_metering, if the adapter supports it. Returns whether it is supported.
///
/// The feature struct chained onto the create info must outlive device creation, which
/// stored callbacks such as Bevy's `RawVulkanInitSettings` can't guarantee from the stack,
/// so it is leaked. This costs 16 bytes per device creation when supported.
pub fn register_present_metering(
    args: &mut CreateDeviceCallbackArgs,
    raw_adapter: &wgpu::hal::vulkan::Adapter,
) -> bool {
    if !raw_adapter
        .physical_device_capabilities()
        .supports_extension(NAME)
    {
        return false;
    }
    let features = Box::leak(Box::new(PhysicalDevicePresentMeteringFeaturesNV {
        present_metering: vk::TRUE,
        ..Default::default()
    }));
    args.extensions.push(NAME);
    *args.create_info = args.create_info.push_next(features);
    true
}
