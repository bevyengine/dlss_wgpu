use crate::{
    feature_info::with_feature_info,
    nvsdk_ngx::{
        check_ngx_result, DlssError, NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements,
    },
};
use ash::vk::{DeviceCreateInfo, DeviceQueueCreateInfo, Instance, PhysicalDevice};
use std::{ffi::CStr, path::Path, ptr, slice};
use uuid::Uuid;
use wgpu::{hal::api::Vulkan, Adapter, Device, DeviceDescriptor, Queue};

/// TODO: Docs
pub fn request_device(
    project_id: Uuid,
    adapter: &Adapter,
    device_descriptor: &DeviceDescriptor,
    trace_path: Option<&Path>,
) -> Result<(Device, Queue), RequestDeviceError> {
    unsafe {
        let open_device: Result<_, RequestDeviceError> =
            adapter.as_hal::<Vulkan, _, _>(|raw_adapter| {
                let raw_adapter = raw_adapter.unwrap();
                let raw_instance = raw_adapter.shared_instance().raw_instance();
                let raw_physical_device = raw_adapter.raw_physical_device();

                let mut enabled_extensions =
                    raw_adapter.required_device_extensions(device_descriptor.required_features);
                enabled_extensions.extend(dlss_device_extensions(
                    project_id,
                    raw_adapter,
                    raw_instance.handle(),
                    raw_physical_device,
                )?);
                let mut enabled_phd_features = raw_adapter.physical_device_features(
                    &enabled_extensions,
                    device_descriptor.required_features,
                );

                let family_index = 0;
                let family_info = DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(&[1.0]);
                let family_infos = [family_info];

                let str_pointers = enabled_extensions
                    .iter()
                    .map(|&s| s.as_ptr())
                    .collect::<Vec<_>>();

                let pre_info = DeviceCreateInfo::default()
                    .queue_create_infos(&family_infos)
                    .enabled_extension_names(&str_pointers);
                let info = enabled_phd_features.add_to_device_create(pre_info);

                let raw_device = raw_instance.create_device(raw_physical_device, &info, None)?;

                Ok(raw_adapter.device_from_raw(
                    raw_device,
                    None,
                    &enabled_extensions,
                    device_descriptor.required_features,
                    &device_descriptor.memory_hints,
                    family_info.queue_family_index,
                    0,
                )?)
            });

        Ok(
            adapter.create_device_from_hal::<Vulkan>(
                open_device?,
                device_descriptor,
                trace_path,
            )?,
        )
    }
}

fn dlss_device_extensions(
    project_id: Uuid,
    raw_adapter: &wgpu::hal::vulkan::Adapter,
    raw_instance: Instance,
    raw_physical_device: PhysicalDevice,
) -> Result<impl Iterator<Item = &'static CStr>, DlssError> {
    with_feature_info(project_id, |feature_info| unsafe {
        let mut dlss_device_extensions = ptr::null_mut();
        let mut dlss_device_extension_count = 0;

        check_ngx_result(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            raw_instance,
            raw_physical_device,
            feature_info,
            &mut dlss_device_extension_count,
            &mut dlss_device_extensions,
        ))?;

        let dlss_device_extensions =
            slice::from_raw_parts(dlss_device_extensions, dlss_device_extension_count as usize);

        let dlss_device_extensions = dlss_device_extensions
            .iter()
            .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()));

        if !dlss_device_extensions.clone().all(|extension| {
            raw_adapter
                .physical_device_capabilities()
                .supports_extension(extension)
        }) {
            return Err(DlssError::FeatureNotSupported);
        }

        Ok(dlss_device_extensions)
    })
}

/// TODO: Docs
#[derive(thiserror::Error, Debug)]
pub enum RequestDeviceError {
    #[error(transparent)]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error(transparent)]
    DeviceError(#[from] wgpu::hal::DeviceError),
    #[error(transparent)]
    VulkanError(#[from] ash::vk::Result),
    #[error(transparent)]
    DlssError(#[from] DlssError),
}
