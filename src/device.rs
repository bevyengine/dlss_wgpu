use crate::feature_info::with_feature_info;
use crate::nvsdk_ngx::{
    check_ngx_result, DlssError, NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements,
    RequestDeviceError,
};
use ash::vk::{DeviceCreateInfo, DeviceQueueCreateInfo, Instance, PhysicalDevice};
use std::ffi::CStr;
use std::path::Path;
use std::ptr;
use std::slice;
use uuid::Uuid;
use wgpu::{Adapter, Device, DeviceDescriptor, Queue};
use wgpu_core::api::Vulkan;

pub fn request_device(
    project_id: Uuid,
    adapter: &Adapter,
    device_descriptor: &DeviceDescriptor,
    trace_path: Option<&Path>,
) -> Result<(Device, Queue), RequestDeviceError> {
    unsafe {
        let open_device: Result<_, RequestDeviceError> =
            adapter.as_hal::<Vulkan, _, _>(|adapter| {
                let adapter = adapter.unwrap();
                let vk_instance = adapter.shared_instance().raw_instance();
                let vk_physical_device = adapter.raw_physical_device();

                let mut extensions = adapter.required_device_extensions(device_descriptor.features);
                extensions.extend(dlss_device_extensions(
                    project_id,
                    vk_instance.handle(),
                    vk_physical_device,
                )?);
                let extension_pointers = extensions.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

                let queue_family_index = 0;
                let queue_family_info = DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&[1.0])
                    .build();

                let device_info = adapter
                    .physical_device_features(&extensions, device_descriptor.features)
                    .add_to_device_create_builder(
                        DeviceCreateInfo::builder()
                            .queue_create_infos(&[queue_family_info])
                            .enabled_extension_names(&extension_pointers),
                    )
                    .build();

                let vk_device =
                    vk_instance.create_device(vk_physical_device, &device_info, None)?;

                Ok(adapter.device_from_raw(
                    vk_device,
                    true,
                    &extensions,
                    device_descriptor.features,
                    queue_family_index,
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
    vk_instance: Instance,
    vk_physical_device: PhysicalDevice,
) -> Result<impl Iterator<Item = &'static CStr>, DlssError> {
    with_feature_info(project_id, |feature_info| unsafe {
        let mut dlss_device_extensions = ptr::null_mut();
        let mut dlss_device_extension_count = 0;

        check_ngx_result(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            vk_instance,
            vk_physical_device,
            feature_info,
            &mut dlss_device_extension_count,
            &mut dlss_device_extensions,
        ))?;

        let dlss_device_extensions =
            slice::from_raw_parts(dlss_device_extensions, dlss_device_extension_count as usize);

        let dlss_device_extensions = dlss_device_extensions
            .iter()
            .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()));

        Ok(dlss_device_extensions)
    })
}
