use crate::feature_info::with_feature_info;
use crate::nvsdk_ngx::{
    check_ngx_result, DlssError, DlssRequestDeviceError,
    NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements,
};
use ash::vk::{
    DeviceCreateInfo, DeviceQueueCreateInfo, Instance, PhysicalDevice,
    PhysicalDeviceBufferDeviceAddressFeaturesEXT, PhysicalDeviceHostQueryResetFeaturesEXT,
};
use std::ffi::CStr;
use std::path::Path;
use std::ptr;
use std::slice;
use uuid::Uuid;
use wgpu::hal::api::Vulkan;
use wgpu::{Adapter, Device, DeviceDescriptor, Queue};

// TODO: Combine request_device() and DlssSdk::new()

pub fn request_device(
    project_id: Uuid,
    adapter: &Adapter,
    device_descriptor: &DeviceDescriptor,
    trace_path: Option<&Path>,
) -> Result<(Device, Queue), DlssRequestDeviceError> {
    unsafe {
        let open_device: Result<_, DlssRequestDeviceError> =
            adapter.as_hal::<Vulkan, _, _>(|adapter| {
                let adapter = adapter.unwrap();
                let vk_instance = adapter.shared_instance().raw_instance();
                let vk_physical_device = adapter.raw_physical_device();

                let mut extensions =
                    adapter.required_device_extensions(device_descriptor.required_features);
                extensions.extend(dlss_device_extensions(
                    project_id,
                    vk_instance.handle(),
                    vk_physical_device,
                )?);
                let extension_pointers = extensions.iter().map(|&s| s.as_ptr()).collect::<Vec<_>>();

                let queue_family_index = 0;
                let queue_family_info = DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&[1.0]);

                let device_info = adapter
                    .physical_device_features(&extensions, device_descriptor.required_features)
                    .add_to_device_create(
                        DeviceCreateInfo::default()
                            .queue_create_infos(&[queue_family_info])
                            .enabled_extension_names(&extension_pointers)
                            // TODO: Varies per gpu/driver?
                            .push_next(
                                &mut PhysicalDeviceBufferDeviceAddressFeaturesEXT::default()
                                    .buffer_device_address(true),
                            )
                            .push_next(
                                &mut PhysicalDeviceHostQueryResetFeaturesEXT::default()
                                    .host_query_reset(true),
                            ),
                    );

                let vk_device =
                    vk_instance.create_device(vk_physical_device, &device_info, None)?;

                Ok(adapter.device_from_raw(
                    vk_device,
                    None,
                    &extensions,
                    device_descriptor.required_features,
                    &device_descriptor.memory_hints,
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
