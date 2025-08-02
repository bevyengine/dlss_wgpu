use crate::{
    feature_info::with_feature_info,
    nvsdk_ngx::{
        DlssError, NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements,
        NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements, check_ngx_result,
    },
};
use ash::{Entry, vk::PhysicalDevice};
use std::{ffi::CStr, ptr, slice};
use uuid::Uuid;
use wgpu::{
    Adapter, Device, DeviceDescriptor, Instance, InstanceDescriptor, Queue, RequestDeviceError,
    hal::{DeviceError, InstanceError, api::Vulkan},
};

/// Creates a wgpu [`Instance`] with the extensions required for DLSS.
///
/// If the system does not support DLSS, it will set `dlss_supported` to false.
pub fn create_instance(
    project_id: Uuid,
    instance_descriptor: &InstanceDescriptor,
    dlss_supported: &mut bool,
) -> Result<Instance, InitializationError> {
    unsafe {
        let mut result = Ok(());
        let raw_instance = wgpu::hal::vulkan::Instance::init_with_callback(
            &wgpu::hal::InstanceDescriptor {
                name: "wgpu",
                flags: instance_descriptor.flags,
                memory_budget_thresholds: instance_descriptor.memory_budget_thresholds,
                backend_options: instance_descriptor.backend_options.clone(),
            },
            Some(Box::new(|args| {
                match required_instance_extensions(project_id, args.entry) {
                    Ok((extensions, true)) => args.extensions.extend(extensions),
                    Ok((_, false)) => *dlss_supported = false,
                    Err(err) => result = Err(err),
                }
            })),
        )?;
        result?;

        Ok(Instance::from_hal::<Vulkan>(raw_instance))
    }
}

/// Creates a wgpu [`Device`] and [`Queue`] with the extensions required for DLSS.
///
/// If the system does not support DLSS, it will set `dlss_supported` to false.
///
/// The provided [`Adapter`] must be using the Vulkan backend.
pub fn request_device(
    project_id: Uuid,
    adapter: &Adapter,
    device_descriptor: &DeviceDescriptor,
    dlss_supported: &mut bool,
) -> Result<(Device, Queue), InitializationError> {
    unsafe {
        let raw_adapter = adapter
            .as_hal::<Vulkan>()
            .ok_or(InitializationError::UnsupportedBackend)?;
        let raw_instance = raw_adapter.shared_instance().raw_instance();
        let raw_physical_device = raw_adapter.raw_physical_device();

        let mut result = Ok(());
        let open_device = raw_adapter.open_with_callback(
            device_descriptor.required_features,
            &device_descriptor.memory_hints,
            Some(Box::new(|args| {
                match required_device_extensions(
                    project_id,
                    &raw_adapter,
                    raw_instance.handle(),
                    raw_physical_device,
                ) {
                    Ok((extensions, true)) => args.extensions.extend(extensions),
                    Ok((_, false)) => *dlss_supported = false,
                    Err(err) => result = Err(err),
                }
            })),
        )?;
        result?;

        Ok(adapter.create_device_from_hal::<Vulkan>(open_device, device_descriptor)?)
    }
}

fn required_instance_extensions(
    project_id: Uuid,
    entry: &Entry,
) -> Result<(impl Iterator<Item = &'static CStr>, bool), InitializationError> {
    with_feature_info(project_id, |feature_info| unsafe {
        // Get required extension names
        let mut required_extensions = ptr::null_mut();
        let mut required_extension_count = 0;
        check_ngx_result(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            feature_info,
            &mut required_extension_count,
            &mut required_extensions,
        ))?;
        let required_extensions =
            slice::from_raw_parts(required_extensions, required_extension_count as usize);
        let required_extensions = required_extensions
            .iter()
            .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()));

        // Check that the required extensions are supported
        let supported_extensions = entry.enumerate_instance_extension_properties(None)?;
        let extensions_supported = required_extensions.clone().all(|required_extension| {
            supported_extensions
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(required_extension))
        });

        Ok((required_extensions, extensions_supported))
    })
}

fn required_device_extensions(
    project_id: Uuid,
    raw_adapter: &wgpu::hal::vulkan::Adapter,
    raw_instance: ash::vk::Instance,
    raw_physical_device: PhysicalDevice,
) -> Result<(impl Iterator<Item = &'static CStr>, bool), InitializationError> {
    with_feature_info(project_id, |feature_info| unsafe {
        // Get required extension names
        let mut required_extensions = ptr::null_mut();
        let mut required_extension_count = 0;
        check_ngx_result(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            raw_instance,
            raw_physical_device,
            feature_info,
            &mut required_extension_count,
            &mut required_extensions,
        ))?;
        let required_extensions =
            slice::from_raw_parts(required_extensions, required_extension_count as usize);
        let required_extensions = required_extensions
            .iter()
            .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()));

        // Check that the required extensions are supported
        let extensions_supported = required_extensions.clone().all(|required_extension| {
            raw_adapter
                .physical_device_capabilities()
                .supports_extension(required_extension)
        });

        Ok((required_extensions, extensions_supported))
    })
}

/// Error returned by [`request_device`].
#[derive(thiserror::Error, Debug)]
pub enum InitializationError {
    #[error(transparent)]
    InstanceError(#[from] InstanceError),
    #[error(transparent)]
    RequestDeviceError(#[from] RequestDeviceError),
    #[error(transparent)]
    DeviceError(#[from] DeviceError),
    #[error(transparent)]
    VulkanError(#[from] ash::vk::Result),
    #[error(transparent)]
    DlssError(#[from] DlssError),
    #[error("Provided adapter is not using the Vulkan backend")]
    UnsupportedBackend,
}
