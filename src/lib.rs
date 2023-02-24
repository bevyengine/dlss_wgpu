mod dlss;

pub use dlss::DLSSError;

use directories::BaseDirs;
use dlss::*;
use std::ffi::OsString;
use std::ops::Deref;
use std::ptr;
use wgpu::Device;
use wgpu_core::api::Vulkan;

pub struct DLSSSDK<D: Deref<Target = Device>> {
    device: D,
}

impl<D: Deref<Target = Device>> DLSSSDK<D> {
    pub fn new(application_id: Option<u64>, device: D) -> Result<Self, DLSSError> {
        let cache_dir = BaseDirs::new()
            .map(|bd| bd.cache_dir().as_os_str().to_os_string())
            .unwrap_or(OsString::from("."));

        let feature_info = NVSDK_NGX_FeatureCommonInfo {
            // TODO: Allow passing list of extra DLSS shared library paths
            PathListInfo: NVSDK_NGX_PathListInfo {
                Path: ptr::null(),
                Length: 0,
            },
            InternalData: ptr::null_mut(),
            LoggingInfo: NVSDK_NGX_LoggingInfo {
                LoggingCallback: None,
                MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_OFF,
                DisableOtherLoggingSinks: false,
            },
        };

        unsafe {
            let (vk_instance, vk_physical_device, vk_device, vk_gipa, vk_gdpa) = device
                .as_hal::<Vulkan, _, _>(|device| {
                    let device = device.unwrap();
                    let shared_instance = device.shared_instance();
                    let raw_instance = shared_instance.raw_instance();
                    (
                        raw_instance.handle(),
                        device.raw_physical_device(),
                        device.raw_device().handle(),
                        shared_instance.entry().static_fn().get_instance_proc_addr,
                        raw_instance.fp_v1_0().get_device_proc_addr,
                    )
                });

            check_ngx_result(NVSDK_NGX_VULKAN_Init(
                application_id.unwrap_or(0),
                os_str_to_wchar(&cache_dir).as_ptr() as *const _,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_gipa,
                vk_gdpa,
                &feature_info as *const _,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            ))?;
        }

        Ok(Self { device })
    }
}

impl<D: Deref<Target = Device>> Drop for DLSSSDK<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap().raw_device();

                device
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DLSSSDK");

                check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(device.handle()))
                    .expect("Failed to destroy DLSSSDK");
            });
        }
    }
}

pub struct DLSSContext {}

impl DLSSContext {}

impl Drop for DLSSContext {
    fn drop(&mut self) {
        todo!()
    }
}
