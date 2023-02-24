mod dlss;

pub use dlss::DLSSError;

use dlss::*;
use std::env;
use std::ops::Deref;
use std::ptr;
use wgpu::{CommandEncoder, Device};
use wgpu_core::api::Vulkan;

pub struct DLSSSDK<D: Deref<Target = Device>> {
    device: D,
}

impl<D: Deref<Target = Device>> DLSSSDK<D> {
    pub fn new(application_id: Option<u64>, device: D) -> Result<Self, DLSSError> {
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
                os_str_to_wchar(env::temp_dir().as_os_str()).as_ptr() as *const _,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_gipa,
                vk_gdpa,
                &feature_info as *const _,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            ))?;

            // TODO: Check if DLSS is available on the system

            // TODO: Obtain optimal settings for each display resolution and DLSS Execution Mode
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

impl DLSSContext {
    pub fn new<D: Deref<Target = Device>>(
        dlss_sdk: &DLSSSDK<D>,
        command_encoder: &mut CommandEncoder,
    ) -> Result<Self, DLSSError> {
        unsafe {
            check_ngx_result(NGX_VULKAN_CREATE_DLSS_EXT(
                todo!(),
                1,
                1,
                todo!(),
                todo!(),
                todo!(),
            ))?;
        }

        Ok(Self {})
    }
}

impl Drop for DLSSContext {
    fn drop(&mut self) {
        todo!()
    }
}
