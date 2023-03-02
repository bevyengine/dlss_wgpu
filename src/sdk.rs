use crate::feature_info::with_feature_info;
use crate::nvsdk_ngx::*;
use std::ops::Deref;
use std::ptr;
use std::rc::Rc;
use uuid::Uuid;
use wgpu::Device;
use wgpu_core::api::Vulkan;

pub struct DlssSdk<D: Deref<Target = Device>> {
    pub(crate) parameters: *mut NVSDK_NGX_Parameter,
    pub(crate) device: D,
}

impl<D: Deref<Target = Device>> DlssSdk<D> {
    pub fn new(project_id: Uuid, device: D) -> Result<Rc<Self>, DlssError> {
        let mut parameters = ptr::null_mut();
        unsafe {
            device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap();
                let shared_instance = device.shared_instance();
                let raw_instance = shared_instance.raw_instance();

                with_feature_info(project_id, |feature_info| {
                    check_ngx_result(NVSDK_NGX_VULKAN_Init_with_ProjectID(
                        feature_info.Identifier.v.ProjectDesc.ProjectId,
                        NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                        feature_info.Identifier.v.ProjectDesc.EngineVersion,
                        feature_info.ApplicationDataPath,
                        raw_instance.handle(),
                        device.raw_physical_device(),
                        device.raw_device().handle(),
                        shared_instance.entry().static_fn().get_instance_proc_addr,
                        raw_instance.fp_v1_0().get_device_proc_addr,
                        feature_info.FeatureInfo,
                        NVSDK_NGX_Version_NVSDK_NGX_Version_API,
                    ))
                })?;

                check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters))
            })?;

            let mut dlss_supported = 0;
            let result = check_ngx_result(NVSDK_NGX_Parameter_GetI(
                parameters,
                NVSDK_NGX_Parameter_SuperSampling_Available.as_ptr().cast(),
                &mut dlss_supported,
            ));
            if result.is_err() {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                result?;
            }
            if dlss_supported == 0 {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                return Err(DlssError::FeatureNotSupported);
            }

            Ok(Rc::new(Self { parameters, device }))
        }
    }
}

impl<D: Deref<Target = Device>> Drop for DlssSdk<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap().raw_device();

                device
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DlssSdk");

                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(self.parameters))
                    .expect("Failed to destroy DlssSdk parameters");
                check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(device.handle()))
                    .expect("Failed to destroy DlssSdk");
            });
        }
    }
}
