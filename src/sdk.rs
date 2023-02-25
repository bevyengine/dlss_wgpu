use crate::dlss::*;
use std::env;
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;
use uuid::Uuid;
use wgpu::Device;
use wgpu_core::api::Vulkan;

pub struct DLSSSDK<D: Deref<Target = Device> + Clone> {
    pub(crate) parameters: *mut NVSDK_NGX_Parameter,
    pub(crate) device: D,
}

impl<D: Deref<Target = Device> + Clone> DLSSSDK<D> {
    pub fn dlss_supported(project_id: Uuid, device: D) -> Result<bool, DLSSError> {
        let project_id = CString::new(project_id.to_string()).unwrap();
        let engine_version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
        let sdk_info = NVSDK_NGX_FeatureDiscoveryInfo {
            SDKVersion: NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            FeatureID: NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
            Identifier: NVSDK_NGX_Application_Identifier {
                IdentifierType: NVSDK_NGX_Application_Identifier_Type_NVSDK_NGX_Application_Identifier_Type_Project_Id,
                v: NVSDK_NGX_Application_Identifier_v {
                    ProjectDesc: NVSDK_NGX_ProjectIdDescription {
                        ProjectId: project_id.as_ptr(),
                        EngineType: NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                        EngineVersion: engine_version.as_ptr(),
                    },
                },
            },
            ApplicationDataPath: os_str_to_wchar(env::temp_dir().as_os_str()).as_ptr(),
            FeatureInfo: &NVSDK_NGX_FeatureCommonInfo {
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
            },
        };

        unsafe {
            let (vk_instance, vk_physical_device) = device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap();
                (
                    device.shared_instance().raw_instance().handle(),
                    device.raw_physical_device(),
                )
            });

            let mut supported_features = MaybeUninit::uninit();
            check_ngx_result(NVSDK_NGX_VULKAN_GetFeatureRequirements(
                vk_instance,
                vk_physical_device,
                &sdk_info,
                supported_features.as_mut_ptr(),
            ))?;
            let supported_features = supported_features.assume_init();

            Ok(supported_features.FeatureSupported
                == NVSDK_NGX_Feature_Support_Result_NVSDK_NGX_FeatureSupportResult_Supported)
        }
    }

    pub fn new(project_id: Uuid, device: D) -> Result<Self, DLSSError> {
        let project_id = CString::new(project_id.to_string()).unwrap();
        let engine_version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
        let sdk_info = NVSDK_NGX_FeatureCommonInfo {
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

            check_ngx_result(NVSDK_NGX_VULKAN_Init_with_ProjectID(
                project_id.as_ptr(),
                NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                engine_version.as_ptr(),
                os_str_to_wchar(env::temp_dir().as_os_str()).as_ptr(),
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_gipa,
                vk_gdpa,
                &sdk_info,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            ))?;

            let mut parameters = ptr::null_mut();
            check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(
                &mut parameters as *mut _,
            ))?;

            let mut dlss_supported = MaybeUninit::uninit();
            NVSDK_NGX_Parameter_GetI(
                parameters,
                &NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult[0] as *const u8 as *const i8,
                dlss_supported.as_mut_ptr(),
            );
            let dlss_supported = dlss_supported.assume_init();
            if dlss_supported == 0 {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                return Err(DLSSError::FeatureNotSupported);
            }

            Ok(Self { parameters, device })
        }
    }
}

impl<D: Deref<Target = Device> + Clone> Drop for DLSSSDK<D> {
    fn drop(&mut self) {
        unsafe {
            self.device.as_hal::<Vulkan, _, _>(|device| {
                let device = device.unwrap().raw_device();

                device
                    .device_wait_idle()
                    .expect("Failed to wait for idle device when destroying DLSSSDK");

                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(self.parameters))
                    .expect("Failed to destroy DLSSSDK parameters");
                check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(device.handle()))
                    .expect("Failed to destroy DLSSSDK");
            });
        }
    }
}
