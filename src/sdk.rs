use crate::dlss::*;
use std::env;
use std::ffi::{CString, OsString};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;
use uuid::Uuid;
use wgpu::Device;
use wgpu_core::api::Vulkan;

pub struct DlssSdk<D: Deref<Target = Device> + Clone> {
    pub(crate) parameters: *mut NVSDK_NGX_Parameter,
    pub(crate) device: D,
}

impl<D: Deref<Target = Device> + Clone> DlssSdk<D> {
    pub fn dlss_available(project_id: Uuid, device: D) -> Result<bool, DlssError> {
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
                PathListInfo: dlss_shared_libary_paths(),
                InternalData: ptr::null_mut(),
                // TODO: Allow configuring logging
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

    pub fn new(project_id: Uuid, device: D) -> Result<Self, DlssError> {
        let project_id = CString::new(project_id.to_string()).unwrap();
        let engine_version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
        let sdk_info = NVSDK_NGX_FeatureCommonInfo {
            PathListInfo: dlss_shared_libary_paths(),
            InternalData: ptr::null_mut(),
            // TODO: Allow configuring logging
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
            check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters))?;

            let mut dlss_supported = 0;
            check_ngx_result(NVSDK_NGX_Parameter_GetI(
                parameters,
                &NVSDK_NGX_Parameter_SuperSampling_Available[0] as *const u8 as *const i8,
                &mut dlss_supported,
            ))?;
            if dlss_supported == 0 {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                return Err(DlssError::FeatureNotSupported);
            }

            Ok(Self { parameters, device })
        }
    }
}

impl<D: Deref<Target = Device> + Clone> Drop for DlssSdk<D> {
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

pub fn dlss_shared_libary_paths() -> NVSDK_NGX_PathListInfo {
    #[cfg(not(target_os = "windows"))]
    let platform = "Linux_x86_64";
    #[cfg(target_os = "windows")]
    let platform = "Windowsx86_64";
    #[cfg(debug_assertions)]
    let profile = "dev";
    #[cfg(not(debug_assertions))]
    let profile = "rel";
    let sdk_path = format!("{}/lib/{platform}/{profile}", env!("DLSS_SDK"));

    NVSDK_NGX_PathListInfo {
        Path: [
            os_str_to_wchar(&OsString::from(".")).as_ptr(),
            os_str_to_wchar(&OsString::from(sdk_path)).as_ptr(),
        ]
        .as_ptr(),
        Length: 2,
    }
}
