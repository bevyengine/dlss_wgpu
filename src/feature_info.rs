use crate::nvsdk_ngx::*;
use std::env;
use std::ffi::{CString, OsStr, OsString};
use std::ptr;
use uuid::Uuid;

// TODO: Avoid allocations: generate static cstr's at compile time

pub struct FeatureInfo {
    pub project_id: CString,
    pub engine_version: CString,
    pub data_path: Box<[wchar_t]>,
    pub feature_common_info: FeatureCommonInfo,
}

impl FeatureInfo {
    pub fn new(project_id: Uuid) -> Self {
        Self {
            project_id: CString::new(project_id.to_string()).unwrap(),
            engine_version: CString::new(env!("CARGO_PKG_VERSION")).unwrap(),
            data_path: os_str_to_wchar(env::temp_dir().as_os_str()),
            feature_common_info: FeatureCommonInfo::new(),
        }
    }

    pub fn as_nvsdk(&self) -> NVSDK_NGX_FeatureDiscoveryInfo {
        NVSDK_NGX_FeatureDiscoveryInfo {
            SDKVersion: NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            FeatureID: NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
            Identifier: NVSDK_NGX_Application_Identifier {
                IdentifierType: NVSDK_NGX_Application_Identifier_Type_NVSDK_NGX_Application_Identifier_Type_Project_Id,
                v: NVSDK_NGX_Application_Identifier_v {
                    ProjectDesc: NVSDK_NGX_ProjectIdDescription {
                        ProjectId: self.project_id.as_ptr(),
                        EngineType: NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                        EngineVersion: self.engine_version.as_ptr(),
                    },
                },
            },
            ApplicationDataPath: self.data_path.as_ptr(),
            FeatureInfo: &self.feature_common_info.as_nvsdk(),
        }
    }
}

pub struct FeatureCommonInfo {
    path_list: [Box<[wchar_t]>; 2],
}

impl FeatureCommonInfo {
    fn new() -> Self {
        #[cfg(not(target_os = "windows"))]
        let platform = "Linux_x86_64";
        #[cfg(target_os = "windows")]
        let platform = "Windows_x86_64";
        #[cfg(debug_assertions)]
        let profile = "dev";
        #[cfg(not(debug_assertions))]
        let profile = "rel";
        let sdk_path = format!("{}/lib/{platform}/{profile}", env!("DLSS_SDK"));

        Self {
            path_list: [
                os_str_to_wchar(&OsString::from(".")),
                os_str_to_wchar(&OsString::from(sdk_path)),
            ],
        }
    }

    pub fn as_nvsdk(&self) -> NVSDK_NGX_FeatureCommonInfo {
        NVSDK_NGX_FeatureCommonInfo {
            PathListInfo: NVSDK_NGX_PathListInfo {
                Path: self.path_list.as_ptr().cast(),
                Length: self.path_list.len() as u32,
            },
            InternalData: ptr::null_mut(),
            // TODO: Allow configuring logging
            LoggingInfo: NVSDK_NGX_LoggingInfo {
                LoggingCallback: None,
                MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_OFF,
                DisableOtherLoggingSinks: false,
            },
        }
    }
}

#[cfg(target_os = "windows")]
fn os_str_to_wchar(s: &OsStr) -> Box<[wchar_t]> {
    use std::os::windows::ffi::OsStrExt;

    s.encode_wide().map(|c| c as wchar_t).collect()
}

#[cfg(not(target_os = "windows"))]
fn os_str_to_wchar(s: &OsStr) -> Box<[wchar_t]> {
    s.to_str()
        .unwrap_or("")
        .chars()
        .map(|c| c as wchar_t)
        .collect()
}
