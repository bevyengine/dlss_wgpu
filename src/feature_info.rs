use crate::nvsdk_ngx::*;
use std::env;
use std::ffi::{CString, OsStr, OsString};
use std::ptr;
use uuid::Uuid;

// TODO: Remove most allocations
pub fn with_feature_info<F, T>(project_id: Uuid, callback: F) -> T
where
    F: FnOnce(&NVSDK_NGX_FeatureDiscoveryInfo) -> T,
{
    let project_id = CString::new(project_id.to_string()).unwrap();
    let engine_version = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    let data_path = os_str_to_wchar(env::temp_dir().as_os_str());

    #[cfg(not(target_os = "windows"))]
    let platform = "Linux_x86_64";
    #[cfg(target_os = "windows")]
    let platform = "Windows_x86_64";
    #[cfg(debug_assertions)]
    let profile = "dev";
    #[cfg(not(debug_assertions))]
    let profile = "rel";
    let sdk_path = format!("{}/lib/{platform}/{profile}", env!("DLSS_SDK"));
    let shared_library_paths = [
        os_str_to_wchar(&OsString::from(".")),
        os_str_to_wchar(&OsString::from(sdk_path)),
    ];

    let shared_library_path_pointers = shared_library_paths
        .iter()
        .map(Vec::as_ptr)
        .collect::<Vec<_>>();

    let feature_info_common = NVSDK_NGX_FeatureCommonInfo {
        PathListInfo: NVSDK_NGX_PathListInfo {
            Path: shared_library_path_pointers.as_ptr(),
            Length: shared_library_paths.len() as u32,
        },
        InternalData: ptr::null_mut(),
        // TODO: Allow configuring logging
        LoggingInfo: NVSDK_NGX_LoggingInfo {
            LoggingCallback: None,
            MinimumLoggingLevel: NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_OFF,
            DisableOtherLoggingSinks: false,
        },
    };

    let feature_info = NVSDK_NGX_FeatureDiscoveryInfo {
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
        ApplicationDataPath: data_path.as_ptr(),
        FeatureInfo: &feature_info_common,
    };

    (callback)(&feature_info)
}

#[cfg(target_os = "windows")]
fn os_str_to_wchar(s: &OsStr) -> Vec<wchar_t> {
    use std::os::windows::ffi::OsStrExt;

    s.encode_wide().chain([0]).map(|c| c as wchar_t).collect()
}

#[cfg(not(target_os = "windows"))]
fn os_str_to_wchar(s: &OsStr) -> Vec<wchar_t> {
    s.to_str()
        .unwrap_or("")
        .chars()
        .chain([0 as char])
        .map(|c| c as wchar_t)
        .collect()
}
