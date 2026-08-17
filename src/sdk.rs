use crate::{feature_info::with_feature_info, nvsdk_ngx::*};
use std::{
    ptr,
    sync::{Arc, Mutex},
    thread,
};
use uuid::Uuid;
use wgpu::{Device, hal::api::Vulkan};

/// Application-wide DLSS object.
pub struct DlssSdk {
    pub(crate) parameters: *mut NVSDK_NGX_Parameter,
    pub(crate) device: Device,
    feature_supported: [bool; DlssFeature::ALL.len()],
    multi_frame_count_max: u32,
}

/// DLSS features whose runtime availability [`DlssSdk::feature_supported`] can report.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DlssFeature {
    SuperResolution,
    RayReconstruction,
    FrameGeneration,
}

impl DlssFeature {
    pub(crate) const ALL: [DlssFeature; 3] = [
        DlssFeature::SuperResolution,
        DlssFeature::RayReconstruction,
        DlssFeature::FrameGeneration,
    ];

    pub(crate) fn ngx_feature(self) -> NVSDK_NGX_Feature {
        match self {
            DlssFeature::SuperResolution => NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
            DlssFeature::RayReconstruction => NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction,
            DlssFeature::FrameGeneration => NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration,
        }
    }

    fn availability_key(self) -> &'static [u8] {
        match self {
            DlssFeature::SuperResolution => NVSDK_NGX_Parameter_SuperSampling_Available,
            DlssFeature::RayReconstruction => NVSDK_NGX_Parameter_SuperSamplingDenoising_Available,
            DlssFeature::FrameGeneration => NVSDK_NGX_Parameter_FrameGeneration_Available,
        }
    }
}

impl DlssSdk {
    /// Creates the DLSS SDK.
    ///
    /// This should be done once per application.
    pub fn new(project_id: Uuid, device: Device) -> Result<Arc<Mutex<Self>>, DlssError> {
        check_for_updates(project_id);

        let mut parameters = ptr::null_mut();
        unsafe {
            let hal_device = device.as_hal::<Vulkan>().unwrap();
            let shared_instance = hal_device.shared_instance();
            let raw_instance = shared_instance.raw_instance();

            with_feature_info(project_id, Default::default(), |feature_info| {
                check_ngx_result(NVSDK_NGX_VULKAN_Init_with_ProjectID(
                    feature_info.Identifier.v.ProjectDesc.ProjectId,
                    NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                    feature_info.Identifier.v.ProjectDesc.EngineVersion,
                    feature_info.ApplicationDataPath,
                    raw_instance.handle(),
                    hal_device.raw_physical_device(),
                    hal_device.raw_device().handle(),
                    shared_instance.entry().static_fn().get_instance_proc_addr,
                    raw_instance.fp_v1_0().get_device_proc_addr,
                    feature_info.FeatureInfo,
                    NVSDK_NGX_Version_NVSDK_NGX_Version_API,
                ))
            })?;

            check_ngx_result(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters))?;

            let mut feature_supported = [false; DlssFeature::ALL.len()];
            for feature in DlssFeature::ALL {
                // A failed query means the driver doesn't know the feature
                feature_supported[feature as usize] =
                    get_i32(parameters, feature.availability_key()).unwrap_or(0) != 0;
            }
            if !feature_supported.contains(&true) {
                check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(parameters))?;
                return Err(DlssError::FeatureNotSupported);
            }
            let multi_frame_count_max =
                get_u32(parameters, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax).unwrap_or(0);

            Ok(Arc::new(Mutex::new(Self {
                parameters,
                device,
                feature_supported,
                // Absent or zero means only single-frame generation is supported
                multi_frame_count_max: multi_frame_count_max.max(1),
            })))
        }
    }

    /// Returns the number of bytes of VRAM allocated by DLSS.
    pub fn get_vram_allocated_bytes(&mut self) -> Result<u64, DlssError> {
        let mut vram_allocated_bytes = 0;
        check_ngx_result(unsafe {
            NGX_DLSS_GET_STATS(self.parameters, &mut vram_allocated_bytes)
        })?;
        Ok(vram_allocated_bytes)
    }

    /// Returns whether the NGX runtime reports the given feature as available on this system.
    ///
    /// This can be false even when the required device extensions are present, for example
    /// on an unsupported GPU.
    pub fn feature_supported(&self, feature: DlssFeature) -> bool {
        self.feature_supported[feature as usize]
    }

    /// Returns the maximum number of frames DLSS Frame Generation can generate between each
    /// pair of rendered frames.
    ///
    /// 1 means only single-frame generation for 2x output, 3 means up to 4x.
    /// Only meaningful when [`DlssFeature::FrameGeneration`] is supported.
    pub fn multi_frame_count_max(&self) -> u32 {
        self.multi_frame_count_max
    }
}

fn check_for_updates(project_id: Uuid) {
    thread::spawn(move || {
        for feature in DlssFeature::ALL {
            with_feature_info(project_id, feature.ngx_feature(), |feature_info| unsafe {
                NVSDK_NGX_UpdateFeature(&feature_info.Identifier, feature_info.FeatureID);
            });
        }
    });
}

impl Drop for DlssSdk {
    fn drop(&mut self) {
        unsafe {
            let hal_device = self.device.as_hal::<Vulkan>().unwrap();
            hal_device
                .raw_device()
                .device_wait_idle()
                .expect("Failed to wait for idle device when destroying DlssSdk");

            check_ngx_result(NVSDK_NGX_VULKAN_DestroyParameters(self.parameters))
                .expect("Failed to destroy DlssSdk parameters");
            check_ngx_result(NVSDK_NGX_VULKAN_Shutdown1(hal_device.raw_device().handle()))
                .expect("Failed to destroy DlssSdk");
        }
    }
}

unsafe impl Send for DlssSdk {}
unsafe impl Sync for DlssSdk {}
