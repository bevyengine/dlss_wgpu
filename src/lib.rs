//! # dlss_wgpu
//!
//! This crate provides Rust bindings for integrating NVIDIA DLSS (Deep Learning Super Sampling) with the `wgpu` graphics API.
//!
//! ## Setup
//! See <https://github.com/JMS55/dlss_wgpu/blob/main/README.md> for setup instructions.
//!
//! This crate only works with wgpu's Vulkan backend. Other backends are not supported.
//!
//! For further info on how to integrate DLSS into your application, read `$DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf`.
//!
//! ## API Usage
//! ```rust
//! use dlss_wgpu::{FeatureSupport, DlssSdk, DlssPerfQualityMode, DlssFeatureFlags};
//! use dlss_wgpu::super_resolution::{DlssSuperResolution, DlssSuperResolutionRenderParameters};
//!
//! let project_id = Uuid::parse_str("...").unwrap();
//! let mut feature_support = FeatureSupport::default();
//!
//! // Initialize wgpu
//! let instance = dlss_wgpu::create_instance(project_id, &instance_descriptor, &mut feature_support).unwrap();
//! let adapter = instance.request_adapter(&adapter_options).await.unwrap();
//! let (device, queue) = dlss_wgpu::request_device(project_id, &adapter, &device_descriptor, &mut feature_support).unwrap();
//!
//! // Check for feature support, if false don't create DLSS resources
//! println!("DLSS supported: {}", feature_support.super_resolution_supported);
//!
//! // Create the SDK once per application
//! let sdk = DlssSdk::new(project_id, device).expect("Failed to create DlssSdk");
//!
//! // Create a DLSS context once per camera or when DLSS settings change
//! let mut context = DlssSuperResolution::new(
//!     camera.output_resolution,
//!     DlssPerfQualityMode::Auto,
//!     DlssFeatureFlags::empty(),
//!     Arc::clone(&sdk),
//!     &device,
//!     &queue,
//! )
//! .expect("Failed to create DlssSuperResolution");
//!
//! // Setup camera settings
//! camera.view_size = context.render_resolution();
//! camera.subpixel_jitter = context.suggested_jitter(frame_number, camera.view_size);
//! camera.mip_bias = context.suggested_mip_bias(camera.view_size);
//!
//! // Encode DLSS render commands
//! let render_parameters = DlssSuperResolutionRenderParameters { ... };
//! context.render(render_parameters, &mut command_encoder, &adapter)
//!     .expect("Failed to render DLSS");
//! ```

mod feature_info;
mod initialization;
mod nvsdk_ngx;
mod sdk;

/// DLSS Super Resolution.
pub mod ray_reconstruction;
/// DLSS Ray Reconstruction.
pub mod super_resolution;

pub use initialization::{FeatureSupport, InitializationError, create_instance, request_device};
pub use nvsdk_ngx::{DlssError, DlssFeatureFlags, DlssPerfQualityMode};
pub use sdk::DlssSdk;
