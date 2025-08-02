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
//! use dlss_wgpu::{DlssSdk, DlssContext, DlssPerfQualityMode, DlssFeatureFlags, DlssRenderParameters};
//!
//! let project_id = Uuid::parse_str("...").unwrap();
//! let mut dlss_supported = true;
//!
//! // Initialize wgpu
//! let instance = dlss_wgpu::create_instance(project_id, &instance_descriptor, &mut dlss_supported).unwrap();
//! let adapter = instance.request_adapter(&adapter_options).await.unwrap();
//! let (device, queue) = dlss_wgpu::request_device(project_id, &adapter, &device_descriptor, &mut dlss_supported).unwrap();
//!
//! // Check `dlss_supported`, if false don't create DLSS resources
//! println!("DLSS supported: {dlss_supported}");
//!
//! // Create the SDK once per application
//! let sdk = DlssSdk::new(project_id, device).expect("Failed to create DLSS SDK");
//!
//! // Create a DLSS context once per camera or when DLSS settings change
//! let mut context = DlssContext::new(
//!     camera.output_resolution,
//!     DlssPerfQualityMode::Auto,
//!     DlssFeatureFlags::empty(),
//!     Arc::clone(&sdk),
//!     &device,
//!     &queue,
//! )
//! .expect("Failed to create DLSS context");
//!
//! // Setup camera settings
//! camera.view_size = context.render_resolution();
//! camera.subpixel_jitter = context.suggested_jitter(frame_number, camera.view_size);
//! camera.mip_bias = context.suggested_mip_bias(camera.view_size);
//!
//! // Encode DLSS render commands
//! let render_parameters = DlssRenderParameters { ... };
//! context.render(render_parameters, &mut command_encoder, &adapter)
//!     .expect("Failed to render DLSS");
//! ```

mod context;
mod feature_info;
mod initialization;
mod nvsdk_ngx;
mod render_parameters;
mod sdk;

pub use context::DlssContext;
pub use initialization::{InitializationError, create_instance, request_device};
pub use nvsdk_ngx::{DlssError, DlssFeatureFlags, DlssPerfQualityMode};
pub use render_parameters::{DlssExposure, DlssRenderParameters};
pub use sdk::DlssSdk;
