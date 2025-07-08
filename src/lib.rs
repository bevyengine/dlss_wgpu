//! # dlss_wgpu
//!
//! This crate provides safe Rust bindings for integrating NVIDIA DLSS (Deep Learning Super Sampling) with the `wgpu` graphics API.
//!
//! ## Setup
//! See <https://github.com/JMS55/dlss_wgpu/blob/main/README.md> for setup instructions.
//!
//! For further info on how to integrate DLSS into your application, read `$DLSS_SDK/doc/DLSS_Programming_Guide_Release.pdf`.
//!
//! ## API Usage
//! ```rust
//! use dlss_wgpu::{DlssSdk, DlssContext, DlssPerfQualityMode, DlssFeatureFlags, DlssRenderParameters, request_device};
//!
//! let project_id = Uuid::parse_str("...").unwrap();
//!
//! // Request a wgpu device and queue
//! let (device, queue) = {
//!     match dlss_wgpu::request_device(dlss_project_id, &adapter, &device_descriptor) {
//!         Ok(x) => {
//!             dlss_supported = true;
//!             x
//!         }
//!         // Fallback to standard device request if DLSS is not supported
//!         Err(_) => adapter.request_device(&device_descriptor).await.unwrap(),
//!     }
//! };
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
mod nvsdk_ngx;
mod render_parameters;
mod request_device;
mod sdk;

pub use context::DlssContext;
pub use nvsdk_ngx::{DlssError, DlssFeatureFlags, DlssPerfQualityMode};
pub use render_parameters::{DlssExposure, DlssRenderParameters, DlssTexture};
pub use request_device::{RequestDeviceError, request_device};
pub use sdk::DlssSdk;
