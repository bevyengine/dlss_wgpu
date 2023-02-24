use bindgen::Builder;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let dlss_sdk = env::var("DLSS_SDK")
        .expect("DLSS_SDK environment variable not set. Consult the dlss_wgpu readme.");
    let vulkan_sdk = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let out_dir = env::var("OUT_DIR").unwrap();

    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Linux_x86_64");
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Windows_x86_64/x86_64");

    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-lib=dylib=sdk_nvngx");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/lib");
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=nvsdk_ngx_d");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/Lib");
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    }

    #[cfg(not(target_os = "windows"))]
    let vulkan_sdk_include = "include";
    #[cfg(target_os = "windows")]
    let vulkan_sdk_include = "Include";
    Builder::default()
        .header("src/wrapper.h")
        .clang_arg(format!("-I{dlss_sdk}/include"))
        .clang_arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .allowlist_function("NVSDK.*")
        .allowlist_type("NVSDK.*")
        .blocklist_type("Vk.*")
        .blocklist_type("PFN_vk.*")
        .generate()
        .unwrap()
        .write_to_file(PathBuf::from(out_dir.clone()).join("bindings.rs"))
        .unwrap();

    #[cfg(not(target_os = "windows"))]
    let (platform, shared_lib) = ("Linux_x86_64", "libnvidia-ngx-dlss.so.3.1.1");
    #[cfg(target_os = "windows")]
    let (platform, shared_lib) = ("Windows_x86_64", "nvngx_dlss.dll");
    #[cfg(debug_assertions)]
    let profile = "dev";
    #[cfg(not(debug_assertions))]
    let profile = "rel";
    fs::copy(
        format!("{dlss_sdk}/lib/{platform}/{profile}/{shared_lib}"),
        format!("{out_dir}/../../{shared_lib}"),
    )
    .unwrap();
}
