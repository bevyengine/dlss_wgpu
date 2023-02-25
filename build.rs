use bindgen::Builder;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get SDK paths
    let dlss_sdk = env::var("DLSS_SDK")
        .expect("DLSS_SDK environment variable not set. Consult the dlss_wgpu readme.");
    let vulkan_sdk = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_dir_path = PathBuf::from(out_dir.clone());

    // Set DLSS static library search path
    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Linux_x86_64");
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Windows_x86_64/x86_64");

    // Link to needed libraries
    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-lib=static=nvsdk_ngx");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/lib");
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/Lib");
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    }

    // Generate rust bindings
    #[cfg(not(target_os = "windows"))]
    let vulkan_sdk_include = "include";
    #[cfg(target_os = "windows")]
    let vulkan_sdk_include = "Include";
    Builder::default()
        .header("src/wrapper.h")
        .clang_arg(format!("-I{dlss_sdk}/include"))
        .clang_arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .allowlist_function("NGX.*")
        .allowlist_function("NVSDK.*")
        .allowlist_type("NVSDK.*")
        .blocklist_type("Vk.*")
        .blocklist_type("PFN_vk.*")
        .wrap_static_fns(true)
        .generate()
        .unwrap()
        .write_to_file(out_dir_path.join("bindings.rs"))
        .unwrap();

    // Generate static library for static inline functions
    link_static_fns(out_dir_path, &dlss_sdk, &vulkan_sdk, vulkan_sdk_include);

    // Copy DLSS shared library to target dir
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

fn link_static_fns(
    out_dir_path: PathBuf,
    dlss_sdk: &str,
    vulkan_sdk: &str,
    vulkan_sdk_include: &str,
) {
    let obj_path = out_dir_path.join("extern.o");

    let clang_output = Command::new("clang")
        .arg("-O")
        .arg("-c")
        .arg("-o")
        .arg(&obj_path)
        .arg(env::temp_dir().join("bindgen").join("extern.c"))
        .arg("-include")
        .arg("src/wrapper.h")
        .arg(format!("-I{dlss_sdk}/include"))
        .arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .output()
        .unwrap();

    if !clang_output.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output.stderr)
        );
    }

    #[cfg(not(target_os = "windows"))]
    let lib_output = Command::new("ar")
        .arg("rcs")
        .arg(out_dir_path.join("libextern.a"))
        .arg(obj_path)
        .output()
        .unwrap();
    #[cfg(target_os = "windows")]
    let lib_output = Command::new("lib").arg(&obj_path).output().unwrap();

    if !lib_output.status.success() {
        panic!(
            "Could not emit library file:\n{}",
            String::from_utf8_lossy(&lib_output.stderr)
        );
    }

    println!(
        "cargo:rustc-link-search=native={}",
        out_dir_path.to_string_lossy()
    );
    println!("cargo:rustc-link-lib=static=extern");
}
