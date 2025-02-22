use bindgen::Builder;
use std::{env, path::PathBuf, process::Command};

fn main() {
    // Get SDK paths
    let dlss_sdk = env::var("DLSS_SDK")
        .expect("DLSS_SDK environment variable not set. Consult the dlss_wgpu readme.");
    let vulkan_sdk = env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not set");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Link to needed libraries
    #[cfg(not(target_os = "windows"))]
    {
        println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Linux_x86_64");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/lib");

        println!("cargo:rustc-link-lib=static=nvsdk_ngx");
        println!("cargo:rustc-link-lib=dylib=vulkan");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-search=native={dlss_sdk}/lib/Windows_x86_64/x64");
        println!("cargo:rustc-link-search=native={vulkan_sdk}/Lib");

        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
    }

    // Generate rust bindings
    #[cfg(not(target_os = "windows"))]
    let vulkan_sdk_include = "include";
    #[cfg(target_os = "windows")]
    let vulkan_sdk_include = "Include";
    Builder::default()
        .header("src/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_dir.join("extern"))
        .clang_arg(format!("-I{dlss_sdk}/include"))
        .clang_arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .allowlist_item(".*NGX.*")
        .generate()
        .unwrap()
        .write_to_file(out_dir.join("bindings.rs"))
        .unwrap();

    // Generate and link a static library for static inline functions
    link_static_fns(out_dir, &dlss_sdk, &vulkan_sdk, vulkan_sdk_include);
}

fn link_static_fns(out_dir: PathBuf, dlss_sdk: &str, vulkan_sdk: &str, vulkan_sdk_include: &str) {
    let obj_path = out_dir.join("extern.o");

    let clang_output = Command::new("clang")
        .arg("-O")
        .arg("-c")
        .arg("-o")
        .arg(&obj_path)
        .arg(out_dir.join("extern.c"))
        .arg("-include")
        .arg("src/wrapper.h")
        .arg(format!("-I{dlss_sdk}/include"))
        .arg(format!("-I{vulkan_sdk}/{vulkan_sdk_include}"))
        .output()
        .expect("Failed to run clang");

    if !clang_output.status.success() {
        panic!(
            "Could not compile object file:\n{}",
            String::from_utf8_lossy(&clang_output.stderr)
        );
    }

    #[cfg(not(target_os = "windows"))]
    let lib_output = Command::new("ar")
        .arg("rcs")
        .arg(out_dir.join("libextern.a"))
        .arg(obj_path)
        .output()
        .expect("Failed to run ar");
    #[cfg(target_os = "windows")]
    let lib_output = Command::new("llvm-lib")
        .arg(&obj_path)
        .output()
        .expect("Failed to run llvm-lib");

    if !lib_output.status.success() {
        panic!(
            "Could not emit library file:\n{}",
            String::from_utf8_lossy(&lib_output.stderr)
        );
    }

    println!(
        "cargo:rustc-link-search=native={}",
        out_dir.to_string_lossy()
    );
    println!("cargo:rustc-link-lib=static=extern");
}
