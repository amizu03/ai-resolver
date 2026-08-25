#![no_std]
#![no_main]
#![feature(
    const_trait_impl,
    stmt_expr_attributes,
    naked_functions,
    rustc_private,
    optimize_attribute,
    allow_internal_unstable,
    c_variadic,
    generic_arg_infer,
    generic_const_exprs,
    associated_type_defaults,
    array_ptr_get,
    let_chains
)]
#![windows_subsystem = "console"]
#![allow(warnings, dead_code, static_mut_refs)]

extern crate alloc;

#[macro_use]
extern crate static_assertions;

mod allocator;
mod error;
mod hooks;
mod neural_network;
pub(crate) mod prelude;
mod utils;
mod winapi;

use crate::prelude::*;

#[cfg(not(test))]
use core::panic::PanicInfo;

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn start_rs<'a>() -> Result<'a, ()> {
    use ::winapi::um::synchapi::Sleep;
    use ::winapi::um::winuser::{GetAsyncKeyState, VK_END};

    // apply patches
    let hooks = hooks::init()?;

    // wait for unload key press
    unsafe {
        while GetAsyncKeyState(VK_END) == 0 {
            // loop {
            Sleep(250);
        }
    }

    Ok(())
}

unsafe extern "stdcall" fn start_c(instance: usize) {
    use ::winapi::um::libloaderapi::FreeLibraryAndExitThread;

    println!("start_c");
    let _ = dbg!(start_rs());

    println!("Unloading...");
    unsafe {
        FreeLibraryAndExitThread(instance as _, 0);
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn start(base: usize, reason: u32, _reserved: usize) -> i32 {
    if reason == 1 {
        use ::winapi::um::processthreadsapi::CreateThread;

        dbg!(reason);

        unsafe {
            CreateThread(
                null_mut(),
                0,
                Some(transmute(start_c as usize)),
                base as _,
                0,
                null_mut(),
            );
        }
    }

    1
}
