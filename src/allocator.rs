use core::alloc::{GlobalAlloc, Layout};

use crate::prelude::*;

use core::arch::asm;
use core::ffi::c_void;
use ntapi::ntpebteb::PEB;
use ntapi::ntrtl::RtlEnumProcessHeaps;
use winapi::shared::minwindef::LPVOID;

struct MyAllocator;

fn peb() -> *mut PEB {
    let mut peb: *mut PEB;

    unsafe {
        asm!(
            "mov {peb}, gs:0x60",
            peb = out(reg) peb
        );
    }

    peb
}

fn find_heap<F: Fn(usize) -> bool>(callback: F) -> Option<usize> {
    let peb = peb();
    let num_heaps = unsafe { (*peb).NumberOfHeaps } as usize;

    for i in 0..num_heaps {
        let heap = unsafe { *(*peb).ProcessHeaps.add(i) };

        if callback(heap as usize) {
            return Some(heap as usize);
        }
    }

    None
}

static mut HEAP: usize = 0;

unsafe impl GlobalAlloc for MyAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        use ntapi::ntrtl::RtlAllocateHeap;

        if HEAP == 0 {
            unsafe extern "system" fn enum_heap_callback(
                heap_handle: LPVOID,
                parameter: LPVOID,
            ) -> i32 {
                if HEAP == 0 {
                    HEAP = heap_handle as _;
                }

                0
            }

            RtlEnumProcessHeaps(Some(enum_heap_callback), core::ptr::null_mut());
            // HEAP = find_heap(|heap| heap != 0).unwrap();
        }

        unsafe { RtlAllocateHeap(HEAP as _, 0, layout.size()) as _ }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        use ntapi::ntrtl::RtlFreeHeap;

        unsafe {
            RtlFreeHeap(HEAP as _, 0, ptr as _);
        }
    }
}

#[global_allocator]
static GLOBAL: MyAllocator = MyAllocator;
