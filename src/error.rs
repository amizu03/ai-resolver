use core::ffi::CStr;
use derive_more::{Display, From};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Display, Serialize)]
pub enum Error<'a> {
    #[display("Failed to get module handle for '{name:?}'")]
    GetModuleHandle { name: &'a CStr },
    #[display("Failed to get module info for '{name:?}'")]
    GetModuleInfo { name: &'a CStr },
    #[display("Failed to get proc address '{name:?}'")]
    GetProcAddress { name: &'a CStr },
    #[display("Failed to apply patch at {module_name:?} + 0x{offset:X} = 0x{:X}", module_base + offset)]
    Patch {
        module_name: &'a CStr,
        module_base: usize,
        offset: usize,
    },
}

pub type Result<'a, T> = core::result::Result<T, Error<'a>>;
