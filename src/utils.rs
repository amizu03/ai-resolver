use crate::prelude::*;

use winapi::um::{libloaderapi::*, processenv::GetStdHandle, wincon::*};

#[macro_export]
#[allow_internal_unstable(print_internals, format_args_nl)]
macro_rules! println {
    () => {
        $crate::utils::print("\n\0")
    };
    ($($arg:tt)*) => {{
        use alloc::string::ToString;
        $crate::utils::print(&fmtools::fmt!(($($arg)*)"\n\0").to_string());
    }};
}

#[macro_export]
macro_rules! dbg {
    // NOTE: We cannot use `concat!` to make a static string as a format argument
    // of `eprintln!` because `file!` could contain a `{` or
    // `$val` expression could be a block (`{ .. }`), in which case the `eprintln!`
    // will be malformed.
    () => {
        $crate::println!("["{file!()}":"{line!()}":"{column!()}"]")
    };
    ($val:expr $(,)?) => {
        // Use of `match` here is intentional because it affects the lifetimes
        // of temporaries - https://stackoverflow.com/a/48732525/1063961
        match $val {
            tmp => {
                $crate::println!("["{file!()}":"{line!()}":"{column!()}"] "{stringify!($val)}" = "{&tmp:#?});
                tmp
            }
        }
    };
    ($($val:expr),+ $(,)?) => {
        ($($crate::dbg!($val)),+,)
    };
}

pub(crate) fn print(string: &str) {
    use winapi::um::consoleapi::{AllocConsole, WriteConsoleA};

    static mut CONSOLE_HANDLES: OnceCell<(usize, usize, unsafe extern "C" fn(*const u8))> =
        OnceCell::new();

    let (stdout, _, msg) = unsafe { &CONSOLE_HANDLES }.get_or_init(|| {
        let msg = Module::from_name(c"tier0.dll")
            .unwrap()
            .proc(c"Msg")
            .unwrap();

        unsafe {
            AllocConsole();
            SetConsoleCP(65001);
            SetConsoleOutputCP(65001);
        }

        let ucrtbase = Module::from_name(c"ucrtbase.dll").unwrap();
        let _open_osfhandle: unsafe extern "C" fn(usize, i32) -> i32 =
            ucrtbase.proc(c"_open_osfhandle").unwrap();
        let _fdopen: unsafe extern "C" fn(i32, &CStr) -> i32 = ucrtbase.proc(c"_fdopen").unwrap();

        unsafe {
            let stdout = GetStdHandle(-11i32 as _) as usize;
            let stdin = GetStdHandle(-10i32 as _) as usize;

            _fdopen(_open_osfhandle(stdout, 0x4000), c"w");
            _fdopen(_open_osfhandle(stdin, 0x4000), c"r");

            (stdout, stdin, msg)
        }
    });

    unsafe {
        msg(string.as_ptr());

        WriteConsoleA(
            *stdout as _,
            string.as_ptr() as _,
            string.len() as u32 - 1,
            null_mut(),
            null_mut(),
        );
    }
}
