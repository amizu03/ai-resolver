use crate::{dbg, prelude::*, println};

use winapi::um::{
    libloaderapi::*, memoryapi::*, processenv::*, processthreadsapi::*, psapi::*, synchapi::Sleep,
    wincon::*, winnt::*,
};

#[derive(Serialize, Debug)]
pub struct Module<'a> {
    pub name: &'a CStr,
    pub memory: &'a mut [u8],
}

impl<'a> Module<'a> {
    pub fn patch<const N: usize>(
        self,
        offset: usize,
        target: *const (),
    ) -> Result<'a, Patch<'a, 'a, N>>
    where
        [(); N + 5]:,
    {
        println!("patch");
        let bytes = unsafe { transmute::<_, *mut [u8; N]>(self.memory.as_ptr().add(offset)) };
        let trampoline = unsafe {
            VirtualAlloc(
                null_mut(),
                N + 5 + 5,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8
        };

        println!("Writing trampoline...");
        unsafe {
            let trampoline_start = trampoline as *mut [u8; N];

            unsafe {
                *trampoline_start = bytes.read_unaligned();
                trampoline.add(N).write_unaligned(0xE8);
                (trampoline.add(N + 1) as *mut u32)
                    .write_unaligned((target as usize - (trampoline as usize + N + 5)) as u32);
                trampoline.add(N + 5).write_unaligned(0xE9);
                (trampoline.add(N + 5 + 1) as *mut u32).write_unaligned(
                    (bytes.as_mut_ptr() as usize + N - (trampoline as usize + N + 5 + 5)) as u32,
                );
            }
        }

        println!("Wrote patch!");

        Ok(Patch {
            module: Module {
                name: self.name,
                memory: self.memory,
            },
            offset,
            trampoline: trampoline as _,
            backup_bytes: unsafe { bytes.read_unaligned() },
            enabled: AtomicBool::new(false),
        })
    }
}

#[derive(Debug)]
pub struct Patch<'a, 'b, const N: usize>
where
    [(); N + 5]:,
{
    module: Module<'a>,
    trampoline: *mut [u8; N + 5],
    offset: usize,
    backup_bytes: [u8; N],
    enabled: AtomicBool,
}

use portable_atomic::Ordering;

impl<'a, 'b, const N: usize> Patch<'a, 'b, N>
where
    [(); N + 5]:,
{
    fn unprotected<F: Fn()>(&self, callback: F) {
        unsafe {
            let mut old_protection = 0;
            VirtualProtect(
                self.module.memory.as_ptr().byte_offset(self.offset as _) as _,
                N,
                PAGE_EXECUTE_READWRITE,
                &mut old_protection,
            );
            callback();
            VirtualProtect(
                self.module.memory.as_ptr().byte_offset(self.offset as _) as _,
                N,
                old_protection,
                &mut old_protection,
            );
        }
    }

    pub fn enable(&self) -> Result<()> {
        if self
            .enabled
            .compare_exchange(false, true, Ordering::Acquire, Ordering::SeqCst)
            .is_err()
        {
            return Err(Error::Patch {
                module_name: self.module.name,
                module_base: self.module.memory.as_ptr() as _,
                offset: self.offset,
            });
        }

        self.unprotected(|| {
            unsafe {
                let code = self.module.memory.as_ptr().byte_add(self.offset) as *mut u8;
                let rel32 = (self.trampoline as usize).unchecked_sub(code as usize + 5);
                // dbg!(rel32);
                (code as *mut u8).write_unaligned(0xE9);
                (code as *mut u32).byte_add(1).write_unaligned(rel32 as u32);
            }
        });

        Ok(())
    }

    pub fn disable(&self) -> Result<()> {
        if self
            .enabled
            .compare_exchange(true, false, Ordering::Acquire, Ordering::SeqCst)
            .is_err()
        {
            return Err(Error::Patch {
                module_name: self.module.name,
                module_base: self.module.memory.as_ptr() as _,
                offset: self.offset,
            });
        }

        self.unprotected(|| unsafe {
            let code = self.module.memory.as_ptr().byte_add(self.offset) as *mut u8;

            for i in 0..self.backup_bytes.len() {
                *code.add(i) = self.backup_bytes[i];
            }
        });

        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

impl<const N: usize> Drop for Patch<'_, '_, N>
where
    [(); N + 5]:,
{
    fn drop(&mut self) {
        self.disable();

        unsafe {
            Sleep(100);
            VirtualFree(self.trampoline as _, 0, MEM_RELEASE);
        }
    }
}

impl<'a> Module<'a> {
    pub fn from_name(name: &'a CStr) -> Result<Self> {
        let module_base = unsafe { GetModuleHandleA(name.as_ptr() as _) };

        if module_base.is_null() {
            return Err(Error::GetModuleHandle { name });
        }

        let mut module_info = Default::default();

        if unsafe {
            GetModuleInformation(
                GetCurrentProcess(),
                module_base,
                &mut module_info,
                size_of_val(&module_info) as _,
            )
        } == 0
        {
            Err(Error::GetModuleInfo { name })
        } else {
            Ok(Module {
                name,
                memory: unsafe {
                    core::slice::from_raw_parts_mut(module_base as _, module_info.SizeOfImage as _)
                },
            })
        }
    }

    pub fn proc<T: Sized>(&self, name: &'a CStr) -> Result<T> {
        let proc = unsafe { GetProcAddress(self.memory.as_ptr() as _, name.as_ptr() as _) };

        if proc.is_null() {
            Err(Error::GetProcAddress { name })
        } else {
            Ok(unsafe { transmute_copy(&proc) })
        }
    }
}
