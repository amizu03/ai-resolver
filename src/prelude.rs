pub(crate) use alloc::boxed::Box;

pub(crate) use alloc::ffi::CString;
pub(crate) use alloc::string::{String, ToString};
pub(crate) use alloc::sync::Arc;
pub(crate) use alloc::vec;
pub(crate) use alloc::vec::Vec;
pub(crate) use core::arch::{asm, naked_asm};
pub(crate) use core::cell::OnceCell;
pub(crate) use core::ffi::CStr;
pub(crate) use core::mem::{size_of, size_of_val, transmute, transmute_copy, zeroed};
pub(crate) use core::pin::Pin;
pub(crate) use core::ptr::{null, null_mut};
pub(crate) use core::slice;
pub(crate) use derive_more::{Display, From};
pub(crate) use portable_atomic::AtomicBool;
pub(crate) use serde::{Deserialize, Serialize};

pub(crate) use crate::error::*;
pub(crate) use crate::utils::*;
pub(crate) use crate::winapi::*;

pub(crate) use nalgebra as na;
pub(crate) type Matrix<const R: usize, const C: usize> = na::Matrix<f64, f64, f64, [[f64; C]; R]>;
pub type Vector3 = na::Vector3<f32>;
pub type Vector2 = na::Vector2<f32>;
