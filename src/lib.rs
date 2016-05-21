//! Mutex and RwLock types that do not poison themselves.
//!
//! These types expose identical APIs to the standard library `Mutex` and
//! `RwLock` except that they do not return `PoisonError`s.
#![doc(html_root_url="https://sfackler.github.io/rust-antidote/doc/v1.0.0")]
#![warn(missing_docs)]

use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync;
use std::time::Duration;

#[doc(inline)]
pub use std::sync::WaitTimeoutResult;

/// Like `std::sync::Mutex` except that it does not poison itself.
pub struct Mutex<T: ?Sized>(sync::Mutex<T>);

impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, fmt)
    }
}

impl<T> Mutex<T> {
    /// Like `std::sync::Mutex::new`.
    #[inline]
    pub fn new(t: T) -> Mutex<T> {
        Mutex(sync::Mutex::new(t))
    }

    /// Like `std::sync::Mutex::into_inner`.
    #[inline]
    pub fn into_inner(self) -> T {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Like `std::sync::Mutex::lock`.
    #[inline]
    pub fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
        MutexGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::Mutex::try_lock`.
    #[inline]
    pub fn try_lock<'a>(&'a self) -> TryLockResult<MutexGuard<'a, T>> {
        match self.0.try_lock() {
            Ok(t) => Ok(MutexGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(MutexGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Like `std::sync::Mutex::get_mut`.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(|e| e.into_inner())
    }
}

/// Like `std::sync::MutexGuard`.
#[must_use]
pub struct MutexGuard<'a, T: ?Sized + 'a>(sync::MutexGuard<'a, T>);

impl<'a, T: ?Sized> Deref for MutexGuard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<'a, T: ?Sized> DerefMut for MutexGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}

/// Like `std::sync::Condvar`.
pub struct Condvar(sync::Condvar);

impl Condvar {
    /// Like `std::sync::Condvar::new`.
    #[inline]
    pub fn new() -> Condvar {
        Condvar(sync::Condvar::new())
    }

    /// Like `std::sync::Condvar::wait`.
    #[inline]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        MutexGuard(self.0.wait(guard.0).unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::Condvar::wait_timeout`.
    #[inline]
    pub fn wait_timeout<'a, T>(&self,
                               guard: MutexGuard<'a, T>,
                               dur: Duration)
                               -> (MutexGuard<'a, T>, WaitTimeoutResult) {
        let (guard, result) = self.0.wait_timeout(guard.0, dur).unwrap_or_else(|e| e.into_inner());
        (MutexGuard(guard), result)
    }

    /// Like `std::sync::Condvar::notify_one`.
    #[inline]
    pub fn notify_one(&self) {
        self.0.notify_one()
    }

    /// Like `std::sync::Condvar::notify_all`.
    #[inline]
    pub fn notify_all(&self) {
        self.0.notify_all()
    }
}

/// Like `std::sync::TryLockResult`.
pub type TryLockResult<T> = Result<T, TryLockError>;

/// Like `std::sync::TryLockError`.
#[derive(Debug)]
pub struct TryLockError(());

impl fmt::Display for TryLockError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(self.description())
    }
}

impl Error for TryLockError {
    fn description(&self) -> &str {
        "lock call failed because the operation would block"
    }
}

/// Like `std::sync::RwLock` except that it does not poison itself.
pub struct RwLock<T: ?Sized>(sync::RwLock<T>);

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, fmt)
    }
}

impl<T> RwLock<T> {
    /// Like `std::sync::RwLock::new`.
    #[inline]
    pub fn new(t: T) -> RwLock<T> {
        RwLock(sync::RwLock::new(t))
    }

    /// Like `std::sync::RwLock::into_inner`.
    #[inline]
    pub fn into_inner(self) -> T where T: Sized {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Like `std::sync::RwLock::read`.
    #[inline]
    pub fn read<'a>(&'a self) -> RwLockReadGuard<'a, T> {
        RwLockReadGuard(self.0.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::RwLock::try_read`.
    #[inline]
    pub fn try_read<'a>(&'a self) -> TryLockResult<RwLockReadGuard<'a, T>> {
        match self.0.try_read() {
            Ok(t) => Ok(RwLockReadGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(RwLockReadGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Like `std::sync::RwLock::write`.
    #[inline]
    pub fn write<'a>(&'a self) -> RwLockWriteGuard<'a, T> {
        RwLockWriteGuard(self.0.write().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::RwLock::try_write`.
    #[inline]
    pub fn try_write<'a>(&'a self) -> TryLockResult<RwLockWriteGuard<'a, T>> {
        match self.0.try_write() {
            Ok(t) => Ok(RwLockWriteGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(RwLockWriteGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Like `std::sync::RwLock::get_mut`.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(|e| e.into_inner())
    }
}

/// Like `std::sync::RwLockReadGuard`.
#[must_use]
pub struct RwLockReadGuard<'a, T: ?Sized + 'a>(sync::RwLockReadGuard<'a, T>);

impl<'a, T: ?Sized> Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

/// Like `std::sync::RwLockWriteGuard`.
#[must_use]
pub struct RwLockWriteGuard<'a, T: ?Sized + 'a>(sync::RwLockWriteGuard<'a, T>);

impl<'a, T: ?Sized> Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<'a, T: ?Sized> DerefMut for RwLockWriteGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}
