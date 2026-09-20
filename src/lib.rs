//! `Mutex` and `RwLock` types that do not poison themselves.
//!
//! These types expose identical APIs to the standard library `Mutex` and
//! `RwLock` except that they do not return `PoisonError`s.

#[doc(inline)]
pub use std::sync::WaitTimeoutResult;
use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync,
    time::Duration,
};

#[derive(Debug, Default)]
#[repr(transparent)]
/// Like `std::sync::Mutex` except that it does not poison itself.
pub struct Mutex<T: ?Sized>(sync::Mutex<T>);

impl<T> Mutex<T> {
    /// Like `std::sync::Mutex::new`.
    #[inline]
    pub const fn new(t: T) -> Mutex<T> {
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
    pub fn lock(&self) -> MutexGuard<'_, T> {
        MutexGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::Mutex::try_lock`.
    #[inline]
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
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

#[derive(Debug)]
#[repr(transparent)]
#[must_use]
/// Like `std::sync::MutexGuard`.
pub struct MutexGuard<'a, T: ?Sized + 'a>(sync::MutexGuard<'a, T>);

impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}

#[derive(Debug, Default)]
#[repr(transparent)]
/// Like `std::sync::Condvar`.
pub struct Condvar(sync::Condvar);

impl Condvar {
    /// Like `std::sync::Condvar::new`.
    #[inline]
    pub const fn new() -> Condvar {
        Condvar(sync::Condvar::new())
    }

    /// Like `std::sync::Condvar::wait`.
    #[inline]
    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        MutexGuard(self.0.wait(guard.0).unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::Condvar::wait_timeout`.
    #[inline]
    pub fn wait_timeout<'a, T>(
        &self,
        guard: MutexGuard<'a, T>,
        dur: Duration,
    ) -> (MutexGuard<'a, T>, WaitTimeoutResult) {
        let (guard, result) = self
            .0
            .wait_timeout(guard.0, dur)
            .unwrap_or_else(|e| e.into_inner());
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
        fmt.write_str("lock call failed because the operation would block")
    }
}

#[derive(Debug, Default)]
#[repr(transparent)]
/// Like `std::sync::RwLock` except that it does not poison itself.
pub struct RwLock<T: ?Sized>(sync::RwLock<T>);

impl<T> RwLock<T> {
    /// Like `std::sync::RwLock::new`.
    #[inline]
    pub const fn new(t: T) -> RwLock<T> {
        RwLock(sync::RwLock::new(t))
    }

    /// Like `std::sync::RwLock::into_inner`.
    #[inline]
    pub fn into_inner(self) -> T
    where
        T: Sized,
    {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Like `std::sync::RwLock::read`.
    #[inline]
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        RwLockReadGuard(self.0.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::RwLock::try_read`.
    #[inline]
    pub fn try_read(&self) -> TryLockResult<RwLockReadGuard<'_, T>> {
        match self.0.try_read() {
            Ok(t) => Ok(RwLockReadGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(RwLockReadGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Like `std::sync::RwLock::write`.
    #[inline]
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        RwLockWriteGuard(self.0.write().unwrap_or_else(|e| e.into_inner()))
    }

    /// Like `std::sync::RwLock::try_write`.
    #[inline]
    pub fn try_write(&self) -> TryLockResult<RwLockWriteGuard<'_, T>> {
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

#[derive(Debug)]
#[repr(transparent)]
#[must_use]
/// Like `std::sync::RwLockReadGuard`.
pub struct RwLockReadGuard<'a, T: ?Sized + 'a>(sync::RwLockReadGuard<'a, T>);

impl<T: ?Sized> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

#[derive(Debug)]
#[repr(transparent)]
#[must_use]
/// Like `std::sync::RwLockWriteGuard`.
pub struct RwLockWriteGuard<'a, T: ?Sized + 'a>(sync::RwLockWriteGuard<'a, T>);

impl<T: ?Sized> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<T: ?Sized> DerefMut for RwLockWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}
