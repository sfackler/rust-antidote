//! Mutex and RwLock types that do not poison themselves.
#![warn(missing_docs)]

use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available.
/// Each mutex has a type parameter which represents the data that it is
/// protecting. The data can only be accessed through the RAII guards returned
/// from lock and try_lock, which guarantees that the data is only ever accessed
/// when the mutex is locked.
///
/// Unlike the standard library mutex, this mutex will not poison itself if a
/// thread panics while holding the lock.
pub struct Mutex<T: ?Sized>(sync::Mutex<T>);

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    pub fn new(t: T) -> Mutex<T> {
        Mutex(sync::Mutex::new(t))
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Acquires a mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the local thread until it is available to
    /// acquire the mutex. Upon returning, the thread is the only thread with
    /// the mutex held. An RAII guard is returned to allow scoped unlock of the
    /// lock. When the guard goes out of scope, the mutex will be unlocked.
    pub fn lock(&self) -> MutexGuard<T> {
        MutexGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then Err is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    pub fn try_lock(&self) -> TryLockResult<T> {
        match self.0.try_lock() {
            Ok(t) => Ok(MutexGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(MutexGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Consumes this mutex, returning the underlying data.
    pub fn into_inner(self) -> T where T: Sized {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the Mutex mutably, no actual locking needs to
    /// take place - the mutable borrow statically guarantees no locks exist.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(|e| e.into_inner())
    }
}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure
/// is dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// Deref and DerefMut implementations.
#[must_use]
pub struct MutexGuard<'a, T: ?Sized + 'a>(sync::MutexGuard<'a, T>);

impl<'a, T: ?Sized> Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<'a, T: ?Sized> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}

/// A type alias for the result of a nonblocking locking method.
pub type TryLockResult<'a, T> = Result<MutexGuard<'a, T>, TryLockError>;


/// An error indicating tha the lock could not be acquired at this time because
/// the operation would otherwise block.
#[derive(Debug)]
pub struct TryLockError(());

impl fmt::Display for TryLockError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str(self.description())
    }
}

impl Error for TryLockError {
    fn description(&self) -> &str {
        "try_lock failed because the operation would block"
    }
}
