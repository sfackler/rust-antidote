//! Mutex and RwLock types that do not poison themselves.
#![warn(missing_docs)]

use std::error::Error;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync;

/// A mutual exclusion primitive useful for protecting shared data.
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
    pub fn lock<'a>(&'a self) -> MutexGuard<'a, T> {
        MutexGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then Err is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    ///
    /// This function does not block.
    pub fn try_lock<'a>(&'a self) -> TryLockResult<MutexGuard<'a, T>> {
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
pub type TryLockResult<T> = Result<T, TryLockError>;

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

/// A reader-writer lock.
///
/// This type of lock allows a number of readers or at most one writer at any
/// point in time. The write portion of this lock typically allows modification
/// of the underlying data (exclusive access) and the read portion of this lock
/// typically allows for read-only access (shared access).
///
/// The priority policy of the lock is dependent on the underlying operating
/// system's implementation, and this type does not guarantee that any
/// particular policy will be used.
///
/// The type parameter T represents the data that this lock protects. It is
/// required that T satisfies Send to be shared across threads and Sync to allow
/// concurrent access through readers. The RAII guards returned from the locking
/// methods implement Deref (and DerefMut for the write methods) to allow access
/// to the contained of the lock.
///
/// Unlike the standard library RwLock, this lock will not poison itself if a
/// thread panics while holding the lock.
pub struct RwLock<T: ?Sized>(sync::RwLock<T>);

impl<T> RwLock<T> {
    /// Creates a new instance of an `RwLock<T>` which is unlocked.
    pub fn new(t: T) -> RwLock<T> {
        RwLock(sync::RwLock::new(t))
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// The calling thread will be blocked until there are no more writers which
    /// hold the lock. There may be other readers currently inside the lock when
    /// this method returns. This method does not provide any guarantees with
    /// respect to the ordering of whether contentious readers or writers will
    /// acquire the lock first.
    ///
    /// Returns an RAII guard which will release this thread's shared access
    /// once it is dropped.
    pub fn read<'a>(&'a self) -> RwLockReadGuard<'a, T> {
        RwLockReadGuard(self.0.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Attempts to acquire this rwlock with shared read access.
    ///
    /// If the access could not be granted at this time, then `Err` is returned.
    /// Otherwise, an RAII guard is returned which will release the shared
    /// access when it is dropped.
    ///
    /// This function does not block.
    ///
    /// This function does not provide any guarantees with respect to the
    /// ordering of whether contentious readers or writers will acquire the lock
    /// first.
    pub fn try_read<'a>(&'a self) -> TryLockResult<RwLockReadGuard<'a, T>> {
        match self.0.try_read() {
            Ok(t) => Ok(RwLockReadGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(RwLockReadGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Locks this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this rwlock
    /// when dropped.
    pub fn write<'a>(&'a self) -> RwLockWriteGuard<'a, T> {
        RwLockWriteGuard(self.0.write().unwrap_or_else(|e| e.into_inner()))
    }

    /// Attempts to lock this rwlock with exclusive write access.
    ///
    /// If the lock could not be acquired at this time, then `Err` is returned.
    /// Otherwise, an RAII guard is returned which will release the lock when
    /// it is dropped.
    ///
    /// This function does not block.
    ///
    /// This function does not provide any guarantees with respect to the
    /// ordering of whether contentious readers or writers will acquire the lock
    /// first.
    pub fn try_write<'a>(&'a self) -> TryLockResult<RwLockWriteGuard<'a, T>> {
        match self.0.try_write() {
            Ok(t) => Ok(RwLockWriteGuard(t)),
            Err(sync::TryLockError::Poisoned(e)) => Ok(RwLockWriteGuard(e.into_inner())),
            Err(sync::TryLockError::WouldBlock) => Err(TryLockError(())),
        }
    }

    /// Consumes this `RwLock`, returning the underlying data.
    pub fn into_inner(self) -> T where T: Sized {
        self.0.into_inner().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `RwLock` mutably, no actual locking needs to
    /// take place - the mutable borrow statically guarantees no locks exist.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut().unwrap_or_else(|e| e.into_inner())
    }
}

/// RAII structure used to release the shared read access of a lock when
/// dropped.
#[must_use]
pub struct RwLockReadGuard<'a, T: ?Sized + 'a>(sync::RwLockReadGuard<'a, T>);

impl<'a, T: ?Sized> Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.deref()
    }
}

/// RAII structure used to release the exclusive write access of a lock when
/// dropped.
#[must_use]
pub struct RwLockWriteGuard<'a, T: ?Sized + 'a>(sync::RwLockWriteGuard<'a, T>);

impl<'a, T: ?Sized> Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.deref()
    }
}

impl<'a, T: ?Sized> DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.deref_mut()
    }
}
