use std::any::Any;
use std::cell::Ref;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::io;
use std::io::Write;
use std::mem;
use std::ops::Deref;

pub struct LookAheadIterator<TIterator: Iterator> {
    look_ahead: Option<TIterator::Item>,
    iterator: TIterator
}

impl <TIterator: Iterator> LookAheadIterator<TIterator> {
    pub fn new(mut iterator: TIterator) -> LookAheadIterator<TIterator> {
        LookAheadIterator {
            look_ahead: iterator.next(),
            iterator: iterator
        }
    }

    pub fn peek(&self) -> &Option<TIterator::Item> {
        &self.look_ahead
    }

    pub fn pop(&mut self) -> Option<TIterator::Item> {
        let mut elem = self.iterator.next();
        mem::swap(&mut elem, &mut self.look_ahead);
        elem
    }
}

pub struct NullWriter();

impl NullWriter {
    pub fn new() -> NullWriter {
        NullWriter()
    }
}

impl Write for NullWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Result::Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Result::Ok(())
    }
}

pub fn follow<'a, T: Eq + Hash>(m: &'a HashMap<T, T>, v: &'a T) -> &'a T {
    if let Some(next) = m.get(v) {
        follow(m, next)
    } else {
        v
    }
}

pub fn follow_option<'a, T: Eq + Hash>(m: &'a HashMap<T, Option<T>>, v: &'a T) -> Option<&'a T> {
    match m.get(v) {
        Some(&None) => None,
        Some(&Some(ref next)) => follow_option(m, next),
        None => Some(v)
    }
}

pub trait PrettyDisplay where Self: Sized {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result;

    fn pretty(&self) -> PrettyPrinter<Self> {
        PrettyPrinter { val: self, indent: "" }
    }

    fn pretty_indented<'a>(&'a self, indent: &'a str) -> PrettyPrinter<'a, Self> {
        PrettyPrinter { val: self, indent: indent }
    }
}

pub struct PrettyPrinter<'a, T: PrettyDisplay + 'a> {
    val: &'a T,
    indent: &'a str
}

impl <'a, T: PrettyDisplay + 'a> fmt::Display for PrettyPrinter<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.val.fmt(self.indent, f)
    }
}

/// Provides a method of representing a reference to a value that has been accessed through an
/// arbitrary chain of RefCells. All RefCells that were along the path used to arrive at this
/// reference will be kept borrowed as long as this ChainRef lives.
pub struct ChainRef<'a, T: 'a> {
    refs: Vec<Ref<'a, Any>>,
    val: &'a T
}

impl <'a, T: 'static> ChainRef<'a, T> {
    /// Creates a new ChainRef from a simple reference. This ChainRef will represent a chain of no
    /// RefCells leading to the given reference.
    pub fn new(r: &'a T) -> ChainRef<'a, T> {
        ChainRef { refs: vec![], val: r }
    }

    /// Creates a ChainRef from a reference borrowed from a RefCell. This ChainRef will represent a
    /// chain of one RefCell leading to the value referenced by the given RefCell.
    pub fn from_ref(r: Ref<'a, T>) -> ChainRef<'a, T> {
        // This allows us to get a reference to the data referenced by r even though we're about to
        // move it into a vector. This is only safe because we know that, once moved into the
        // vector, r will remain there until the ChainRef is dropped, at which point this reference
        // will be invalid.
        let ri = unsafe { &*(r.deref() as *const T) };

        ChainRef { refs: vec![r], val: ri }
    }

    /// Creates a clone of a ChainRef.
    pub fn clone(orig: &ChainRef<'a, T>) -> ChainRef<'a, T> {
        ChainRef {
            refs: orig.refs.iter().map(|r| Ref::clone(r)).collect(),
            val: orig.val
        }
    }

    /// Maps the reference contained in the given ChainRef into a new reference reached through the
    /// same chain of RefCells.
    pub fn map<U, F: FnOnce(&'a T) -> &'a U>(orig: ChainRef<'a, T>, f: F) -> ChainRef<'a, U> {
        ChainRef {
            refs: orig.refs,
            val: f(orig.val)
        }
    }

    /// Maps the reference contained in the given ChainRef into a new reference reached through the
    /// same chain of RefCells, or returns None.
    pub fn map_option<U, F: FnOnce(&'a T) -> Option<&'a U>>(orig: ChainRef<'a, T>, f: F) -> Option<ChainRef<'a, U>> {
        if let Some(next) = f(orig.val) {
            Some(ChainRef { refs: orig.refs, val: next })
        } else {
            None
        }
    }

    /// Maps the reference contained in the given ChainRef into a new reference reached through a
    /// number of additional RefCells.
    pub fn and_then<U, F: FnOnce(&'a T) -> ChainRef<'a, U>>(orig: ChainRef<'a, T>, f: F) -> ChainRef<'a, U> {
        let ChainRef { refs: mut orig_refs, val: orig_val } = orig;
        let mut next = f(orig_val);

        next.refs.append(&mut orig_refs);
        next
    }

    /// Maps the reference contained in the given ChainRef into a new reference reached through a
    /// number of additional RefCells, or returns None.
    pub fn and_then_option<U, F: FnOnce(&'a T) -> Option<ChainRef<'a, U>>>(orig: ChainRef<'a, T>, f: F) -> Option<ChainRef<'a, U>> {
        let ChainRef { refs: mut orig_refs, val: orig_val } = orig;

        if let Some(mut next) = f(orig_val) {
            next.refs.append(&mut orig_refs);
            Some(next)
        } else {
            None
        }
    }

    /// Creates a new chain containing only the given RefCell, then maps that chain into a new
    /// reference reached through a number of additional RefCells. This is preferred over manually
    /// creating a new ChainRef and then using and_then since this function avoids allocating heap
    /// space for two ChainRefs.
    pub fn and_then_one<U, F: FnOnce(&'a T) -> ChainRef<'a, U>>(orig: Ref<'a, T>, f: F) -> ChainRef<'a, U> {
        // This is safe for the same reason outlined in ChainRef::from_ref
        let orig_val = unsafe { &*(orig.deref() as *const T) };
        let mut next = f(orig_val);

        next.refs.push(orig);
        next
    }

    /// Creates a new chain containing only the given RefCell, then maps that chain into a new
    /// reference reached through a number of additional RefCells, or returns None. This is
    /// preferred over creating a new ChainRef and then using and_then_option since this function
    /// avoids allocating heap space for two ChainRefs.
    pub fn and_then_one_option<U, F: FnOnce(&'a T) -> Option<ChainRef<'a, U>>>(orig: Ref<'a, T>, f: F) -> Option<ChainRef<'a, U>> {
        // This is safe for the same reason outlined in ChainRef::from_ref
        let orig_val = unsafe { &*(orig.deref() as *const T) };

        if let Some(mut next) = f(orig_val) {
            next.refs.push(orig);
            Some(next)
        } else {
            None
        }
    }
}

impl <'a, T: 'static> Deref for ChainRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.val
    }
}

impl <'a, T: fmt::Debug + 'static> fmt::Debug for ChainRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.val)
    }
}

impl <'a, T: fmt::Display + 'static> fmt::Display for ChainRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

#[derive(Debug, Clone)]
pub struct DeferredDisplay<T: Fn (&mut fmt::Formatter) -> fmt::Result>(pub T);

impl <T: Fn (&mut fmt::Formatter) -> fmt::Result> fmt::Display for DeferredDisplay<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0(f)
    }
}
