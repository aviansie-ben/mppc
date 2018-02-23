use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::io;
use std::io::Write;
use std::mem;

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
