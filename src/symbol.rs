use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::rc::Rc;

use ast;
use lex::Span;
use util::{ChainRef, PrettyDisplay};

#[derive(Debug, Clone)]
pub struct FunSymbol {
    pub sig: usize,
    pub params: Vec<usize>,
    pub body: RefCell<ast::Block>
}

impl PrettyDisplay for FunSymbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Fun [sig {}]", indent, self.sig)?;

        for param in &self.params {
            write!(f, " {}", param)?;
        };

        write!(f, "\n{}", self.body.borrow().pretty_indented(&next_indent))?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MultiFunSymbol {
    pub funcs: RefCell<Vec<(usize, usize)>>
}

impl PrettyDisplay for MultiFunSymbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}MultiFun", indent)?;

        for &(_, id) in self.funcs.borrow().iter() {
            write!(f, " {}", id)?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VarSymbol {
    pub val_type: Type,
    pub dims: RefCell<Vec<ast::Expr>>
}

impl PrettyDisplay for VarSymbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Var {}", indent, self.val_type)?;

        for dim in self.dims.borrow().iter() {
            write!(f, "\n{}", dim.pretty_indented(&next_indent))?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ParamSymbol {
    pub val_type: Type
}

impl PrettyDisplay for ParamSymbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Param {}", next_indent, self.val_type)
    }
}

#[derive(Debug, Clone)]
pub enum SymbolType {
    Fun(FunSymbol),
    MultiFun(MultiFunSymbol),
    Var(VarSymbol),
    Param(ParamSymbol)
}

impl PrettyDisplay for SymbolType {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use symbol::SymbolType::*;
        match *self {
            Fun(ref sym) => write!(f, "{}", sym.pretty_indented(indent)),
            MultiFun(ref sym) => write!(f, "{}", sym.pretty_indented(indent)),
            Var(ref sym) => write!(f, "{}", sym.pretty_indented(indent)),
            Param(ref sym) => write!(f, "{}", sym.pretty_indented(indent))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub id: usize,
    pub name: String,
    pub span: Span,
    pub node: SymbolType,
    pub defining_fun: usize,
    pub has_nonlocal_references: Cell<bool>
}

impl Symbol {
    pub fn new(
        id: usize,
        name: String,
        span: Span,
        node: SymbolType,
        defining_fun: usize
    ) -> Symbol {
        Symbol {
            id: id,
            name: name,
            span: span,
            node: node,
            defining_fun: defining_fun,
            has_nonlocal_references: Cell::new(false)
        }
    }

    pub fn val_type(&self) -> Type {
        match self.node {
            SymbolType::Fun(ref sym) => Type::Defined(sym.sig),
            SymbolType::MultiFun(ref sym) => Type::union(
                sym.funcs.borrow().iter().map(|&(type_id, _)| Type::Defined(type_id))
            ),
            SymbolType::Var(ref sym) => sym.val_type.clone(),
            SymbolType::Param(ref sym) => sym.val_type.clone()
        }
    }

    pub fn is_assignable(&self) -> bool {
        match self.node {
            SymbolType::Fun(_) => false,
            SymbolType::MultiFun(_) => false,
            SymbolType::Var(ref sym) => sym.dims.borrow().len() == 0,
            SymbolType::Param(_) => true
        }
    }
}

impl PrettyDisplay for Symbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Symbol {} {}\n{}", indent, self.name, self.id, self.node.pretty_indented(&next_indent))
    }
}

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub symbol_names: HashMap<String, usize>,
    pub type_names: HashMap<String, (usize, Span)>,
    pub ctor_names: HashMap<String, Vec<(usize, usize)>>,
    pub parent: Option<Rc<RefCell<SymbolTable>>>
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            symbol_names: HashMap::new(),
            type_names: HashMap::new(),
            ctor_names: HashMap::new(),
            parent: None
        }
    }

    pub fn set_parent(&mut self, parent: Rc<RefCell<SymbolTable>>) -> () {
        self.parent = Some(parent);
    }

    pub fn add_symbol(&mut self, name: String, id: usize) -> () {
        self.symbol_names.insert(name, id);
    }

    pub fn add_type(&mut self, name: String, id: usize, span: Span) -> () {
        self.type_names.insert(name, (id, span));
    }

    pub fn add_ctor(&mut self, name: String, ctor: (usize, usize)) -> () {
        if let Some(ctors) = self.ctor_names.get_mut(&name) {
            ctors.push(ctor);
            return;
        };

        let mut ctors = if let Some(ref parent) = self.parent {
            // These let bindings are necessary in order to ensure that these values are dropped in
            // the correct order.
            let parent = parent.borrow();
            let ctors = parent.find_ctors(&name).map(|ctors| (&*ctors).clone()).unwrap_or_else(|| vec![]);

            ctors
        } else {
            vec![]
        };

        ctors.push(ctor);
        self.ctor_names.insert(name, ctors);
    }

    pub fn find_imm_named_symbol(&self, name: &str) -> Option<usize> {
        if let Some(sym_id) = self.symbol_names.get(name).map(|id| *id) {
            Some(sym_id)
        } else {
            None
        }
    }

    pub fn find_named_symbol(&self, name: &str) -> Option<usize> {
        if let Some(sym_id) = self.symbol_names.get(name).map(|id| *id) {
            Some(sym_id)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().find_named_symbol(name)
        } else {
            None
        }
    }

    pub fn find_imm_named_type(&self, name: &str) -> Option<(usize, Span)> {
        self.type_names.get(name).map(|id| *id)
    }

    pub fn find_named_type(&self, name: &str) -> Option<(usize, Span)> {
        if let Some(type_id) = self.type_names.get(name).map(|id| *id) {
            Some(type_id)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().find_named_type(name)
        } else {
            None
        }
    }

    pub fn find_ctors(&self, name: &str) -> Option<ChainRef<Vec<(usize, usize)>>> {
        if let Some(ctors) = self.ctor_names.get(name) {
            Some(ChainRef::new(ctors))
        } else if let Some(ref parent) = self.parent {
            ChainRef::and_then_one_option(
                parent.borrow(),
                |p| p.find_ctors(name)
            )
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct SymbolDefinitionTable {
    symbols: UnsafeCell<Vec<Box<Symbol>>>
}

impl SymbolDefinitionTable {
    pub fn new() -> SymbolDefinitionTable {
        SymbolDefinitionTable {
            symbols: UnsafeCell::new(Vec::new())
        }
    }

    pub fn add_symbol(&self, mut s: Symbol) -> usize {
        let symbols = unsafe { &mut *self.symbols.get() };
        let id = symbols.len();

        s.id = id;
        symbols.push(Box::new(s));
        id
    }

    pub fn add_named_symbol(&self, st: &mut SymbolTable, s: Symbol) -> usize {
        let name = s.name.clone();
        let id = self.add_symbol(s);

        st.symbol_names.insert(name, id);
        id
    }

    pub fn add_named_function_symbol<T: Iterator<Item=Symbol>>(
        &self,
        st: &mut SymbolTable,
        mut s: Symbol,
        args: T
    ) -> usize {
        let symbols = unsafe { &mut *self.symbols.get() };
        let mut id = symbols.len();
        let sig = if let SymbolType::Fun(ref s) = s.node {
            s.sig
        } else {
            unreachable!()
        };

        if let Some(old_sym) = st.find_imm_named_symbol(&s.name) {
            match unsafe { &*(symbols[old_sym].as_ref() as *const Symbol) }.node {
                SymbolType::MultiFun(ref mf) => {
                    mf.funcs.borrow_mut().push((sig, id));
                },
                SymbolType::Fun(ref f) => {
                    symbols.push(Box::new(Symbol::new(
                        id,
                        s.name.clone(),
                        Span::dummy(),
                        SymbolType::MultiFun(MultiFunSymbol {
                            funcs: RefCell::new(vec![(f.sig, old_sym), (sig, id + 1)])
                        }),
                        s.defining_fun
                    )));
                    st.symbol_names.insert(s.name.clone(), id);
                    id = id + 1
                },
                _ => unreachable!()
            };
        } else {
            st.symbol_names.insert(s.name.clone(), id);
        };

        s.id = id;
        symbols.push(Box::new(s));

        let args: Vec<_> = {
            let body = if let SymbolType::Fun(ref mut f) = symbols[id].node {
                f.body.borrow()
            } else {
                unreachable!();
            };
            let fst = &mut body.symbols.borrow_mut();
            args.map(|mut a| {
                a.defining_fun = id;
                self.add_named_symbol(fst, a)
            }).collect()
        };

        if let SymbolType::Fun(ref mut f) = symbols[id].node {
            f.params = args;
        };

        id
    }

    pub fn get_symbol(&self, id: usize) -> &Symbol {
        let symbols = unsafe { &*self.symbols.get() };

        unsafe { &*(symbols[id].as_ref() as *const Symbol) }
    }

    pub fn iter<'a>(&'a self) -> SymbolDefinitionTableIter<'a> {
        SymbolDefinitionTableIter(self, 0)
    }
}

pub struct SymbolDefinitionTableIter<'a>(&'a SymbolDefinitionTable, usize);

impl <'a> Iterator for SymbolDefinitionTableIter<'a> {
    type Item = (usize, &'a Symbol);

    fn next(&mut self) -> Option<Self::Item> {
        let SymbolDefinitionTableIter(sdt, ref mut id) = *self;
        let symbols = unsafe { &*sdt.symbols.get() };

        if *id >= symbols.len() {
            return None;
        } else {
            *id = *id + 1;
            return Some((*id, unsafe { &*(symbols[*id - 1].as_ref() as *const Symbol) }));
        }
    }
}

impl <'a> IntoIterator for &'a SymbolDefinitionTable {
    type Item = (usize, &'a Symbol);
    type IntoIter = SymbolDefinitionTableIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct DataTypeCtor {
    pub name: String,
    pub args: Vec<Type>,
    pub span: Span
}

impl PrettyDisplay for DataTypeCtor {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}Ctor #{}", indent, self.name)?;

        if self.args.len() != 0 {
            write!(f, "({}", self.args[0])?;

            for a in &self.args[1..] {
                write!(f, ", {}", a)?;
            };

            write!(f, ")")?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DataTypeDefinition {
    pub name: String,
    pub ctors: Vec<DataTypeCtor>,
    pub span: Span
}

impl PrettyDisplay for DataTypeDefinition {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}DataType {}", indent, self.name)?;

        for ctor in &self.ctors {
            write!(f, "\n{}", ctor.pretty_indented(&next_indent))?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FunctionTypeDefinition {
    pub params: Vec<Type>,
    pub return_type: Type
}

impl PrettyDisplay for FunctionTypeDefinition {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}FunctionType\n{} Params", indent, indent)?;

        for p in &self.params {
            write!(f, " {}", p)?;
        };

        write!(f, "\n{} Returns {}", indent, self.return_type)?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Data(DataTypeDefinition),
    Function(FunctionTypeDefinition)
}

impl PrettyDisplay for TypeDefinition {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use symbol::TypeDefinition::*;

        match *self {
            Data(ref td) => write!(f, "{}", td.pretty_indented(indent)),
            Function(ref td) => write!(f, "{}", td.pretty_indented(indent))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Defined(usize),
    Array(Box<Type>, u32),
    Never,
    Unresolved(Vec<Type>),
    Unknown,
    Error
}

#[derive(Debug, Clone, Copy)]
pub struct PrettyType<'a>(&'a Type, &'a TypeDefinitionTable);

impl Type {
    pub fn is_resolved(&self) -> bool {
        match *self {
            Type::Unresolved(_) => false,
            Type::Unknown => false,
            _ => true
        }
    }

    pub fn least_upper_bound(t1: &Type, t2: &Type) -> Option<Type> {
        if t1 == &Type::Error || t2 == &Type::Error {
            Some(Type::Error)
        } else if t1 == t2 {
            Some(t1.clone())
        } else if t1 == &Type::Never {
            Some(t2.clone())
        } else if t2 == &Type::Never {
            Some(t1.clone())
        } else if let (Type::Unresolved(ref t1s), Type::Unresolved(ref t2s)) = (t1, t2) {
            let mut ts: Vec<_> = t1s.iter().filter(|t| t2s.contains(t)).map(|t| t.clone()).collect();

            if ts.len() == 0 {
                None
            } else if ts.len() == 1 {
                Some(mem::replace(&mut ts[0], Type::Unknown))
            } else {
                Some(Type::Unresolved(ts))
            }
        } else if let Type::Unresolved(ref t1s) = t1 {
            if t1s.contains(t2) {
                Some(t2.clone())
            } else {
                None
            }
        } else if let Type::Unresolved(ref t2s) = t2 {
            if t2s.contains(t1) {
                Some(t1.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn can_convert_to(&self, t: &Type) -> bool {
        Type::least_upper_bound(self, t).is_some()
    }

    pub fn can_convert_to_exact(&self, t: &Type) -> bool {
        self == t || self == &Type::Error || t == &Type::Error || self == &Type::Never
    }

    pub fn pretty<'a>(&'a self, tdt: &'a TypeDefinitionTable) -> PrettyType<'a> {
        PrettyType(self, tdt)
    }

    pub fn union<T: IntoIterator<Item=Type>>(types: T) -> Type {
        fn do_union<T: IntoIterator<Item=Type>>(types: T, result: &mut Vec<Type>) {
            let types = types.into_iter();

            for t in types {
                match t {
                    Type::Unresolved(ts) => do_union(ts, result),
                    Type::Never => {},
                    t => if !result.contains(&t) { result.push(t); }
                };
            };
        }

        let mut result: Vec<Type> = Vec::new();

        do_union(types, &mut result);

        if result.len() == 0 || result.iter().any(|t| t == &Type::Error) {
            Type::Error
        } else {
            Type::Unresolved(result)
        }
    }

    pub fn are_cases_exhaustive<T>(&self, tdt: &TypeDefinitionTable, cases: &[ast::Case<T>]) -> bool {
        if let Type::Defined(type_id) = *self {
            if let TypeDefinition::Data(ref td) = tdt.get_definition(type_id) {
                let mut covered = vec![false; td.ctors.len()];

                for c in cases {
                    if c.ctor_id != !0 {
                        covered[c.ctor_id] = true;
                    };
                };

                covered.iter().all(|&b| b)
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use symbol::Type::*;
        match *self {
            Int => write!(f, "int")?,
            Real => write!(f, "real")?,
            Bool => write!(f, "bool")?,
            Char => write!(f, "char")?,
            Defined(id) => write!(f, "(typeref {})", id)?,
            Array(ref inner_type, dims) => {
                write!(f, "{}", inner_type)?;

                for _ in 0..dims {
                    write!(f, "[]")?;
                };
            },
            Never => write!(f, "(never type)")?,
            Unresolved(ref possible_types) => {
                if possible_types.len() == 1 {
                    write!(f, "{} (ambiguous)", possible_types[0])?;
                } else {
                    write!(f, "one of {}", possible_types[0])?;

                    if possible_types.len() >= 3 {
                        for i in 1..(possible_types.len() - 1) {
                            write!(f, ", {}", possible_types[i])?;
                        };

                        write!(f, ", or {}", possible_types[possible_types.len() - 1])?;
                    } else {
                        write!(f, " or {}", possible_types[1])?;
                    }
                };
            },
            Unknown => write!(f, "(unknown type)")?,
            Error => write!(f, "(error type)")?
        };
        Result::Ok(())
    }
}

impl <'a> fmt::Display for PrettyType<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use symbol::Type::*;
        use symbol::TypeDefinition::*;

        let PrettyType(t, tdt) = *self;

        match *t {
            Int => write!(f, "int")?,
            Real => write!(f, "real")?,
            Bool => write!(f, "bool")?,
            Char => write!(f, "char")?,
            Defined(id) => {
                match tdt.get_definition(id) {
                    Data(ref td) => write!(f, "{}", td.name)?,
                    Function(ref td) => {
                        write!(f, "fun (")?;

                        if td.params.len() != 0 {
                            write!(f, "{}", td.params[0].pretty(tdt))?;

                            for p in &td.params[1..] {
                                write!(f, ", {}", p.pretty(tdt))?;
                            }
                        }

                        write!(f, ") : {}", td.return_type.pretty(tdt))?;
                    }
                }
            },
            Array(ref inner_type, dims) => {
                write!(f, "{}", inner_type.pretty(tdt))?;

                for _ in 0..dims {
                    write!(f, "[]")?;
                };
            },
            Never => write!(f, "(never type)")?,
            Unresolved(ref possible_types) => {
                if possible_types.len() == 1 {
                    write!(f, "{} (ambiguous)", possible_types[0].pretty(tdt))?;
                } else {
                    write!(f, "one of {}", possible_types[0].pretty(tdt))?;

                    if possible_types.len() >= 3 {
                        for i in 1..(possible_types.len() - 1) {
                            write!(f, ", {}", possible_types[i].pretty(tdt))?;
                        };

                        write!(f, ", or {}", possible_types[possible_types.len() - 1].pretty(tdt))?;
                    } else {
                        write!(f, " or {}", possible_types[1].pretty(tdt))?;
                    }
                }
            },
            Unknown => write!(f, "(unknown type)")?,
            Error => write!(f, "(error type)")?
        };
        Result::Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Unknown,
    BoolNot,
    IntFloat,
    IntNeg,
    RealFloor,
    RealCeil,
    RealNeg
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Unknown,
    BoolOr,
    BoolAnd,
    BoolEq,
    IntEq,
    IntLt,
    IntGt,
    IntLe,
    IntGe,
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    RealEq,
    RealLt,
    RealGt,
    RealLe,
    RealGe,
    RealAdd,
    RealSub,
    RealMul,
    RealDiv,
    CharEq
}

lazy_static! {
    pub static ref UNARY_OPS: HashMap<ast::UnaryOp, Vec<([Type; 1], Type, UnaryOp)>> = {
        let mut m = HashMap::new();

        m.insert(ast::UnaryOp::Not, vec![
            ([Type::Bool], Type::Bool, UnaryOp::BoolNot)
        ]);
        m.insert(ast::UnaryOp::Float, vec![
            ([Type::Int], Type::Real, UnaryOp::IntFloat)
        ]);
        m.insert(ast::UnaryOp::Floor, vec![
            ([Type::Real], Type::Real, UnaryOp::RealFloor)
        ]);
        m.insert(ast::UnaryOp::Ceil, vec![
            ([Type::Real], Type::Real, UnaryOp::RealCeil)
        ]);
        m.insert(ast::UnaryOp::Neg, vec![
            ([Type::Int], Type::Int, UnaryOp::IntNeg),
            ([Type::Real], Type::Real, UnaryOp::RealNeg)
        ]);

        m
    };

    pub static ref BINARY_OPS: HashMap<ast::BinaryOp, Vec<([Type; 2], Type, BinaryOp)>> = {
        let mut m = HashMap::new();

        m.insert(ast::BinaryOp::Or, vec![
            ([Type::Bool, Type::Bool], Type::Bool, BinaryOp::BoolOr)
        ]);
        m.insert(ast::BinaryOp::And, vec![
            ([Type::Bool, Type::Bool], Type::Bool, BinaryOp::BoolAnd)
        ]);
        m.insert(ast::BinaryOp::Equal, vec![
            ([Type::Bool, Type::Bool], Type::Bool, BinaryOp::BoolEq),
            ([Type::Int, Type::Int], Type::Bool, BinaryOp::IntEq),
            ([Type::Real, Type::Real], Type::Bool, BinaryOp::RealEq),
            ([Type::Char, Type::Char], Type::Bool, BinaryOp::CharEq)
        ]);
        m.insert(ast::BinaryOp::Lt, vec![
            ([Type::Int, Type::Int], Type::Bool, BinaryOp::IntLt),
            ([Type::Real, Type::Real], Type::Bool, BinaryOp::RealLt)
        ]);
        m.insert(ast::BinaryOp::Gt, vec![
            ([Type::Int, Type::Int], Type::Bool, BinaryOp::IntGt),
            ([Type::Real, Type::Real], Type::Bool, BinaryOp::RealGt)
        ]);
        m.insert(ast::BinaryOp::Le, vec![
            ([Type::Int, Type::Int], Type::Bool, BinaryOp::IntLe),
            ([Type::Real, Type::Real], Type::Bool, BinaryOp::RealLe)
        ]);
        m.insert(ast::BinaryOp::Ge, vec![
            ([Type::Int, Type::Int], Type::Bool, BinaryOp::IntGe),
            ([Type::Real, Type::Real], Type::Bool, BinaryOp::RealGe)
        ]);
        m.insert(ast::BinaryOp::Add, vec![
            ([Type::Int, Type::Int], Type::Int, BinaryOp::IntAdd),
            ([Type::Real, Type::Real], Type::Real, BinaryOp::RealAdd)
        ]);
        m.insert(ast::BinaryOp::Sub, vec![
            ([Type::Int, Type::Int], Type::Int, BinaryOp::IntSub),
            ([Type::Real, Type::Real], Type::Real, BinaryOp::RealSub)
        ]);
        m.insert(ast::BinaryOp::Mul, vec![
            ([Type::Int, Type::Int], Type::Int, BinaryOp::IntMul),
            ([Type::Real, Type::Real], Type::Real, BinaryOp::RealMul)
        ]);
        m.insert(ast::BinaryOp::Div, vec![
            ([Type::Int, Type::Int], Type::Int, BinaryOp::IntDiv),
            ([Type::Real, Type::Real], Type::Real, BinaryOp::RealDiv)
        ]);

        m
    };
}

#[derive(Debug)]
pub struct TypeDefinitionTable {
    defs: UnsafeCell<Vec<Option<Box<TypeDefinition>>>>
}

pub struct TypeDefinitionTableIter<'a>(&'a TypeDefinitionTable, usize);

impl <'a> Iterator for TypeDefinitionTableIter<'a> {
    type Item = (usize, &'a TypeDefinition);

    fn next(&mut self) -> Option<Self::Item> {
        let TypeDefinitionTableIter(tdt, ref mut id) = *self;
        let defs = unsafe { &*tdt.defs.get() };

        loop {
            if *id >= defs.len() {
                return None;
            } else if let Some(ref def) = defs[*id] {
                *id = *id + 1;
                return Some((*id - 1, unsafe { &*(def.as_ref() as *const TypeDefinition) }));
            } else {
                *id = *id + 1;
            };
        };
    }
}

impl TypeDefinitionTable {
    pub fn new() -> TypeDefinitionTable {
        TypeDefinitionTable {
            defs: UnsafeCell::new(Vec::new())
        }
    }

    pub fn add_dummy_definition(&self) -> usize {
        let defs = unsafe { &mut *self.defs.get() };

        defs.push(None);
        defs.len() - 1
    }

    pub fn add_definition(&self, def: TypeDefinition) -> usize {
        let defs = unsafe { &mut *self.defs.get() };

        defs.push(Some(Box::new(def)));
        defs.len() - 1
    }

    pub fn define_type(&self, id: usize, def: TypeDefinition) -> () {
        let defs = unsafe { &mut *self.defs.get() };

        if defs[id].is_some() {
            panic!("cannot redefine type more than once");
        };

        defs[id] = Some(Box::new(def));
    }

    pub fn get_definition(&self, id: usize) -> &TypeDefinition {
        let defs = unsafe { &*self.defs.get() };

        unsafe { &*(defs[id].as_ref().unwrap().as_ref() as *const TypeDefinition) }
    }

    pub fn get_function_type(&self, params: &[Type], return_type: &Type) -> usize {
        for (id, type_def) in unsafe { &*self.defs.get() }.iter().enumerate() {
            if let Some(box TypeDefinition::Function(fd)) = type_def {
                if &fd.params[..] == params && &fd.return_type == return_type {
                    return id;
                }
            }
        };

        self.add_definition(TypeDefinition::Function(FunctionTypeDefinition {
            params: params.to_vec(),
            return_type: return_type.clone()
        }))
    }

    pub fn iter<'a>(&'a self) -> TypeDefinitionTableIter<'a> {
        TypeDefinitionTableIter(self, 0)
    }
}

impl <'a> IntoIterator for &'a TypeDefinitionTable {
    type Item = (usize, &'a TypeDefinition);
    type IntoIter = TypeDefinitionTableIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
