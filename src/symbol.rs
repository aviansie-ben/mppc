use std::cell::{Cell, RefCell};
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
    Var(VarSymbol),
    Param(ParamSymbol)
}

impl PrettyDisplay for SymbolType {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use symbol::SymbolType::*;
        match *self {
            Fun(ref sym) => write!(f, "{}", sym.pretty_indented(indent)),
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
    pub node: SymbolType
}

impl Symbol {
    fn val_type(&self) -> Type {
        match self.node {
            SymbolType::Fun(ref sym) => Type::Defined(sym.sig),
            SymbolType::Var(ref sym) => sym.val_type.clone(),
            SymbolType::Param(ref sym) => sym.val_type.clone()
        }
    }

    fn is_assignable(&self) -> bool {
        match self.node {
            SymbolType::Fun(_) => false,
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
    pub symbols: HashMap<usize, Symbol>,
    pub parent: Option<Rc<RefCell<SymbolTable>>>,
    next_id: Rc<Cell<usize>>
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            symbol_names: HashMap::new(),
            type_names: HashMap::new(),
            ctor_names: HashMap::new(),
            symbols: HashMap::new(),
            parent: None,
            next_id: Rc::new(Cell::new(0))
        }
    }

    pub fn set_parent(&mut self, parent: Rc<RefCell<SymbolTable>>) -> () {
        assert!(self.next_id.get() == 0);

        self.next_id = parent.borrow().next_id.clone();
        self.parent = Some(parent);
    }

    fn alloc_symbol_id(&mut self) -> usize {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    pub fn add_symbol(&mut self, mut s: Symbol) -> usize {
        s.id = self.alloc_symbol_id();
        self.add_symbol_with_id(s)
    }

    pub fn add_symbol_with_id(&mut self, s: Symbol) -> usize {
        let id = s.id;

        self.symbol_names.insert(s.name.clone(), id);
        self.symbols.insert(id, s);

        id
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

    pub fn find_symbol(&self, id: usize) -> Option<ChainRef<Symbol>> {
        if let Some(sym) = self.symbols.get(&id) {
            Some(ChainRef::new(sym))
        } else if let Some(ref parent) = self.parent {
            ChainRef::and_then_one_option(
                parent.borrow(),
                |p| p.find_symbol(id)
            )
        } else {
            None
        }
    }

    pub fn find_imm_named_symbol(&self, name: &str) -> Option<ChainRef<Symbol>> {
        if let Some(sym_id) = self.symbol_names.get(name).map(|id| *id) {
            self.find_symbol(sym_id)
        } else {
            None
        }
    }

    pub fn find_named_symbol(&self, name: &str) -> Option<ChainRef<Symbol>> {
        if let Some(sym_id) = self.symbol_names.get(name).map(|id| *id) {
            self.find_symbol(sym_id)
        } else if let Some(ref parent) = self.parent {
            ChainRef::and_then_one_option(
                parent.borrow(),
                |p| p.find_named_symbol(name)
            )
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

    pub fn lift_decls(&mut self, target: &mut SymbolTable) {
        for (id, sym) in mem::replace(&mut self.symbols, HashMap::new()) {
            target.symbols.insert(id, sym);
        };
    }
}

#[derive(Debug, Clone)]
pub struct DataTypeCtor {
    pub name: String,
    pub args: Vec<Type>,
    pub span: Span
}

#[derive(Debug, Clone)]
pub struct DataTypeDefinition {
    pub name: String,
    pub ctors: Vec<DataTypeCtor>,
    pub span: Span
}

#[derive(Debug, Clone)]
pub struct FunctionTypeDefinition {
    pub params: Vec<Type>,
    pub return_type: Type
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Data(DataTypeDefinition),
    Function(FunctionTypeDefinition),
    Dummy
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Defined(usize),
    Array(Box<Type>, u32),
    Unresolved(Vec<Type>),
    Unknown,
    Error
}

#[derive(Debug, Clone, Copy)]
pub struct PrettyType<'a>(&'a Type, &'a TypeDefinitionTable);

impl Type {
    fn is_resolved(&self) -> bool {
        match *self {
            Type::Unresolved(_) => false,
            Type::Unknown => false,
            _ => true
        }
    }

    fn least_upper_bound(t1: &Type, t2: &Type) -> Option<Type> {
        if t1 == &Type::Error || t2 == &Type::Error {
            Some(Type::Error)
        } else if t1 == t2 {
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

    fn can_convert_to(&self, t: &Type) -> bool {
        Type::least_upper_bound(self, t).is_some()
    }

    fn can_convert_to_exact(&self, t: &Type) -> bool {
        self == t || self == &Type::Error || t == &Type::Error
    }

    fn pretty<'a>(&'a self, tdt: &'a TypeDefinitionTable) -> PrettyType<'a> {
        PrettyType(self, tdt)
    }

    fn union<T: IntoIterator<Item=Type>>(types: T) -> Type {
        fn do_union<T: IntoIterator<Item=Type>>(types: T, result: &mut Vec<Type>) {
            let types = types.into_iter();

            for t in types {
                match t {
                    Type::Unresolved(ts) => do_union(ts, result),
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
                match tdt.defs[id] {
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
                    },
                    Dummy => write!(f, "(dummy type)")?
                }
            },
            Array(ref inner_type, dims) => {
                write!(f, "{}", inner_type.pretty(tdt))?;

                for _ in 0..dims {
                    write!(f, "[]")?;
                };
            },
            Unresolved(ref possible_types) => {
                if possible_types.len() == 1 {
                    write!(f, "{}", possible_types[0].pretty(tdt))?;
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
    static ref UNARY_OPS: HashMap<ast::UnaryOp, Vec<([Type; 1], Type, UnaryOp)>> = {
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

    static ref BINARY_OPS: HashMap<ast::BinaryOp, Vec<([Type; 2], Type, BinaryOp)>> = {
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

#[derive(Debug, Clone)]
pub struct TypeDefinitionTable {
    pub defs: Vec<TypeDefinition>
}

impl TypeDefinitionTable {
    pub fn new() -> TypeDefinitionTable {
        TypeDefinitionTable {
            defs: Vec::new()
        }
    }

    pub fn add_definition(&mut self, def: TypeDefinition) -> usize {
        self.defs.push(def);
        self.defs.len() - 1
    }

    pub fn redefine(&mut self, id: usize, def: TypeDefinition) -> () {
        self.defs[id] = def;
    }

    pub fn get_function_type(&mut self, params: &[Type], return_type: &Type) -> usize {
        for (id, type_def) in self.defs.iter().enumerate() {
            if let TypeDefinition::Function(fd) = type_def {
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
}

fn resolve_type(
    tdt: &mut TypeDefinitionTable,
    symbols: &SymbolTable,
    errors: &mut Vec<(String, Span)>,
    t: &ast::Type,
) -> Type {
    match *t {
        ast::Type::Int => Type::Int,
        ast::Type::Real => Type::Real,
        ast::Type::Bool => Type::Bool,
        ast::Type::Char => Type::Char,
        ast::Type::Id(ref name, ref span) => {
            if let Some((type_id, _)) = symbols.find_named_type(name) {
                Type::Defined(type_id)
            } else {
                errors.push((
                    format!("no type {} has been defined", name),
                    *span
                ));
                Type::Error
            }
        },
        ast::Type::Array(ref inner_type, ref dims) => Type::Array(
            Box::new(resolve_type(tdt, symbols, errors, inner_type)),
            *dims
        )
    }
}

fn create_symbol_for_decl(
    tdt: &mut TypeDefinitionTable,
    symbols: &mut SymbolTable,
    errors: &mut Vec<(String, Span)>,
    decl: ast::Decl
) {
    match decl.node {
        ast::DeclType::Data(name, type_id, ctors) => {
            let mut type_def = DataTypeDefinition {
                name: name,
                ctors: ctors.into_iter().map(|ctor| {
                    DataTypeCtor {
                        name: ctor.cid.to_string(),
                        args: ctor.types.iter()
                            .map(|t| resolve_type(tdt, symbols, errors, t))
                            .collect(),
                        span: ctor.span
                    }
                }).collect(),
                span: decl.span
            };

            // Check that constructor signatures are unique within a single type. Yes, this is
            // O(n^2), but it should be fine since the number of constructors for a type is expected
            // to be relatively small.
            let mut bad_ctors = vec![];
            for (i, c1) in type_def.ctors[1..].iter().enumerate() {
                for c2 in &type_def.ctors[..(i + 1)] {
                    if c1.name == c2.name {
                        errors.push((
                            format!(
                                "constructor #{} redefined (original definition is at line {}, col {})",
                                c1.name,
                                c2.span.lo.line,
                                c2.span.lo.col
                            ),
                            c1.span
                        ));
                        bad_ctors.push(i + 1);
                        break;
                    };
                };
            };

            for i in bad_ctors.into_iter().rev() {
                type_def.ctors.remove(i);
            };

            for (i, ctor) in type_def.ctors.iter().enumerate() {
                symbols.add_ctor(ctor.name.clone(), (type_id, i));
            };

            tdt.redefine(type_id, TypeDefinition::Data(type_def));
        },
        ast::DeclType::Fun(name, sig, body) => {
            let params: Vec<_> = sig.params.iter()
                .map(|p| resolve_type(tdt, symbols, errors, &p.val_type))
                .collect();
            let return_type = resolve_type(tdt, symbols, errors, &sig.return_type);
            let fn_type = tdt.get_function_type(&params, &return_type);

            // TODO Function overloading
            let define = if let Some(old_sym) = symbols.find_imm_named_symbol(&name) {
                errors.push((
                    format!(
                        "the name '{}' has already been defined in this scope (original definition is at line {}, col {})",
                        name,
                        old_sym.span.lo.line,
                        old_sym.span.lo.col
                    ),
                    decl.span
                ));
                false
            } else {
                true
            };

            let params: Vec<_> = {
                let mut fn_symbols = body.symbols.borrow_mut();
                sig.params.into_iter().map(|p| fn_symbols.add_symbol_with_id(Symbol {
                    id: symbols.alloc_symbol_id(),
                    name: p.id,
                    span: p.span,
                    node: SymbolType::Param(ParamSymbol {
                        val_type: resolve_type(tdt, symbols, errors, &p.val_type)
                    })
                })).collect()
            };

            if define {
                symbols.add_symbol(Symbol {
                    id: 0,
                    name: name.to_string(),
                    span: decl.span,
                    node: SymbolType::Fun(FunSymbol {
                        sig: fn_type,
                        params: params,
                        body: RefCell::new(body)
                    })
                });
            };
        },
        ast::DeclType::Var(spec, val_type) => {
            let val_type = resolve_type(tdt, symbols, errors, &val_type);

            let define = if let Some(old_sym) = symbols.find_imm_named_symbol(&spec.id) {
                errors.push((
                    format!(
                        "the name '{}' has already been defined in this scope (original definition is at line {}, col {})",
                        spec.id,
                        old_sym.span.lo.line,
                        old_sym.span.lo.col
                    ),
                    spec.span
                ));
                false
            } else {
                true
            };

            if define {
                symbols.add_symbol(Symbol {
                    id: 0,
                    name: spec.id,
                    span: spec.span,
                    node: SymbolType::Var(VarSymbol {
                        val_type: val_type,
                        dims: RefCell::new(spec.dims)
                    })
                });
            };
        }
    };
}

fn get_valid_call_signatures<'a, T: IntoIterator<Item=&'a [Type]>>(
    actual_types: &[Type],
    expected_types: T
) -> Vec<usize> {
    let mut result = Vec::new();

    for (i, expected_types) in expected_types.into_iter().enumerate() {
        if expected_types.iter().zip(actual_types.iter()).all(|(et, at)| at.can_convert_to(et)) {
            result.push(i);
        };
    };

    result
}

fn analyze_call_signature(
    span: &Span,
    expected_types: &[Type],
    params: &mut [ast::Expr],
    tdt: &TypeDefinitionTable,
    symbols: &SymbolTable,
    errors: &mut Vec<(String, Span)>
) {
    if params.len() != expected_types.len() {
        errors.push((
            format!(
                "wrong number of arguments: expected {}, but found {}",
                expected_types.len(),
                params.len()
            ),
            *span
        ));
    };

    for (et, param) in expected_types.iter().zip(params.iter_mut()) {
        let at = &analyze_expression(param, tdt, symbols, Some(et), errors);

        if !at.can_convert_to_exact(et) {
            errors.push((
                format!("cannot convert from {} to {}", at.pretty(tdt), et.pretty(tdt)),
                param.span
            ));
        };
    };
}

fn do_analyze_expression(
    expr: &mut ast::Expr,
    tdt: &TypeDefinitionTable,
    symbols: &SymbolTable,
    expected_type: Option<&Type>,
    errors: &mut Vec<(String, Span)>
) -> Type {
    match expr.node {
        ast::ExprType::BinaryOp(op, ref mut lhs, ref mut rhs, ref mut sym_op) => {
            let val_types = [
                analyze_expression(lhs, tdt, symbols, None, errors),
                analyze_expression(rhs, tdt, symbols, None, errors)
            ];

            if val_types[0] == Type::Error || val_types[1] == Type::Error {
                return Type::Error;
            };

            let op_impl = if let Some(impls) = BINARY_OPS.get(&op) {
                let valid_impls = get_valid_call_signatures(
                    &val_types,
                    impls.iter().map(|&(ref params, _, _)| { &params[..] })
                );

                if valid_impls.len() == 0 {
                    None
                } else if valid_impls.len() == 1 {
                    Some(&impls[valid_impls[0]])
                } else {
                    return Type::union(valid_impls.iter().map(|&i| impls[i].1.clone()));
                }
            } else {
                None
            };

            if let Some(&(ref params, ref result, op)) = op_impl {
                analyze_expression(lhs, tdt, symbols, Some(&params[0]), errors);
                analyze_expression(rhs, tdt, symbols, Some(&params[1]), errors);
                *sym_op = op;

                return result.clone();
            } else {
                errors.push((
                    format!("no operator {} exists for {} and {}", op, val_types[0].pretty(tdt), val_types[0].pretty(tdt)),
                    expr.span
                ));
                return Type::Error;
            };
        },
        ast::ExprType::UnaryOp(op, ref mut val, ref mut sym_op) => {
            let val_types = [analyze_expression(val, tdt, symbols, None, errors)];

            if val_types[0] == Type::Error {
                return Type::Error;
            };

            let op_impl = if let Some(impls) = UNARY_OPS.get(&op) {
                let valid_impls = get_valid_call_signatures(
                    &val_types,
                    impls.iter().map(|&(ref params, _, _)| { &params[..] })
                );

                if valid_impls.len() == 0 {
                    None
                } else if valid_impls.len() == 1 {
                    Some(&impls[valid_impls[0]])
                } else {
                    return Type::union(valid_impls.iter().map(|&i| impls[i].1.clone()));
                }
            } else {
                None
            };

            if let Some(&(ref params, ref result, op)) = op_impl {
                analyze_expression(val, tdt, symbols, Some(&params[0]), errors);
                *sym_op = op;

                return result.clone();
            } else {
                errors.push((
                    format!("no operator {} exists for {}", op, val_types[0].pretty(tdt)),
                    expr.span
                ));
                return Type::Error;
            };
        },
        ast::ExprType::Size(ref mut val, dim) => {
            let val_type = analyze_expression(val, tdt, symbols, None, errors);

            if let Type::Array(_, val_dims) = val_type {
                if dim >= val_dims {
                    errors.push((
                        format!("cannot get the size of dimension {} of a value of type {}", dim, val_type.pretty(tdt)),
                        expr.span
                    ));
                };
            } else {
                errors.push((
                    format!("cannot use size operator on value of type {}", val_type.pretty(tdt)),
                    expr.span
                ));
            };
            return Type::Int;
        },
        ast::ExprType::Id(ref name, ref mut sym_id) => {
            return if let Some(sym) = symbols.find_named_symbol(name) {
                expr.assignable = sym.is_assignable();
                *sym_id = sym.id;
                sym.val_type()
            } else {
                expr.assignable = true;
                errors.push((
                    format!("no variable '{}' exists in this scope", name),
                    expr.span
                ));
                Type::Error
            };
        },
        ast::ExprType::Call(ref mut func, ref mut params) => {
            fn get_function_typedefs<'a>(
                t: &Type,
                tdt: &'a TypeDefinitionTable
            ) -> Vec<(usize, &'a FunctionTypeDefinition)> {
                if let Type::Defined(ref type_id) = *t {
                    if let TypeDefinition::Function(ref typedef) = tdt.defs[*type_id] {
                        vec![(*type_id, typedef)]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            }

            let func_type = analyze_expression(func, tdt, symbols, None, errors);

            if expr.val_type == Type::Unknown {
                for param in params.iter_mut() {
                    analyze_expression(param, tdt, symbols, None, errors);
                };
            };

            let typedefs = get_function_typedefs(&func_type, tdt);

            if typedefs.len() == 0 {
                errors.push((
                    format!("cannot call expression of type {}", func_type.pretty(tdt)),
                    expr.span
                ));
                return Type::Error;
            } else if typedefs.len() == 1 {
                analyze_call_signature(&expr.span, &typedefs[0].1.params, params, tdt, symbols, errors);
                return typedefs[0].1.return_type.clone();
            } else {
                // TODO Allow function overloading
                return Type::Error;
            };
        },
        ast::ExprType::Index(ref mut val, ref mut index) => {
            let val_type = analyze_expression(val, tdt, symbols, None, errors);
            let index_type = analyze_expression(index, tdt, symbols, Some(&Type::Int), errors);

            if !index_type.can_convert_to_exact(&Type::Int) {
                errors.push((
                    format!("cannot convert from {} to int", index_type.pretty(tdt)),
                    index.span
                ));
            };

            if let Type::Array(inner_type, dims) = val_type {
                return if dims == 1 {
                    expr.assignable = true;
                    *inner_type
                } else {
                    Type::Array(inner_type, dims - 1)
                };
            } else {
                expr.assignable = true;
                errors.push((
                    format!("cannot index into value of type {}", val_type.pretty(tdt)),
                    expr.span
                ));
                return Type::Error;
            };
        },
        ast::ExprType::Cons(ref name, ref mut params, ref mut expr_ctor_id) => {
            if expr.val_type == Type::Unknown {
                for param in params.iter_mut() {
                    analyze_expression(param, tdt, symbols, None, errors);
                };
            };

            let ctors = if let Some(ctors) = symbols.find_ctors(name) {
                ctors
            } else {
                errors.push((
                    format!("no constructor #{} has been declared in this scope", name),
                    expr.span
                ));
                return Type::Error;
            };

            if ctors.len() == 1 {
                let ctor_id = ctors[0];
                let ctor = match tdt.defs[ctor_id.0] {
                    TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                    _ => panic!("invalid data type")
                };

                analyze_call_signature(&expr.span, &ctor.args, params, tdt, symbols, errors);
                *expr_ctor_id = ctor_id.1;
                return Type::Defined(ctor_id.0);
            } else {
                if let Some(Type::Defined(expected_type)) = expected_type {
                    let ctor_id = ctors.iter().find(|ctor_id| &ctor_id.0 == expected_type);

                    if let Some(ctor_id) = ctor_id {
                        let ctor = match tdt.defs[ctor_id.0] {
                            TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                            _ => panic!("invalid data type")
                        };

                        analyze_call_signature(&expr.span, &ctor.args, params, tdt, symbols, errors);
                        *expr_ctor_id = ctor_id.1;
                        return Type::Defined(ctor_id.0);
                    };
                };

                let ctors: Vec<_> = ctors.iter().map(|ctor_id| {
                    let ctor = match tdt.defs[ctor_id.0] {
                        TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                        _ => panic!("invalid data type")
                    };

                    (ctor, *ctor_id)
                }).filter(|(ctor, _)| ctor.args.len() == params.len()).collect();

                if ctors.len() == 0 {
                    errors.push((
                        format!("no constructor #{} in this scope takes {} arguments", name, params.len()),
                        expr.span
                    ));
                    return Type::Error;
                } else if ctors.len() == 1 {
                    let (ctor, ctor_id) = ctors[0];

                    analyze_call_signature(&expr.span, &ctor.args, params, tdt, symbols, errors);
                    *expr_ctor_id = ctor_id.1;
                    return Type::Defined(ctor_id.0);
                }

                let param_types: Vec<_> = params.iter().map(|p| p.val_type.clone()).collect();
                let valid_ctors: Vec<_> = ctors.iter().filter(|&&(ctor, _)| {
                    for (et, at) in ctor.args.iter().zip(param_types.iter()) {
                        if !at.can_convert_to(et) {
                            return false;
                        };
                    };
                    true
                }).collect();

                if valid_ctors.len() == 0 {
                    // TODO Better error message
                    errors.push((
                        format!("no constructor #{} in this scope matches the given arguments", name),
                        expr.span
                    ));
                    return Type::Error;
                } else if valid_ctors.len() == 1 {
                    let (ctor, ctor_id) = valid_ctors[0];
                    analyze_call_signature(&expr.span, &ctor.args, params, tdt, symbols, errors);
                    *expr_ctor_id = ctor_id.1;
                    return Type::Defined(ctor_id.0);
                } else {
                    return Type::union(valid_ctors.into_iter().map(|&(_, (type_id, _))| {
                        Type::Defined(type_id)
                    }));
                };
            };
        },
        ast::ExprType::Int(_) => return Type::Int,
        ast::ExprType::Real(_) => return Type::Real,
        ast::ExprType::Bool(_) => return Type::Bool,
        ast::ExprType::Char(_) => return Type::Char
    };
}

fn analyze_expression(
    expr: &mut ast::Expr,
    tdt: &TypeDefinitionTable,
    symbols: &SymbolTable,
    expected_type: Option<&Type>,
    errors: &mut Vec<(String, Span)>
) -> Type {
    if expr.val_type.is_resolved() {
        return expr.val_type.clone();
    };

    expr.val_type = do_analyze_expression(expr, tdt, symbols, expected_type, errors);
    expr.val_type.clone()
}

fn get_statement_symbols(
    stmt: &mut ast::Stmt
) -> &Rc<RefCell<SymbolTable>> {
    if let ast::StmtType::Block(ref block) = stmt.node {
        return &block.symbols;
    };

    let old_stmt = mem::replace(stmt, ast::Stmt::block(ast::Block::new(Vec::new(), Vec::new())));
    if let ast::StmtType::Block(ref mut block) = stmt.node {
        block.stmts.push(old_stmt);
        &block.symbols
    } else {
        unreachable!();
    }
}

fn analyze_statement(
    stmt: &mut ast::Stmt,
    tdt: &mut TypeDefinitionTable,
    symbols: &Rc<RefCell<SymbolTable>>,
    expected_return: Option<&Type>,
    errors: &mut Vec<(String, Span)>
) {
    match stmt.node {
        ast::StmtType::IfThenElse(ref mut cond, ref mut then_stmt, ref mut else_stmt) => {
            let cond_type = analyze_expression(cond, tdt, &symbols.borrow(), Some(&Type::Bool), errors);
            if !cond_type.can_convert_to_exact(&Type::Bool) {
                errors.push((
                    format!("cannot convert from {} to bool", cond_type.pretty(tdt)),
                    cond.span
                ));
            };

            analyze_statement(then_stmt, tdt, symbols, expected_return, errors);
            analyze_statement(else_stmt, tdt, symbols, expected_return, errors);
        },
        ast::StmtType::WhileDo(ref mut cond, ref mut do_stmt) => {
            let cond_type = analyze_expression(cond, tdt, &symbols.borrow(), Some(&Type::Bool), errors);
            if !cond_type.can_convert_to_exact(&Type::Bool) {
                errors.push((
                    format!("cannot convert from {} to bool", cond_type.pretty(tdt)),
                    cond.span
                ));
            };

            analyze_statement(do_stmt, tdt, symbols, expected_return, errors);
        },
        ast::StmtType::Read(ref mut loc) => {
            let val_type = analyze_expression(loc, tdt, &symbols.borrow(), None, errors);
            let is_valid = match val_type {
                Type::Bool => true,
                Type::Char => true,
                Type::Int => true,
                Type::Real => true,
                Type::Error => true,
                _ => false
            };

            if !loc.assignable {
                errors.push((
                    "cannot assign a value to this expression".to_string(),
                    loc.span
                ));
            } else if !is_valid {
                errors.push((
                    format!("cannot read a value of type {}", val_type.pretty(tdt)),
                    stmt.span
                ));
            };
        },
        ast::StmtType::Assign(ref mut loc, ref mut val) => {
            let symbols = symbols.borrow();
            let expected_type = analyze_expression(loc, tdt, &symbols, None, errors);
            let actual_type = analyze_expression(val, tdt, &symbols, if expected_type.is_resolved() {
                Some(&expected_type)
            } else {
                None
            }, errors);

            if !loc.assignable {
                errors.push((
                    "cannot assign a value to this expression".to_string(),
                    loc.span
                ));
            } else if !actual_type.can_convert_to_exact(&expected_type) {
                errors.push((
                    format!(
                        "cannot convert from {} to {}",
                        actual_type.pretty(tdt),
                        expected_type.pretty(tdt)
                    ),
                    val.span
                ));
            };
        },
        ast::StmtType::Print(ref mut val) => {
            let val_type = analyze_expression(val, tdt, &symbols.borrow(), None, errors);
            let is_valid = match val_type {
                Type::Bool => true,
                Type::Char => true,
                Type::Int => true,
                Type::Real => true,
                Type::Error => true,
                _ => false
            };

            if !is_valid {
                errors.push((
                    format!("cannot print a value of type {}", val_type.pretty(tdt)),
                    stmt.span
                ));
            };
        },
        ast::StmtType::Block(ref mut inner_block) => {
            if inner_block.symbols.borrow().parent.is_none() {
                inner_block.symbols.borrow_mut().set_parent(symbols.clone());
            };

            populate_block_symbol_table(tdt, inner_block, expected_return, errors);

            inner_block.symbols.borrow_mut().lift_decls(&mut symbols.borrow_mut());
        },
        ast::StmtType::Case(ref mut val, ref mut cases) => {
            fn analyze_case_error(
                case: &mut ast::Case,
                symbols: &Rc<RefCell<SymbolTable>>,
                errors: &mut Vec<(String, Span)>
            ) {
                let mut sub_symbols = get_statement_symbols(&mut case.stmt).borrow_mut();
                sub_symbols.set_parent(symbols.clone());

                for (name, span) in case.vars.drain(..) {
                    let define = if let Some(old_sym) = sub_symbols.find_imm_named_symbol(&name) {
                        errors.push((
                            format!(
                                "the name '{}' has already been defined in this scope (original definition is at line {}, col {})",
                                name,
                                old_sym.span.lo.line,
                                old_sym.span.lo.col
                            ),
                            span
                        ));
                        false
                    } else {
                        true
                    };

                    if define {
                        case.var_bindings.push(sub_symbols.add_symbol(Symbol {
                            id: 0,
                            name: name,
                            span: span,
                            node: SymbolType::Var(VarSymbol {
                                val_type: Type::Error,
                                dims: RefCell::new(vec![])
                            })
                        }));
                    } else {
                        case.var_bindings.push(!0);
                    };
                };
            }

            fn analyze_case(
                case: &mut ast::Case,
                typedef: &DataTypeDefinition,
                symbols: &Rc<RefCell<SymbolTable>>,
                errors: &mut Vec<(String, Span)>
            ) {
                let (ctor_id, ctor) = if let Some(ctor) = typedef.ctors.iter().enumerate().find(|&(id, ctor)| ctor.name == case.cid) {
                    ctor
                } else {
                    errors.push((
                        format!(
                            "no constructor #{} exists for type {}",
                            case.cid,
                            typedef.name
                        ),
                        case.span
                    ));

                    analyze_case_error(case, symbols, errors);
                    return;
                };

                case.ctor_id = ctor_id;

                if ctor.args.len() != case.vars.len() {
                    errors.push((
                        format!(
                            "wrong number of arguments to destructure: expected {}, but found {}",
                            ctor.args.len(),
                            case.vars.len()
                        ),
                        case.span
                    ));

                    analyze_case_error(case, symbols, errors);
                    return;
                };

                let mut sub_symbols = get_statement_symbols(&mut case.stmt).borrow_mut();
                sub_symbols.set_parent(symbols.clone());

                for ((name, span), val_type) in case.vars.drain(..).zip(ctor.args.iter()) {
                    let define = if let Some(old_sym) = sub_symbols.find_imm_named_symbol(&name) {
                        errors.push((
                            format!(
                                "the name '{}' has already been defined in this scope (original definition is at line {}, col {})",
                                name,
                                old_sym.span.lo.line,
                                old_sym.span.lo.col
                            ),
                            span
                        ));
                        false
                    } else {
                        true
                    };

                    if define {
                        case.var_bindings.push(sub_symbols.add_symbol(Symbol {
                            id: 0,
                            name: name,
                            span: span,
                            node: SymbolType::Var(VarSymbol {
                                val_type: val_type.clone(),
                                dims: RefCell::new(vec![])
                            })
                        }));
                    } else {
                        case.var_bindings.push(!0);
                    };
                };
            }

            {
                let val_type = analyze_expression(val, tdt, &symbols.borrow(), None, errors);
                let typedef = if let Type::Defined(ref type_id) = val_type {
                    if let TypeDefinition::Data(ref typedef) = tdt.defs[*type_id] {
                        Some(typedef)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(typedef) = typedef {
                    for case in cases.iter_mut() {
                        analyze_case(case, typedef, symbols, errors);
                    };
                } else {
                    errors.push((
                        format!("cannot pattern match on a value of type {}", val_type.pretty(tdt)),
                        val.span
                    ));

                    for case in cases.iter_mut() {
                        analyze_case_error(case, symbols, errors);
                    };
                };
            };

            for case in cases {
                analyze_statement(&mut case.stmt, tdt, symbols, expected_return, errors);
            };
        },
        ast::StmtType::Return(ref mut val) => {
            let val_type = analyze_expression(val, tdt, &symbols.borrow(), expected_return, errors);

            if let Some(expected_return) = expected_return {
                if !val_type.can_convert_to_exact(expected_return) {
                    errors.push((
                        format!("cannot convert from {} to {}", val_type.pretty(tdt), expected_return.pretty(tdt)),
                        val.span
                    ));
                };
            } else {
                errors.push((
                    "cannot return from outside a function".to_string(),
                    stmt.span
                ));
            };
        }
    };
}

fn populate_block_symbol_table(
    tdt: &mut TypeDefinitionTable,
    block: &mut ast::Block,
    expected_return: Option<&Type>,
    errors: &mut Vec<(String, Span)>
) {
    {
        let symbols = &mut block.symbols.borrow_mut();

        // To achieve simultaneous declaration syntax, we need to set up some dummy type definitions
        // before we can start processing the actual declarations. This is because types can be
        // mutually recursive and constructor type information needs to be resolved as the data
        // declaration is being processed. No such dummy definitions are needed for variables or
        // functions since they can only be referenced from blocks and expressions, which aren't
        // processed until after all declarations have been processed.
        for decl in &mut block.decls {
            if let ast::DeclType::Data(ref name, ref mut type_id, _) = decl.node {
                *type_id = tdt.add_definition(TypeDefinition::Dummy);
                if let Some((_, old_type_span)) = symbols.find_imm_named_type(&name) {
                    errors.push((
                        format!(
                            "a type '{}' already exists in this scope (original definition is at line {}, col {})",
                            name,
                            old_type_span.lo.line,
                            old_type_span.lo.col
                        ),
                        decl.span
                    ));
                } else {
                    symbols.add_type(name.clone(), *type_id, decl.span);
                };
            };
        };

        for decl in mem::replace(&mut block.decls, vec![]) {
            create_symbol_for_decl(tdt, symbols, errors, decl);
        };
    };

    {
        let symbols = &block.symbols.borrow();

        for (_, sym) in &symbols.symbols {
            match sym.node {
                SymbolType::Fun(ref fs) => {
                    populate_function_symbol_table(tdt, &block.symbols, &fs, errors);
                },
                SymbolType::Var(ref vs) => {
                    for d in vs.dims.borrow_mut().iter_mut() {
                        let dim_type = analyze_expression(d, tdt, symbols, Some(&Type::Int), errors);

                        if !dim_type.can_convert_to_exact(&Type::Int) {
                            errors.push((
                                format!("cannot convert from {} to int", dim_type.pretty(tdt)),
                                d.span
                            ));
                        };
                    };
                },
                _ => {}
            };
        };
    };

    for stmt in &mut block.stmts {
        analyze_statement(stmt, tdt, &block.symbols, expected_return, errors);
    };
}

fn populate_function_symbol_table(
    tdt: &mut TypeDefinitionTable,
    parent: &Rc<RefCell<SymbolTable>>,
    sym: &FunSymbol,
    errors: &mut Vec<(String, Span)>
) {
    let block = &mut sym.body.borrow_mut();
    let return_type = if let TypeDefinition::Function(ref sig) = &tdt.defs[sym.sig] {
        sig.return_type.clone()
    } else {
        panic!("Invalid function signature")
    };

    block.symbols.borrow_mut().set_parent(parent.clone());

    populate_block_symbol_table(
        tdt,
        block,
        Some(&return_type),
        errors
    )
}

pub fn populate_symbol_tables(
    program: &mut ast::Program,
    errors: &mut Vec<(String, Span)>
) {
    populate_block_symbol_table(
        &mut program.types,
        &mut program.block,
        None,
        errors
    );
}
