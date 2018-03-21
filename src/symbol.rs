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
    pub dims: Vec<ast::Expr>
}

impl PrettyDisplay for VarSymbol {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Var {}", indent, self.val_type)?;

        for dim in &self.dims {
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
    pub symbols: HashMap<usize, Symbol>,
    pub parent: Option<Rc<RefCell<SymbolTable>>>,
    next_id: Rc<Cell<usize>>
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            symbol_names: HashMap::new(),
            type_names: HashMap::new(),
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
    Error
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
                write!(f, "one of {}", possible_types[0])?;

                if possible_types.len() >= 3 {
                    for i in 1..(possible_types.len() - 1) {
                        write!(f, ", {}", possible_types[i])?;
                    };

                    write!(f, ", or {}", possible_types[possible_types.len() - 1])?;
                } else {
                    write!(f, " or {}", possible_types[1])?;
                }
            },
            Error => write!(f, "(error type)")?
        };
        Result::Ok(())
    }
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

    pub fn find_ctors(&self, name: &str, symtab: &SymbolTable, ctors: &mut Vec<(usize, usize)>) -> () {
        for (_, (id, _)) in &symtab.type_names {
            if let TypeDefinition::Data(td) = &self.defs[*id] {
                for (ctor_id, ctor) in td.ctors.iter().enumerate() {
                    if &ctor.name == name {
                        ctors.push((*id, ctor_id));
                    };
                };
            };
        };

        if let Some(ref parent) = symtab.parent {
            self.find_ctors(name, &parent.borrow(), ctors);
        };
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
        ast::DeclType::Data(name, ctors) => {
            // Data types can be defined recursively, so we add a dummy definition now.
            let type_id = tdt.add_definition(TypeDefinition::Dummy);
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
                symbols.add_type(name.clone(), type_id, decl.span);
            };

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
                        dims: spec.dims
                    })
                });
            };
        }
    };
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
            // TODO Typecheck condition

            analyze_statement(then_stmt, tdt, symbols, expected_return, errors);
            analyze_statement(else_stmt, tdt, symbols, expected_return, errors);
        },
        ast::StmtType::WhileDo(ref mut cond, ref mut do_stmt) => {
            // TODO Typecheck condition

            analyze_statement(do_stmt, tdt, symbols, expected_return, errors);
        },
        ast::StmtType::Read(ref mut loc) => {
            // TODO Typecheck location
        },
        ast::StmtType::Assign(ref mut loc, ref mut val) => {
            // TODO Typecheck location
            // TODO Typecheck value
        },
        ast::StmtType::Print(ref mut val) => {
            // TODO Typecheck value
        },
        ast::StmtType::Block(ref mut inner_block) => {
            inner_block.symbols.borrow_mut().set_parent(symbols.clone());
            populate_block_symbol_table(tdt, inner_block, expected_return, errors);

            inner_block.symbols.borrow_mut().lift_decls(&mut symbols.borrow_mut());
        },
        ast::StmtType::Case(ref mut val, ref mut cases) => {
            // TODO Typecheck value

            for case in cases {
                // TODO Typecheck constructor
                // TODO Define variables

                analyze_statement(&mut case.stmt, tdt, symbols, expected_return, errors);
            };
        },
        ast::StmtType::Return(ref mut val) => {
            // TODO Typecheck value
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

        for decl in mem::replace(&mut block.decls, vec![]) {
            create_symbol_for_decl(tdt, symbols, errors, decl);
        };
    }

    for (_, sym) in &block.symbols.borrow().symbols {
        if let SymbolType::Fun(ref fs) = sym.node {
            populate_function_symbol_table(tdt, &block.symbols, &fs, errors);
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
