use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use lex::Span;
use symbol;
use util::{DeferredDisplay, PrettyDisplay};

#[derive(Debug)]
pub struct Program {
    pub block: Block,
    pub features: HashMap<String, Span>,
    pub symbols: symbol::SymbolDefinitionTable,
    pub types: symbol::TypeDefinitionTable
}

impl Program {
    pub fn new(block: Block, features: HashMap<String, Span>) -> Program {
        Program {
            block: block,
            features: features,
            symbols: symbol::SymbolDefinitionTable::new(),
            types: symbol::TypeDefinitionTable::new()
        }
    }
}

impl PrettyDisplay for Program {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        for (feature, _) in &self.features {
            write!(f, "{}Feature {}\n", indent, feature)?;
        };

        for (id, td) in &self.types {
            write!(f, "{}TypeDef {}\n{}\n", indent, id, td.pretty_indented(&next_indent))?;
        };

        for (_, sym) in &self.symbols {
            write!(f, "{}\n", sym.pretty_indented(indent))?;
        };

        write!(f, "{}", self.block.pretty_indented(indent))?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>,
    pub symbols: Rc<RefCell<symbol::SymbolTable>>
}

impl Block {
    pub fn new(decls: Vec<Decl>, stmts: Vec<Stmt>) -> Block {
        Block {
            decls: decls,
            stmts: stmts,
            symbols: Rc::new(RefCell::new(symbol::SymbolTable::new()))
        }
    }
}

impl PrettyDisplay for Block {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        let mut next_next_indent = next_indent.to_string();
        next_next_indent.push_str("  ");

        write!(f, "{}Block", indent)?;

        {
            let symbols = self.symbols.borrow();

            for (name, &(id, _)) in &symbols.type_names {
                write!(f, "\n{}TypeRef {} {}", next_indent, name, id)?;
            };

            for (name, id) in &symbols.symbol_names {
                write!(f, "\n{}SymRef {} {}", next_indent, name, id)?;
            };
        };

        for decl in &self.decls {
            write!(f, "\n{}", decl.pretty_indented(&next_indent))?;
        };

        for stmt in &self.stmts {
            write!(f, "\n{}", stmt.pretty_indented(&next_indent))?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Id(String, Span),
    Array(Box<Type>, u32)
}

impl Type {
    pub fn wrap_in_array(self, dims: u32) -> Type {
        use ast::Type::*;

        if dims == 0 {
            self
        } else {
            match self {
                Array(t, prev_dims) => Array(t, prev_dims + dims),
                t => Array(Box::new(t), dims)
            }
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::Type::*;
        match *self {
            Int => write!(f, "int")?,
            Real => write!(f, "real")?,
            Bool => write!(f, "bool")?,
            Char => write!(f, "char")?,
            Id(ref name, _) => write!(f, "{}", name)?,
            Array(ref inner_type, ref dims) => {
                write!(f, "{}", inner_type)?;

                for _ in 0..*dims {
                    write!(f, "[]")?;
                };
            }
        };
        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct VarSpec {
    pub id: String,
    pub dims: Vec<Expr>,
    pub span: Span
}

impl VarSpec {
    pub fn new(id: String, dims: Vec<Expr>) -> VarSpec {
        VarSpec { id: id, dims: dims, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> VarSpec {
        self.span = span;
        self
    }
}

impl PrettyDisplay for VarSpec {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}VarSpec {}", indent, self.id)?;

        for dim in &self.dims {
            write!(f, "\n{}", dim.pretty_indented(&next_indent))?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FunParam {
    pub id: String,
    pub val_type: Type,
    pub span: Span
}

impl FunParam {
    pub fn new(id: String, val_type: Type) -> FunParam {
        FunParam { id: id, val_type: val_type, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> FunParam {
        self.span = span;
        self
    }
}

impl PrettyDisplay for FunParam {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{} {}", indent, self.id, self.val_type)
    }
}

#[derive(Debug, Clone)]
pub struct FunSig {
    pub params: Vec<FunParam>,
    pub return_type: Type
}

impl FunSig {
    pub fn new(params: Vec<FunParam>, return_type: Type) -> FunSig {
        FunSig { params: params, return_type: return_type }
    }
}

impl PrettyDisplay for FunSig {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str("  ");

        write!(f, "{}FunSig\n{} Params", indent, indent)?;

        for param in &self.params {
            write!(f, "\n{}", param.pretty_indented(&next_indent))?;
        };

        write!(f, "\n{} Returns {}", indent, self.return_type)?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DataCons {
    pub cid: String,
    pub types: Vec<Type>,
    pub span: Span
}

impl DataCons {
    pub fn new(cid: String, types: Vec<Type>) -> DataCons {
        DataCons { cid: cid, types: types, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> DataCons {
        self.span = span;
        self
    }
}

impl PrettyDisplay for DataCons {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}#{}", indent, self.cid)?;

        if self.types.len() != 0 {
            write!(f, " of {}", self.types.first().unwrap())?;

            for t in &self.types[1..] {
                write!(f, " * {}", t)?;
            };
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum DeclType {
    Var(VarSpec, Type),
    Fun(String, FunSig, Block),
    Data(String, usize, Vec<DataCons>)
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub node: DeclType,
    pub span: Span
}

// This enum is used when parsing a single declaration. The reason for this is that variable
// declarations will actually desugar into multiple Decls due to the fact that multiple variables
// can be declared in a single declaration. However, we want to avoid allocating heap space for a
// Vec<Decl> for other Decls that don't exhibit this behaviour.
#[derive(Debug, Clone)]
pub enum MultiDecl {
    Single(Decl),
    Multiple(Vec<Decl>)
}

impl Decl {
    pub fn new(node: DeclType) -> Decl {
        Decl { node: node, span: Span::dummy() }
    }

    pub fn var(spec: VarSpec, val_type: Type) -> Decl {
        Decl::new(DeclType::Var(spec, val_type))
    }

    pub fn func(name: String, sig: FunSig, body: Block) -> Decl {
        Decl::new(DeclType::Fun(name, sig, body))
    }

    pub fn data(name: String, ctors: Vec<DataCons>) -> Decl {
        Decl::new(DeclType::Data(name, !0, ctors))
    }

    pub fn at(mut self, span: Span) -> Decl {
        self.span = span;
        self
    }
}

impl PrettyDisplay for Decl {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::DeclType::*;

        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        match self.node {
            Var(ref var_spec, ref val_type) => {
                write!(
                    f,
                    "{}VarDecl {}\n{}",
                    indent,
                    val_type,
                    var_spec.pretty_indented(&next_indent)
                )?;
            },
            Fun(ref id, ref sig, ref block) => {
                write!(
                    f,
                    "{}FunDecl {}\n{}\n{}",
                    indent,
                    id,
                    sig.pretty_indented(&next_indent),
                    block.pretty_indented(&next_indent)
                )?;
            },
            Data(ref id, _, ref ctors) => {
                write!(f, "{}DataDecl {}", indent, id)?;

                for ctor in ctors {
                    write!(f, "\n{}", ctor.pretty_indented(&next_indent))?;
                };
            }
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Or,
    And,
    Equal,
    Lt,
    Gt,
    Le,
    Ge,
    Add,
    Sub,
    Mul,
    Div
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::BinaryOp::*;
        match self {
            Or => write!(f, "||"),
            And => write!(f, "&&"),
            Equal => write!(f, "="),
            Lt => write!(f, "<"),
            Gt => write!(f, ">"),
            Le => write!(f, "=<"),
            Ge => write!(f, ">="),
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    Float,
    Floor,
    Ceil,
    Neg
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::UnaryOp::*;
        match self {
            Not => write!(f, "not"),
            Float => write!(f, "float"),
            Floor => write!(f, "floor"),
            Ceil => write!(f, "ceil"),
            Neg => write!(f, "-")
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExprType {
    BinaryOp(BinaryOp, Box<Expr>, Box<Expr>, symbol::BinaryOp),
    UnaryOp(UnaryOp, Box<Expr>, symbol::UnaryOp),
    Size(Box<Expr>, u32),
    Id(String, usize),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Cons(String, Vec<Expr>, usize),
    Int(i32),
    Real(f64),
    Bool(bool),
    Char(char),
    Block(Block, Option<Box<Expr>>),
    Case(Box<Expr>, Vec<Case<Expr>>)
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub node: ExprType,
    pub span: Span,
    pub val_type: symbol::Type,
    pub assignable: bool,
    pub synthetic: bool
}

impl Expr {
    pub fn new(node: ExprType) -> Expr {
        Expr {
            node: node,
            span: Span::dummy(),
            val_type: symbol::Type::Unknown,
            assignable: false,
            synthetic: false
        }
    }

    pub fn or(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Or, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn and(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::And, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn not(val: Expr) -> Expr {
        Expr::new(ExprType::UnaryOp(UnaryOp::Not, Box::new(val), symbol::UnaryOp::Unknown))
    }

    pub fn equal_to(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Equal, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn less_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Lt, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn greater_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Gt, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn less_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Le, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn greater_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Ge, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn add(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Add, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn sub(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Sub, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn mul(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Mul, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn div(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::BinaryOp(BinaryOp::Div, Box::new(lhs), Box::new(rhs), symbol::BinaryOp::Unknown))
    }

    pub fn size_of(id: String, array_derefs: u32) -> Expr {
        Expr::new(ExprType::Size(Box::new(Expr::identifier(id)), array_derefs))
    }

    pub fn float(val: Expr) -> Expr {
        Expr::new(ExprType::UnaryOp(UnaryOp::Float, Box::new(val), symbol::UnaryOp::Unknown))
    }

    pub fn floor(val: Expr) -> Expr {
        Expr::new(ExprType::UnaryOp(UnaryOp::Floor, Box::new(val), symbol::UnaryOp::Unknown))
    }

    pub fn ceil(val: Expr) -> Expr {
        Expr::new(ExprType::UnaryOp(UnaryOp::Ceil, Box::new(val), symbol::UnaryOp::Unknown))
    }

    pub fn identifier(id: String) -> Expr {
        Expr::new(ExprType::Id(id, !0))
    }

    pub fn call(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Call(Box::new(func), args))
    }

    pub fn index(arr: Expr, ind: Expr) -> Expr {
        Expr::new(ExprType::Index(Box::new(arr), Box::new(ind)))
    }

    pub fn cons(cid: String, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Cons(cid, args, !0))
    }

    pub fn int(val: i32) -> Expr {
        Expr::new(ExprType::Int(val))
    }

    pub fn real(val: f64) -> Expr {
        Expr::new(ExprType::Real(val))
    }

    pub fn bool(val: bool) -> Expr {
        Expr::new(ExprType::Bool(val))
    }

    pub fn char(val: char) -> Expr {
        Expr::new(ExprType::Char(val))
    }

    pub fn negate(val: Expr) -> Expr {
        Expr::new(ExprType::UnaryOp(UnaryOp::Neg, Box::new(val), symbol::UnaryOp::Unknown))
    }

    pub fn block(block: Block, result: Option<Expr>) -> Expr {
        Expr::new(ExprType::Block(block, result.map(Box::new)))
    }

    pub fn case(val: Expr, cases: Vec<Case<Expr>>) -> Expr {
        Expr::new(ExprType::Case(Box::new(val), cases))
    }

    pub fn at(mut self, span: Span) -> Expr {
        self.span = span;
        self
    }
}

impl PrettyDisplay for Expr {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::ExprType::*;

        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        let meta_display = DeferredDisplay(|f| {
            if self.val_type != symbol::Type::Unknown {
                write!(f, " [TYPE: {}]", self.val_type)?;
            };

            if self.assignable {
                write!(f, " [ASSIGNABLE]")?;
            };

            Result::Ok(())
        });

        match self.node {
            BinaryOp(op, ref lhs, ref rhs, sym_op) => {
                let meta_display = DeferredDisplay(|f| {
                    if sym_op != symbol::BinaryOp::Unknown {
                        write!(f, "{} [OP: {:?}]", meta_display, sym_op)
                    } else {
                        write!(f, "{}", meta_display)
                    }
                });
                write!(
                    f,
                    "{}{:?}{}\n{}\n{}",
                    indent,
                    op,
                    meta_display,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            UnaryOp(op, ref val, sym_op) => {
                let meta_display = DeferredDisplay(|f| {
                    if sym_op != symbol::UnaryOp::Unknown {
                        write!(f, "{} [OP: {:?}]", meta_display, sym_op)
                    } else {
                        write!(f, "{}", meta_display)
                    }
                });
                write!(
                    f,
                    "{}{:?}{}\n{}",
                    indent,
                    op,
                    meta_display,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Size(ref val, ref dims) => {
                write!(
                    f,
                    "{}Size {}{}\n{}",
                    indent,
                    dims,
                    meta_display,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Id(ref id, sym_id) => {
                let meta_display = DeferredDisplay(|f| {
                    if sym_id != !0 {
                        write!(f, "{} [SYM: {}]", meta_display, sym_id)
                    } else {
                        write!(f, "{}", meta_display)
                    }
                });
                write!(
                    f,
                    "{}Id {}{}",
                    indent,
                    id,
                    meta_display
                )?;
            },
            Call(ref func, ref args) => {
                write!(
                    f,
                    "{}Call{}\n{}",
                    indent,
                    meta_display,
                    func.pretty_indented(&next_indent)
                )?;
                for a in args {
                    write!(f, "\n{}", a.pretty_indented(&next_indent))?;
                };
            },
            Index(ref arr, ref ind) => {
                write!(
                    f,
                    "{}Index{}\n{}\n{}",
                    indent,
                    meta_display,
                    arr.pretty_indented(&next_indent),
                    ind.pretty_indented(&next_indent)
                )?;
            },
            Cons(ref cid, ref args, ctor_id) => {
                write!(f, "{}Cons #{}{}", indent, cid, meta_display)?;

                if ctor_id != !0 {
                    write!(f, " [CTOR: {}]", ctor_id)?;
                };

                for a in args {
                    write!(f, "\n{}", a.pretty_indented(&next_indent))?;
                };
            },
            Int(ref val) => write!(f, "{}Int {}{}", indent, val, meta_display)?,
            Real(ref val) => write!(f, "{}Real {}{}", indent, val, meta_display)?,
            Bool(ref val) => write!(f, "{}Bool {}{}", indent, val, meta_display)?,
            Char(ref val) => write!(f, "{}Char {:?}{}", indent, val, meta_display)?,
            Block(ref block, ref result) => {
                write!(f, "{}", block.pretty_indented(indent))?;

                if let Some(ref result) = *result {
                    write!(f, "\n{}", result.pretty_indented(&next_indent))?;
                };
            },
            Case(ref val, ref cases) => {
                write!(f, "{}Case\n{}", indent, val.pretty_indented(&next_indent))?;

                for c in cases {
                    write!(f, "\n{}", c.pretty_indented(&next_indent))?;
                };
            }
        };
        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub cid: String,
    pub ctor_id: usize,
    pub vars: Vec<(String, Span)>,
    pub var_bindings: Vec<usize>,
    pub branch: T,
    pub span: Span
}

impl <T> Case<T> {
    pub fn new(cid: String, vars: Vec<(String, Span)>, branch: T) -> Case<T> {
        Case { cid: cid, ctor_id: !0, vars: vars, var_bindings: Vec::new(), branch: branch, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> Case<T> {
        self.span = span;
        self
    }
}

impl <T: PrettyDisplay> PrettyDisplay for Case<T> {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}#{}", indent, self.cid)?;

        if self.vars.len() != 0 {
            write!(f, "({}", self.vars.first().unwrap().0)?;
            for (var, _) in &self.vars[1..] {
                write!(f, ", {}", var)?;
            }
            write!(f, ")")?;
        };

        if self.var_bindings.len() != 0 {
            write!(f, "(${}", self.var_bindings.first().unwrap())?;
            for var in &self.var_bindings[1..] {
                write!(f, ", ${}", var)?;
            };
            write!(f, ")")?;
        };

        if self.ctor_id != !0 {
            write!(f, " [CTOR: {}]", self.ctor_id)?;
        };

        write!(f, "\n{}", self.branch.pretty_indented(&next_indent))?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum StmtType {
    IfThenElse(Expr, Box<Stmt>, Option<Box<Stmt>>),
    WhileDo(Expr, Box<Stmt>),
    Read(Expr),
    Assign(Expr, Expr),
    Print(Expr),
    Block(Block),
    Case(Expr, Vec<Case<Stmt>>),
    Return(Expr)
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub node: StmtType,
    pub span: Span,
    pub will_return: bool
}

impl Stmt {
    pub fn new(node: StmtType) -> Stmt {
        Stmt { node: node, span: Span::dummy(), will_return: false }
    }

    pub fn if_then_else(cond: Expr, true_stmt: Stmt, false_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::IfThenElse(cond, Box::new(true_stmt), Some(Box::new(false_stmt))))
    }

    pub fn if_then(cond: Expr, true_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::IfThenElse(cond, Box::new(true_stmt), None))
    }

    pub fn while_do(cond: Expr, do_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::WhileDo(cond, Box::new(do_stmt)))
    }

    pub fn read(target: Expr) -> Stmt {
        Stmt::new(StmtType::Read(target))
    }

    pub fn assign(target: Expr, val: Expr) -> Stmt {
        Stmt::new(StmtType::Assign(target, val))
    }

    pub fn print(val: Expr) -> Stmt {
        Stmt::new(StmtType::Print(val))
    }

    pub fn block(block: Block) -> Stmt {
        Stmt::new(StmtType::Block(block))
    }

    pub fn case(val: Expr, cases: Vec<Case<Stmt>>) -> Stmt {
        Stmt::new(StmtType::Case(val, cases))
    }

    pub fn return_value(val: Expr) -> Stmt {
        Stmt::new(StmtType::Return(val))
    }

    pub fn at(mut self, span: Span) -> Stmt {
        self.span = span;
        self
    }
}

impl PrettyDisplay for Stmt {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::StmtType::*;

        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        match self.node {
            IfThenElse(ref cond, ref true_stmt, ref false_stmt) => {
                if let Some(false_stmt) = false_stmt {
                    write!(
                        f,
                        "{}IfThenElse\n{}\n{}\n{}",
                        indent,
                        cond.pretty_indented(&next_indent),
                        true_stmt.pretty_indented(&next_indent),
                        false_stmt.pretty_indented(&next_indent)
                    )?;
                } else {
                    write!(
                        f,
                        "{}IfThen\n{}\n{}",
                        indent,
                        cond.pretty_indented(&next_indent),
                        true_stmt.pretty_indented(&next_indent)
                    )?;
                };
            },
            WhileDo(ref cond, ref do_stmt) => {
                write!(
                    f,
                    "{}WhileDo\n{}\n{}",
                    indent,
                    cond.pretty_indented(&next_indent),
                    do_stmt.pretty_indented(&next_indent)
                )?;
            },
            Read(ref target) => {
                write!(
                    f,
                    "{}Read\n{}",
                    indent,
                    target.pretty_indented(&next_indent)
                )?;
            },
            Assign(ref target, ref val) => {
                write!(
                    f,
                    "{}Assign\n{}\n{}",
                    indent,
                    target.pretty_indented(&next_indent),
                    val.pretty_indented(&next_indent)
                )?;
            },
            Print(ref val) => {
                write!(
                    f,
                    "{}Print\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Block(ref block) => write!(f, "{}", block.pretty_indented(indent))?,
            Case(ref val, ref cases) => {
                write!(
                    f,
                    "{}Case\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;

                for case in cases {
                    write!(f, "\n{}", case.pretty_indented(&next_indent))?;
                };
            },
            Return(ref val) => {
                write!(
                    f,
                    "{}Return\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            }
        };
        Result::Ok(())
    }
}
