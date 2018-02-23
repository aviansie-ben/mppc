use std::fmt;

use lex::Span;
use util::PrettyDisplay;

#[derive(Debug, Clone)]
pub struct Block {
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>
}

impl Block {
    pub fn new(decls: Vec<Decl>, stmts: Vec<Stmt>) -> Block {
        Block { decls: decls, stmts: stmts }
    }
}

impl PrettyDisplay for Block {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        let mut next_indent = indent.to_string();
        next_indent.push_str(" ");

        write!(f, "{}Block", indent)?;

        for decl in &self.decls {
            write!(f, "\n{}", decl.pretty_indented(&next_indent))?;
        };

        for stmt in &self.stmts {
            write!(f, "\n{}", stmt.pretty_indented(&next_indent))?;
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Id(String, Span)
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ast::Type::*;
        match *self {
            Int => write!(f, "int"),
            Real => write!(f, "real"),
            Bool => write!(f, "bool"),
            Char => write!(f, "char"),
            Id(ref name, _) => write!(f, "{}", name)
        }
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
    pub dims: u32,
    pub span: Span
}

impl FunParam {
    pub fn new(id: String, val_type: Type, dims: u32) -> FunParam {
        FunParam { id: id, val_type: val_type, dims: dims, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> FunParam {
        self.span = span;
        self
    }
}

impl PrettyDisplay for FunParam {
    fn fmt(&self, indent: &str, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{} {} {}", indent, self.id, self.val_type, self.dims)
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
    Var(Vec<VarSpec>, Type),
    Fun(String, FunSig, Block),
    Data(String, Vec<DataCons>)
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub node: DeclType,
    pub span: Span
}

impl Decl {
    pub fn new(node: DeclType) -> Decl {
        Decl { node: node, span: Span::dummy() }
    }

    pub fn var(specs: Vec<VarSpec>, val_type: Type) -> Decl {
        Decl::new(DeclType::Var(specs, val_type))
    }

    pub fn func(name: String, sig: FunSig, body: Block) -> Decl {
        Decl::new(DeclType::Fun(name, sig, body))
    }

    pub fn data(name: String, ctors: Vec<DataCons>) -> Decl {
        Decl::new(DeclType::Data(name, ctors))
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
            Var(ref var_specs, ref val_type) => {
                write!(f, "{}VarDecl {}", indent, val_type)?;

                for var_spec in var_specs {
                    write!(f, "\n{}", var_spec.pretty_indented(&next_indent))?;
                }
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
            Data(ref id, ref ctors) => {
                write!(f, "{}DataDecl {}", indent, id)?;

                for ctor in ctors {
                    write!(f, "\n{}", ctor.pretty_indented(&next_indent))?;
                };
            }
        };

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ExprType {
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Equal(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Size(String, u32),
    Float(Box<Expr>),
    Floor(Box<Expr>),
    Ceil(Box<Expr>),
    Id(String),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Cons(String, Vec<Expr>),
    Int(i32),
    Real(f64),
    Bool(bool),
    Char(char),
    Neg(Box<Expr>)
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub node: ExprType,
    pub span: Span
}

impl Expr {
    pub fn new(node: ExprType) -> Expr {
        Expr { node: node, span: Span::dummy() }
    }

    pub fn or(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Or(Box::new(lhs), Box::new(rhs)))
    }

    pub fn and(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::And(Box::new(lhs), Box::new(rhs)))
    }

    pub fn not(val: Expr) -> Expr {
        Expr::new(ExprType::Not(Box::new(val)))
    }

    pub fn equal_to(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Equal(Box::new(lhs), Box::new(rhs)))
    }

    pub fn less_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Lt(Box::new(lhs), Box::new(rhs)))
    }

    pub fn greater_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Ge(Box::new(lhs), Box::new(rhs)))
    }

    pub fn less_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Le(Box::new(lhs), Box::new(rhs)))
    }

    pub fn greater_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Ge(Box::new(lhs), Box::new(rhs)))
    }

    pub fn add(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Add(Box::new(lhs), Box::new(rhs)))
    }

    pub fn sub(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Sub(Box::new(lhs), Box::new(rhs)))
    }

    pub fn mul(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Mul(Box::new(lhs), Box::new(rhs)))
    }

    pub fn div(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Div(Box::new(lhs), Box::new(rhs)))
    }

    pub fn size_of(id: String, array_derefs: u32) -> Expr {
        Expr::new(ExprType::Size(id, array_derefs))
    }

    pub fn float(val: Expr) -> Expr {
        Expr::new(ExprType::Float(Box::new(val)))
    }

    pub fn floor(val: Expr) -> Expr {
        Expr::new(ExprType::Floor(Box::new(val)))
    }

    pub fn ceil(val: Expr) -> Expr {
        Expr::new(ExprType::Ceil(Box::new(val)))
    }

    pub fn identifier(id: String) -> Expr {
        Expr::new(ExprType::Id(id))
    }

    pub fn call(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Call(Box::new(func), args))
    }

    pub fn index(arr: Expr, ind: Expr) -> Expr {
        Expr::new(ExprType::Index(Box::new(arr), Box::new(ind)))
    }

    pub fn cons(cid: String, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Cons(cid, args))
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
        Expr::new(ExprType::Neg(Box::new(val)))
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

        match self.node {
            Or(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Or\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            And(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}And\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Not(ref val) => {
                write!(
                    f,
                    "{}Not\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Equal(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Equal\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Lt(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Lt\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Gt(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Gt\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Le(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Le\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Ge(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Ge\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Add(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Add\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Sub(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Sub\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Mul(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Mul\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Div(ref lhs, ref rhs) => {
                write!(
                    f,
                    "{}Div\n{}\n{}",
                    indent,
                    lhs.pretty_indented(&next_indent),
                    rhs.pretty_indented(&next_indent)
                )?;
            },
            Size(ref id, ref dims) => {
                write!(
                    f,
                    "{}Size {} {}",
                    indent,
                    id,
                    dims
                )?;
            },
            Float(ref val) => {
                write!(
                    f,
                    "{}Float\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Floor(ref val) => {
                write!(
                    f,
                    "{}Floor\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Ceil(ref val) => {
                write!(
                    f,
                    "{}Ceil\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            },
            Id(ref id) => {
                write!(
                    f,
                    "{}Id {}",
                    indent,
                    id
                )?;
            },
            Call(ref func, ref args) => {
                write!(
                    f,
                    "{}Call\n{}",
                    indent,
                    func.pretty_indented(&next_indent)
                )?;
                for a in args {
                    write!(f, "\n{}", a.pretty_indented(&next_indent))?;
                };
            },
            Index(ref arr, ref ind) => {
                write!(
                    f,
                    "{}Index\n{}\n{}",
                    indent,
                    arr.pretty_indented(&next_indent),
                    ind.pretty_indented(&next_indent)
                )?;
            },
            Cons(ref cid, ref args) => {
                write!(f, "{}Cons #{}", indent, cid)?;
                for a in args {
                    write!(f, "\n{}", a.pretty_indented(&next_indent))?;
                };
            },
            Int(ref val) => write!(f, "{}Int {}", indent, val)?,
            Real(ref val) => write!(f, "{}Real {}", indent, val)?,
            Bool(ref val) => write!(f, "{}Bool {}", indent, val)?,
            Char(ref val) => write!(f, "{}Char \"{}\"", indent, val)?,
            Neg(ref val) => {
                write!(
                    f,
                    "{}Neg\n{}",
                    indent,
                    val.pretty_indented(&next_indent)
                )?;
            }
        };
        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub cid: String,
    pub vars: Vec<(String, Span)>,
    pub stmt: Stmt,
    pub span: Span
}

impl Case {
    pub fn new(cid: String, vars: Vec<(String, Span)>, stmt: Stmt) -> Case {
        Case { cid: cid, vars: vars, stmt: stmt, span: Span::dummy() }
    }

    pub fn at(mut self, span: Span) -> Case {
        self.span = span;
        self
    }
}

impl PrettyDisplay for Case {
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
        }

        write!(f, "\n{}", self.stmt.pretty_indented(&next_indent))?;

        Result::Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum StmtType {
    IfThenElse(Expr, Box<Stmt>, Box<Stmt>),
    WhileDo(Expr, Box<Stmt>),
    Read(Expr),
    Assign(Expr, Expr),
    Print(Expr),
    Block(Block),
    Case(Expr, Vec<Case>),
    Return(Expr)
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub node: StmtType,
    pub span: Span
}

impl Stmt {
    pub fn new(node: StmtType) -> Stmt {
        Stmt { node: node, span: Span::dummy() }
    }

    pub fn if_then_else(cond: Expr, true_stmt: Stmt, false_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::IfThenElse(cond, Box::new(true_stmt), Box::new(false_stmt)))
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

    pub fn case(val: Expr, cases: Vec<Case>) -> Stmt {
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
                write!(
                    f,
                    "{}IfThenElse\n{}\n{}\n{}",
                    indent,
                    cond.pretty_indented(&next_indent),
                    true_stmt.pretty_indented(&next_indent),
                    false_stmt.pretty_indented(&next_indent)
                )?;
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
