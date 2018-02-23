#![allow(unused_mut)]

use std::fmt;

use lex;
use lex::{Span, Token, TokenStream};

type Error = (String, Span);

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

#[derive(Debug, Clone)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Id(String)
}

#[derive(Debug, Clone)]
pub struct VarSpec {
    pub id: String,
    pub dims: Vec<Expr>
}

#[derive(Debug, Clone)]
pub struct FunParam {
    pub id: String,
    pub val_type: Type,
    pub dims: u32
}

impl FunParam {
    pub fn new(id: String, val_type: Type, dims: u32) -> FunParam {
        FunParam { id: id, val_type: val_type, dims: dims }
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

#[derive(Debug, Clone)]
pub struct DataCons {
    pub cid: String,
    pub types: Vec<Type>
}

impl DataCons {
    pub fn new(cid: String, types: Vec<Type>) -> DataCons {
        DataCons { cid: cid, types: types }
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
    pub node: DeclType
}

impl Decl {
    pub fn new(node: DeclType) -> Decl {
        Decl { node: node }
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
    pub node: ExprType
}

impl Expr {
    pub fn new(node: ExprType) -> Expr {
        Expr { node: node }
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
}

#[derive(Debug, Clone)]
pub struct Case {
    pub cid: String,
    pub vars: Vec<String>,
    pub stmt: Stmt
}

impl Case {
    pub fn new(cid: String, vars: Vec<String>, stmt: Stmt) -> Case {
        Case { cid: cid, vars: vars, stmt: stmt }
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
    pub node: StmtType
}

impl Stmt {
    pub fn new(node: StmtType) -> Stmt {
        Stmt { node: node }
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
}

parser! {
    fn _parse(Token, Span);

    (a, b) { lex::combine_spans(a, b) }

    block: Block {
        decls[ds] stmts[ss] => Block::new(ds, ss)
    }

    decls: Vec<Decl> {
        => vec![],
        decls[mut ds] decl[d] Semicolon => {
            ds.push(d);
            ds
        }
    }

    decl: Decl {
        Var var_specs[vss] Colon type_[t] => Decl::var(vss, t),
        Fun Id(id) LPar params[ps] RPar Colon type_[rt] CLPar decls[ds] fun_body[body] CRPar => Decl::func(
            id,
            FunSig::new(ps, rt),
            Block::new(ds, body)
        ),
        Data Id(id) Equal cons_decls[cs] => Decl::data(id, cs)
    }

    type_: Type {
        Int => Type::Int,
        Real => Type::Real,
        Bool => Type::Bool,
        Char => Type::Char,
        Id(id) => Type::Id(id)
    }

    var_specs: Vec<VarSpec> {
        var_spec[vs] => vec![vs],
        var_specs[mut vss] Comma var_spec[vs] => {
            vss.push(vs);
            vss
        }
    }

    var_spec: VarSpec {
        Id(id) => VarSpec { id: id, dims: vec![] },
        var_spec[mut vs] SLPar expr[e] SRPar => {
            vs.dims.push(e);
            vs
        }
    }

    params: Vec<FunParam> {
        => vec![],
        param[p] => vec![p],
        params[mut ps] Comma param[p] => {
            ps.push(p);
            ps
        }
    }

    param: FunParam {
        Id(id) param_dims[dims] Colon type_[t] => FunParam::new(id, t, dims)
    }

    param_dims: u32 {
        => 0,
        param_dims[dims] SLPar SRPar => dims + 1
    }

    fun_body: Vec<Stmt> {
        Begin stmts[mut ss] Return expr[e] Semicolon End => {
            ss.push(Stmt::return_value(e));
            ss
        },
        stmts[mut ss] Return expr[e] Semicolon => {
            ss.push(Stmt::return_value(e));
            ss
        }
    }

    stmt: Stmt {
        If expr[cond] Then stmt[then_stmt] Else stmt[else_stmt] => Stmt::if_then_else(
            cond, then_stmt, else_stmt
        ),
        While expr[cond] Do stmt[do_stmt] => Stmt::while_do(cond, do_stmt),
        Read loc[l] => Stmt::read(l),
        loc[l] Assign expr[e] => Stmt::assign(l, e),
        Print expr[val] => Stmt::print(val),
        CLPar block[b] CRPar => Stmt::block(b),
        Case expr[e] Of CLPar cases[cs] CRPar => Stmt::case(e, cs)
    }

    stmts: Vec<Stmt> {
        => vec![],
        stmts[mut ss] stmt[s] Semicolon => {
            ss.push(s);
            ss
        }
    }

    loc: Expr {
        Id(id) => Expr::identifier(id),
        loc[l] SLPar expr[e] SRPar => Expr::index(l, e)
    }

    case: Case {
        CId(cid) Arrow stmt[s] => Case::new(cid, vec![], s),
        CId(cid) LPar var_list[vs] RPar Arrow stmt[s] => Case::new(cid, vs, s)
    }

    cases: Vec<Case> {
        case[c] => vec![c],
        cases[mut cs] Slash case[c] => {
            cs.push(c);
            cs
        }
    }

    var_list: Vec<String> {
        Id(id) => vec![id],
        var_list[mut vs] Comma Id(id) => {
            vs.push(id);
            vs
        }
    }

    expr: Expr {
        expr[lhs] Or bint_term[rhs] => Expr::or(lhs, rhs),
        bint_term[t] => t
    }

    bint_term: Expr {
        bint_term[lhs] And bint_factor[rhs] => Expr::and(lhs, rhs),
        bint_factor[f] => f
    }

    bint_factor: Expr {
        Not bint_factor[f] => Expr::not(f),
        int_expr[lhs] Equal int_expr[rhs] => Expr::equal_to(lhs, rhs),
        int_expr[lhs] Lt int_expr[rhs] => Expr::less_than(lhs, rhs),
        int_expr[lhs] Gt int_expr[rhs] => Expr::greater_than(lhs, rhs),
        int_expr[lhs] Le int_expr[rhs] => Expr::less_than_or_equal(lhs, rhs),
        int_expr[lhs] Ge int_expr[rhs] => Expr::greater_than_or_equal(lhs, rhs),
        int_expr[e] => e
    }

    int_expr: Expr {
        int_expr[lhs] Add int_term[rhs] => Expr::add(lhs, rhs),
        int_expr[lhs] Sub int_term[rhs] => Expr::sub(lhs, rhs),
        int_term[t] => t
    }

    int_term: Expr {
        int_term[lhs] Mul int_factor[rhs] => Expr::mul(lhs, rhs),
        int_term[lhs] Div int_factor[rhs] => Expr::div(lhs, rhs),
        int_factor[f] => f
    }

    int_factor: Expr {
        LPar expr[e] RPar => e,
        Size LPar Id(id) param_dims[dims] RPar => Expr::size_of(id, dims),
        Float LPar expr[e] RPar => Expr::float(e),
        Floor LPar expr[e] RPar => Expr::floor(e),
        Ceil LPar expr[e] RPar => Expr::ceil(e),
        Id(id) LPar arg_list[args] RPar => Expr::call(Expr::identifier(id), args),
        Id(id) expr_dims[args] => {
            let mut e = Expr::identifier(id);

            for a in args {
                e = Expr::index(e, a);
            };
            e
        },
        CId(cid) => Expr::cons(cid, vec![]),
        CId(cid) LPar arg_list[args] RPar => Expr::cons(cid, args),
        IVal(i) => Expr::int(i),
        RVal(f) => Expr::real(f),
        BVal(b) => Expr::bool(b),
        CVal(c) => Expr::char(c),
        Sub int_factor[f] => Expr::negate(f)
    }

    arg_list: Vec<Expr> {
        => vec![],
        expr[e] => vec![e],
        arg_list[mut args] Comma expr[e] => {
            args.push(e);
            args
        }
    }

    expr_dims: Vec<Expr> {
        => vec![],
        expr_dims[mut args] SLPar expr[e] SRPar => {
            args.push(e);
            args
        }
    }

    cons_decls: Vec<DataCons> {
        cons_decl[c] => vec![c],
        cons_decls[mut cs] Slash cons_decl[c] => {
            cs.push(c);
            cs
        }
    }

    cons_decl: DataCons {
        CId(cid) => DataCons::new(cid, vec![]),
        CId(cid) Of cons_types[ts] => DataCons::new(cid, ts)
    }

    cons_types: Vec<Type> {
        type_[t] => vec![t],
        cons_types[mut ts] Mul type_[t] => {
            ts.push(t);
            ts
        }
    }
}

fn build_error((bad_tok, bad_span): (Token, Span), err: &'static str) -> Error {
    (format!("{}, but found {}", err, bad_tok), bad_span)
}

pub fn parse_block<'a: 'b, 'b>(tokens: &'b mut TokenStream<'a>)
    -> Result<Block, Error> {
    match _parse(tokens.iter()) {
        Result::Ok(block) => Result::Ok(block),
        Result::Err((None, err)) => Result::Err(build_error(tokens.pop(), err)),
        Result::Err((Some(bad_tok), err)) => Result::Err(build_error(bad_tok, err))
    }
}
