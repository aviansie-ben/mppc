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

#[derive(Debug, Clone)]
pub struct FunSig {
    pub params: Vec<FunParam>,
    pub return_type: Type
}

#[derive(Debug, Clone)]
pub struct FunBody {
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>
}

#[derive(Debug, Clone)]
pub struct DataCons {
    pub cid: String,
    pub types: Vec<Type>
}

#[derive(Debug, Clone)]
pub enum Decl {
    Var(Vec<VarSpec>, Type),
    Fun(String, FunSig, FunBody),
    Data(String, Vec<DataCons>)
}

#[derive(Debug, Clone)]
pub enum Expr {
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
pub struct Case {
    pub cid: String,
    pub vars: Vec<String>,
    pub stmt: Stmt
}

#[derive(Debug, Clone)]
pub enum Stmt {
    IfThenElse(Expr, Box<Stmt>, Box<Stmt>),
    WhileDo(Expr, Box<Stmt>),
    Read(Expr),
    Assign(Expr, Expr),
    Print(Expr),
    Block(Block),
    Case(Expr, Vec<Case>),
    Return(Expr)
}

parser! {
    fn _parse(Token, Span);

    (a, b) { lex::combine_spans(a, b) }

    block: Block {
        decls[ds] stmts[ss] => Block { decls: ds, stmts: ss }
    }

    decls: Vec<Decl> {
        => vec![],
        decls[mut ds] decl[d] Semicolon => {
            ds.push(d);
            ds
        }
    }

    decl: Decl {
        Var var_specs[vss] Colon type_[t] => Decl::Var(vss, t),
        Fun Id(id) LPar params[ps] RPar Colon type_[rt] CLPar decls[ds] fun_body[body] CRPar => Decl::Fun(
            id,
            FunSig { params: ps, return_type: rt },
            FunBody { decls: ds, stmts: body }
        ),
        Data Id(id) Equal cons_decls[cs] => Decl::Data(id, cs)
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
        Id(id) param_dims[dims] Colon type_[t] => FunParam {
            id: id,
            val_type: t,
            dims: dims
        }
    }

    param_dims: u32 {
        => 0,
        param_dims[dims] SLPar SRPar => dims + 1
    }

    fun_body: Vec<Stmt> {
        Begin stmts[mut ss] Return expr[e] Semicolon End => {
            ss.push(Stmt::Return(e));
            ss
        },
        stmts[mut ss] Return expr[e] Semicolon => {
            ss.push(Stmt::Return(e));
            ss
        }
    }

    stmt: Stmt {
        If expr[cond] Then stmt[then_stmt] Else stmt[else_stmt] => Stmt::IfThenElse(
            cond, Box::new(then_stmt), Box::new(else_stmt)
        ),
        While expr[cond] Do stmt[do_stmt] => Stmt::WhileDo(cond, Box::new(do_stmt)),
        Read loc[l] => Stmt::Read(l),
        loc[l] Assign expr[e] => Stmt::Assign(l, e),
        Print expr[val] => Stmt::Print(val),
        CLPar block[b] CRPar => Stmt::Block(b),
        Case expr[e] Of CLPar cases[cs] CRPar => Stmt::Case(e, cs)
    }

    stmts: Vec<Stmt> {
        => vec![],
        stmts[mut ss] stmt[s] Semicolon => {
            ss.push(s);
            ss
        }
    }

    loc: Expr {
        Id(id) => Expr::Id(id),
        loc[l] SLPar expr[e] SRPar => Expr::Index(Box::new(l), Box::new(e))
    }

    case: Case {
        CId(cid) Arrow stmt[s] => Case { cid: cid, vars: vec![], stmt: s },
        CId(cid) LPar var_list[vs] RPar Arrow stmt[s] => Case { cid: cid, vars: vs, stmt: s }
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
        expr[lhs] Or bint_term[rhs] => Expr::Or(Box::new(lhs), Box::new(rhs)),
        bint_term[t] => t
    }

    bint_term: Expr {
        bint_term[lhs] And bint_factor[rhs] => Expr::And(Box::new(lhs), Box::new(rhs)),
        bint_factor[f] => f
    }

    bint_factor: Expr {
        Not bint_factor[f] => Expr::Not(Box::new(f)),
        int_expr[lhs] Equal int_expr[rhs] => Expr::Equal(Box::new(lhs), Box::new(rhs)),
        int_expr[lhs] Lt int_expr[rhs] => Expr::Lt(Box::new(lhs), Box::new(rhs)),
        int_expr[lhs] Gt int_expr[rhs] => Expr::Gt(Box::new(lhs), Box::new(rhs)),
        int_expr[lhs] Le int_expr[rhs] => Expr::Le(Box::new(lhs), Box::new(rhs)),
        int_expr[lhs] Ge int_expr[rhs] => Expr::Ge(Box::new(lhs), Box::new(rhs)),
        int_expr[e] => e
    }

    int_expr: Expr {
        int_expr[lhs] Add int_term[rhs] => Expr::Add(Box::new(lhs), Box::new(rhs)),
        int_expr[lhs] Sub int_term[rhs] => Expr::Sub(Box::new(lhs), Box::new(rhs)),
        int_term[t] => t
    }

    int_term: Expr {
        int_term[lhs] Mul int_factor[rhs] => Expr::Mul(Box::new(lhs), Box::new(rhs)),
        int_term[lhs] Div int_factor[rhs] => Expr::Div(Box::new(lhs), Box::new(rhs)),
        int_factor[f] => f
    }

    int_factor: Expr {
        LPar expr[e] RPar => e,
        Size LPar Id(id) param_dims[dims] RPar => Expr::Size(id, dims),
        Float LPar expr[e] RPar => Expr::Float(Box::new(e)),
        Floor LPar expr[e] RPar => Expr::Floor(Box::new(e)),
        Ceil LPar expr[e] RPar => Expr::Ceil(Box::new(e)),
        Id(id) LPar arg_list[args] RPar => Expr::Call(Box::new(Expr::Id(id)), args),
        Id(id) expr_dims[args] => {
            let mut e = Expr::Id(id);

            for a in args {
                e = Expr::Index(Box::new(e), Box::new(a));
            };
            e
        },
        CId(cid) => Expr::Cons(cid, vec![]),
        CId(cid) LPar arg_list[args] RPar => Expr::Cons(cid, args),
        IVal(i) => Expr::Int(i),
        RVal(f) => Expr::Real(f),
        BVal(b) => Expr::Bool(b),
        CVal(c) => Expr::Char(c),
        Sub int_factor[f] => Expr::Neg(Box::new(f))
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
        CId(cid) => DataCons { cid: cid, types: vec![] },
        CId(cid) Of cons_types[ts] => DataCons { cid: cid, types: ts }
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
