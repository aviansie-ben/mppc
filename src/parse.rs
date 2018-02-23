#![allow(unused_mut)]

use std::fmt;

use ast::{Block, Case, DataCons, Decl, Expr, FunParam, FunSig, Stmt, Type, VarSpec};
use lex;
use lex::{Span, Token, TokenStream};

type Error = (String, Span);

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
