#![allow(unused_mut)]

use std::collections::HashMap;

use ast::{Block, Case, DataCons, Decl, Expr, FunParam, FunSig, MultiDecl, Program, Stmt, Type, VarSpec};
use lex::{Span, Token, TokenStream};

type Error = (String, Span);

parser! {
    fn _parse(Token, Span);

    (a, b) { Span::combine(a, b) }

    program: Program {
        features[fs] block[b] => Program::new(b, fs)
    }

    feature: (String, Span) {
        Feature(f) => (f, span!())
    }

    features: HashMap<String, Span> {
        => HashMap::new(),
        features[mut fs] feature[(f, span)] => {
            fs.insert(f, span);
            fs
        }
    }

    block: Block {
        decls[ds] stmts[ss] => Block::new(ds, ss),
        decls[ds] Begin stmts[ss] End => Block::new(ds, ss)
    }

    maybe_expr: Option<Expr> {
        => None,
        expr[e] => Some(e)
    }

    id_with_span: (String, Span) {
        Id(id) => (id, span!())
    }

    cid_with_span: (String, Span) {
        CId(cid) => (cid, span!())
    }

    decls: Vec<Decl> {
        => vec![],
        decls[mut ds] decl[nd] Semicolon => {
            match nd {
                MultiDecl::Single(nd) => {
                    ds.push(nd);
                },
                MultiDecl::Multiple(mut nds) => {
                    ds.append(&mut nds);
                }
            };
            ds
        }
    }

    decl: MultiDecl {
        Var var_specs[vss] Colon type_[t] => MultiDecl::Multiple(
            vss.into_iter().map(|vs| {
                let t = t.clone().wrap_in_array(vs.dims.len() as u32);
                Decl::var(vs, t).at(span!())
            }).collect::<Vec<_>>()
        ),
        Fun id_with_span[(id, span)] LPar params[ps] RPar Colon type_[rt] CLPar decls[ds] fun_body[body] CRPar => MultiDecl::Single(Decl::func(
            id,
            FunSig::new(ps, rt),
            Block::new(ds, body)
        ).at(span)),
        Data id_with_span[(id, span)] Equal cons_decls[cs] => MultiDecl::Single(Decl::data(id, cs).at(span))
    }

    type_: Type {
        Int => Type::Int,
        Real => Type::Real,
        Bool => Type::Bool,
        Char => Type::Char,
        id_with_span[(id, span)] => Type::Id(id, span)
    }

    var_specs: Vec<VarSpec> {
        var_spec[vs] => vec![vs],
        var_specs[mut vss] Comma var_spec[vs] => {
            vss.push(vs);
            vss
        }
    }

    var_spec: VarSpec {
        id_with_span[(id, span)] => VarSpec::new(id, vec![]).at(span),
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
        id_with_span[(id, span)] param_dims[dims] Colon type_[t] => {
            FunParam::new(id, t.wrap_in_array(dims)).at(span)
        }
    }

    param_dims: u32 {
        => 0,
        param_dims[dims] SLPar SRPar => dims + 1
    }

    fun_body: Vec<Stmt> {
        Begin stmts[mut ss] End => ss,
        stmts[mut ss] => ss
    }

    stmt: Stmt {
        If expr[cond] Then stmt[then_stmt] Else stmt[else_stmt] => Stmt::if_then_else(
            cond, then_stmt, else_stmt
        ).at(span!()),
        #[no_reduce(Else)]
        If expr[cond] Then stmt[then_stmt] => Stmt::if_then(cond, then_stmt).at(span!()),
        While expr[cond] Do stmt[do_stmt] => Stmt::while_do(cond, do_stmt).at(span!()),
        Read expr[l] => Stmt::read(l).at(span!()),
        expr[l] Assign expr[e] => Stmt::assign(l, e).at(span!()),
        Print expr[val] => Stmt::print(val).at(span!()),
        CLPar block[b] CRPar => Stmt::block(b).at(span!()),
        Case expr[e] Of CLPar cases_stmt[cs] CRPar => Stmt::case(e, cs).at(span!()),
        Return expr[e] => Stmt::return_value(e).at(span!())
    }

    stmts: Vec<Stmt> {
        => vec![],
        stmts[mut ss] stmt[s] Semicolon => {
            ss.push(s);
            ss
        }
    }

    case_stmt: Case<Stmt> {
        cid_with_span[(cid, span)] Arrow stmt[s] => Case::new(cid, vec![], s).at(span),
        cid_with_span[(cid, span)] LPar var_list[vs] RPar Arrow stmt[s] => Case::new(cid, vs, s).at(span)
    }

    cases_stmt: Vec<Case<Stmt>> {
        case_stmt[c] => vec![c],
        cases_stmt[mut cs] Slash case_stmt[c] => {
            cs.push(c);
            cs
        }
    }

    var_list: Vec<(String, Span)> {
        id_with_span[(id, span)] => vec![(id, span)],
        var_list[mut vs] Comma id_with_span[(id, span)] => {
            vs.push((id, span));
            vs
        }
    }

    expr: Expr {
        expr[lhs] Or bint_term[rhs] => Expr::or(lhs, rhs).at(span!()),
        bint_term[t] => t
    }

    bint_term: Expr {
        bint_term[lhs] And bint_factor[rhs] => Expr::and(lhs, rhs).at(span!()),
        bint_factor[f] => f
    }

    bint_factor: Expr {
        Not bint_factor[f] => Expr::not(f).at(span!()),
        int_expr[lhs] Equal int_expr[rhs] => Expr::equal_to(lhs, rhs).at(span!()),
        int_expr[lhs] Lt int_expr[rhs] => Expr::less_than(lhs, rhs).at(span!()),
        int_expr[lhs] Gt int_expr[rhs] => Expr::greater_than(lhs, rhs).at(span!()),
        int_expr[lhs] Le int_expr[rhs] => Expr::less_than_or_equal(lhs, rhs).at(span!()),
        int_expr[lhs] Ge int_expr[rhs] => Expr::greater_than_or_equal(lhs, rhs).at(span!()),
        int_expr[e] => e
    }

    int_expr: Expr {
        int_expr[lhs] Add int_term[rhs] => Expr::add(lhs, rhs).at(span!()),
        int_expr[lhs] Sub int_term[rhs] => Expr::sub(lhs, rhs).at(span!()),
        int_term[t] => t
    }

    int_term: Expr {
        int_term[lhs] Mul int_factor[rhs] => Expr::mul(lhs, rhs).at(span!()),
        int_term[lhs] Div int_factor[rhs] => Expr::div(lhs, rhs).at(span!()),
        int_factor[f] => f
    }

    int_factor: Expr {
        LPar expr[e] RPar => e,
        Size LPar Id(id) param_dims[dims] RPar => Expr::size_of(id, dims).at(span!()),
        Float LPar expr[e] RPar => Expr::float(e).at(span!()),
        Floor LPar expr[e] RPar => Expr::floor(e).at(span!()),
        Ceil LPar expr[e] RPar => Expr::ceil(e).at(span!()),
        id_with_span[(id, id_span)] LPar arg_list[args] RPar => {
            Expr::call(Expr::identifier(id).at(id_span), args).at(span!())
        },
        id_with_span[(id, id_span)] expr_dims[args] => {
            let mut e = Expr::identifier(id).at(id_span);

            for (a, s) in args {
                e = Expr::index(e, a).at(s);
            };
            e
        },
        CId(cid) => Expr::cons(cid, vec![]).at(span!()),
        CId(cid) LPar arg_list[args] RPar => Expr::cons(cid, args).at(span!()),
        IVal(i) => Expr::int(i).at(span!()),
        RVal(f) => Expr::real(f).at(span!()),
        BVal(b) => Expr::bool(b).at(span!()),
        CVal(c) => Expr::char(c).at(span!()),
        Sub int_factor[f] => Expr::negate(f).at(span!()),
        LPar CLPar decls[ds] stmts[ss] maybe_expr[e] CRPar RPar => Expr::block(
            Block::new(ds, ss),
            e
        ).at(span!()),
        Case expr[e] Of CLPar cases_expr[cs] CRPar => Expr::case(e, cs).at(span!())
    }

    case_expr: Case<Expr> {
        cid_with_span[(cid, span)] Arrow expr[e] => Case::new(cid, vec![], e).at(span),
        cid_with_span[(cid, span)] LPar var_list[vs] RPar Arrow expr[e] => Case::new(cid, vs, e).at(span)
    }

    cases_expr: Vec<Case<Expr>> {
        case_expr[c] => vec![c],
        cases_expr[mut cs] Slash case_expr[c] => {
            cs.push(c);
            cs
        }
    }

    arg_list: Vec<Expr> {
        => vec![],
        expr[e] => vec![e],
        arg_list[mut args] Comma expr[e] => {
            args.push(e);
            args
        }
    }

    expr_dims: Vec<(Expr, Span)> {
        => vec![],
        expr_dims[mut args] SLPar expr[e] SRPar => {
            args.push((e, span!()));
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
        cid_with_span[(cid, span)] => DataCons::new(cid, vec![]).at(span),
        cid_with_span[(cid, span)] Of cons_types[ts] => DataCons::new(cid, ts).at(span)
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

pub fn parse_program<'a: 'b, 'b>(tokens: &'b mut TokenStream<'a>)
    -> Result<Program, Error> {
    match _parse(tokens.iter()) {
        Result::Ok(block) => Result::Ok(block),
        Result::Err((None, err)) => Result::Err(build_error(tokens.pop(), err)),
        Result::Err((Some(bad_tok), err)) => Result::Err(build_error(bad_tok, err))
    }
}
