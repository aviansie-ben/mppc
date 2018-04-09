use std::cell::{RefCell, UnsafeCell};
use std::collections::{HashMap, HashSet};
use std::mem;
use std::rc::Rc;

use ast;
use lex::Span;
use symbol::*;

lazy_static! {
    static ref VALID_FEATURES: HashSet<&'static str> = {
        let mut fs = HashSet::new();

        fs.insert("return_anywhere");
        fs.insert("optional_else");
        fs.insert("block_expr");
        fs.insert("case_expr");

        fs
    };
}

macro_rules! name_already_defined {
    ($name:expr, $span:expr, $old_sym:expr) => ((
        format!(
            "the name '{}' has already been defined in this scope (original definition is at line {}, col {})",
            $name,
            $old_sym.span.lo.line,
            $old_sym.span.lo.col
        ),
        $span
    ))
}

macro_rules! expect_name_not_defined {
    ($ctx:expr, $symbols:expr, $name:expr, $span:expr) => {{
        if let Some(old_sym) = $symbols.find_imm_named_symbol($name) {
            $ctx.push_error(name_already_defined!($name, $span, $ctx.sdt.get_symbol(old_sym)));
            false
        } else {
            true
        }
    }}
}

macro_rules! variable_not_defined {
    ($name:expr, $span:expr) => ((
        format!("no variable '{}' exists in this scope", $name),
        $span
    ))
}

macro_rules! constructor_already_defined {
    ($name:expr, $span:expr, $old_span:expr) => ((
        format!(
            "constructor #{} redefined (original definition is at line {}, col {})",
            $name,
            $old_span.lo.line,
            $old_span.lo.col
        ),
        $span
    ))
}

macro_rules! constructor_not_defined {
    ($name:expr, $span:expr) => ((
        format!("no constructor #{} has been defined", $name),
        $span
    ));

    ($name:expr, $span:expr, $type_name:expr) => ((
        format!("type {} does not have a constructor #{}", $type_name, $name),
        $span
    ))
}

macro_rules! function_definition_conflict {
    ($decl:expr, $old_sym:expr) => ((
        format!(
            "this function conflicts with a previous definition (original definition is at line {}, col {})",
            $old_sym.span.lo.line,
            $old_sym.span.lo.col
        ),
        $decl.span
    ))
}

macro_rules! type_already_defined {
    ($name:expr, $span:expr, $old_span:expr) => ((
        format!(
            "a type '{}' already exists in this scope (original definition is at line {}, col {})",
            $name,
            $old_span.lo.line,
            $old_span.lo.col
        ),
        $span
    ))
}

macro_rules! type_not_defined {
    ($name:expr, $span:expr) => ((
        format!("no type {} has been defined", $name),
        $span
    ))
}

macro_rules! wrong_number_of_args {
    ($actual_number:expr, $expected_number:expr, $span:expr) => ((
        format!(
            "wrong number of arguments: expected {}, but found {}",
            $expected_number,
            $actual_number
        ),
        $span
    ))
}

macro_rules! cannot_convert {
    ($ctx:expr, $expr:expr, $expected_type:expr) => ((
        format!(
            "cannot convert from {} to {}",
            $expr.val_type.pretty($ctx.tdt),
            $expected_type.pretty($ctx.tdt)
        ),
        $expr.span
    ))
}

macro_rules! expect_convert_exact {
    ($ctx:expr, $expr:expr, $expected_type:expr) => {{
        if !$expr.val_type.can_convert_to_exact($expected_type) {
            $ctx.push_error(cannot_convert!($ctx, $expr, $expected_type));
            false
        } else {
            true
        }
    }}
}

macro_rules! cannot_return_outside_func {
    ($span:expr) => ((
        "cannot return outside function".to_string(),
        $span
    ))
}

macro_rules! cannot_pattern_match {
    ($ctx:expr, $expr:expr) => ((
        format!("cannot pattern match on a value of type {}", $expr.val_type.pretty($ctx.tdt)),
        $expr.span
    ))
}

macro_rules! cannot_read {
    ($ctx:expr, $expr:expr) => ((
        format!("cannot read a value of type {}", $expr.val_type.pretty($ctx.tdt)),
        $expr.span
    ))
}

macro_rules! cannot_print {
    ($ctx:expr, $expr:expr) => ((
        format!("cannot print a value of type {}", $expr.val_type.pretty($ctx.tdt)),
        $expr.span
    ))
}

macro_rules! no_such_operator {
    ($ctx:expr, $op:expr, $val:expr, $span:expr) => ((
        format!("no operator {} exists for {}", $op, $val.val_type.pretty($ctx.tdt)),
        $span
    ));

    ($ctx:expr, $op:expr, $val1:expr, $val2:expr, $span:expr) => ((
        format!(
            "no operator {} exists for {} and {}",
            $op,
            $val1.val_type.pretty($ctx.tdt),
            $val2.val_type.pretty($ctx.tdt)
        ),
        $span
    ))
}

macro_rules! cannot_get_size {
    ($ctx:expr, $val:expr, $span:expr) => ((
        format!("cannot use size operator on value of type {}", $val.val_type.pretty($ctx.tdt)),
        $span
    ));

    ($ctx:expr, $val:expr, $dim:expr, $span:expr) => ((
        format!(
            "cannot get the size of dimension {} of a value of type {}",
            $dim,
            $val.val_type.pretty($ctx.tdt)
        ),
        $span
    ))
}

macro_rules! cannot_call {
    ($ctx:expr, $func:expr, $span:expr) => ((
        format!("cannot call expression of type {}", $func.val_type.pretty($ctx.tdt)),
        $span
    ))
}

macro_rules! cannot_index {
    ($ctx:expr, $val:expr, $span:expr) => ((
        format!("cannot index into value of type {}", $val.val_type.pretty($ctx.tdt)),
        $span
    ))
}

macro_rules! cannot_assign {
    ($val:expr) => ((
        "this expression cannot be assigned to".to_string(),
        $val.span
    ))
}

// TODO Better error message
macro_rules! no_function_match {
    ($ctx:expr, $params:expr, $defs:expr, $span:expr) => ((
        "none of these functions matches the given arguments".to_string(),
        $span
    ))
}

// TODO Better error message
macro_rules! no_constructor_match {
    ($ctx:expr, $name:expr, $params:expr, $defs:expr, $span:expr) => ((
        format!("no constructor #{} in this scope matches the given arguments", $name),
        $span
    ))
}

macro_rules! cannot_return_except_at_end {
    ($span:expr) => ((
        "return statements cannot appear here (did you mean to add `@feature(return_anywhere)'?)".to_string(),
        $span
    ))
}

macro_rules! missing_return_at_end {
    ($span:expr) => ((
        "this function does not end with a return statement (did you mean to add `@feature(return_anywhere)'?)".to_string(),
        $span
    ))
}

macro_rules! missing_return_anywhere {
    ($span:expr) => ((
        "one or more code paths do not return a value".to_string(),
        $span
    ))
}

macro_rules! duplicate_case {
    ($name:expr, $span:expr, $old_span:expr) => ((
        format!(
            "constructor #{} is already covered by a previous branch (at line {}, col {})",
            $name,
            $old_span.lo.line,
            $old_span.lo.col
        ),
        $span
    ))
}

macro_rules! missing_else {
    ($span:expr) => ((
        format!("this if statement is missing an else branch (did you mean to add `@feature(optional_else)'?)"),
        $span
    ))
}

macro_rules! block_expr_disabled {
    ($span:expr) => ((
        format!("block expression support is not enabled (did you mean to add `@feature(block_expr)'?)"),
        $span
    ))
}

macro_rules! block_expr_has_no_value {
    ($span:expr) => ((
        "this block expression has no value and does not always return".to_string(),
        $span
    ))
}

macro_rules! case_expr_disabled {
    ($span:expr) => ((
        format!("case expression support is not enabled (did you mean to add `@feature(case_expr)'?)"),
        $span
    ))
}

macro_rules! case_expr_not_exhaustive {
    ($span:expr) => ((
        format!("case expressions must be exhaustive over the type they match"),
        $span
    ))
}

struct AnalysisContext<'a, 'b> where 'b: 'a {
    function_id: usize,
    tdt: &'a TypeDefinitionTable,
    sdt: &'a SymbolDefinitionTable,
    features: &'a HashMap<String, Span>,
    expected_return: Option<&'a Type>,
    errors: UnsafeCell<&'b mut Vec<(String, Span)>>
}

impl <'a, 'b> AnalysisContext<'a, 'b> {
    fn new(
        function_id: usize,
        tdt: &'a TypeDefinitionTable,
        sdt: &'a SymbolDefinitionTable,
        features: &'a HashMap<String, Span>,
        expected_return: Option<&'a Type>,
        errors: &'b mut Vec<(String, Span)>
    ) -> AnalysisContext<'a, 'b> {
        AnalysisContext {
            function_id: function_id,
            tdt: tdt,
            sdt: sdt,
            features: features,
            expected_return: expected_return,
            errors: UnsafeCell::new(errors)
        }
    }

    fn inside_function<'c>(
        &'c self,
        function_id: usize,
        expected_return: Option<&'c Type>
    ) -> AnalysisContext<'c, 'b> {
        AnalysisContext {
            function_id: function_id,
            tdt: self.tdt,
            sdt: self.sdt,
            features: self.features,
            expected_return: expected_return,
            errors: UnsafeCell::new(unsafe { *self.errors.get() }),
        }
    }

    fn push_error(&self, error: (String, Span)) -> () {
        unsafe { &mut *self.errors.get() }.push(error);
    }
}

fn resolve_type(
    t: &ast::Type,
    ctx: &AnalysisContext,
    symbols: &SymbolTable,
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
                ctx.push_error(type_not_defined!(name, *span));
                Type::Error
            }
        },
        ast::Type::Array(ref inner_type, ref dims) => Type::Array(
            Box::new(resolve_type(inner_type, ctx, symbols)),
            *dims
        )
    }
}

fn create_symbol_for_decl(
    decl: ast::Decl,
    ctx: &AnalysisContext,
    symbols: &mut SymbolTable
) {
    match decl.node {
        ast::DeclType::Data(name, type_id, ctors) => {
            let mut type_def = DataTypeDefinition {
                name: name,
                ctors: ctors.into_iter().map(|ctor| {
                    DataTypeCtor {
                        name: ctor.cid.to_string(),
                        args: ctor.types.iter()
                            .map(|t| resolve_type(t, ctx, symbols))
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
                        ctx.push_error(constructor_already_defined!(c1.name, c1.span, c2.span));
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

            // We don't need to perform error-checking when defining the type, since that is already
            // handled when we created the dummy definitions for the data types earlier.
            ctx.tdt.define_type(type_id, TypeDefinition::Data(type_def));
        },
        ast::DeclType::Fun(name, sig, body) => {
            let params: Vec<_> = sig.params.iter()
                .map(|p| resolve_type(&p.val_type, ctx, symbols))
                .collect();
            let return_type = resolve_type(&sig.return_type, ctx, symbols);
            let fn_type = ctx.tdt.get_function_type(&params, &return_type);

            let define = if let Some(old_sym) = symbols.find_imm_named_symbol(&name) {
                fn does_sig_conflict(
                    tdt: &TypeDefinitionTable,
                    new_params: &[Type],
                    old_sig: usize
                ) -> bool {
                    let old_params = if let TypeDefinition::Function(ref td) = tdt.get_definition(old_sig) {
                        &td.params
                    } else {
                        unreachable!()
                    };

                    if new_params.len() == old_params.len() {
                        new_params.iter().zip(old_params.iter())
                            .all(|(t1, t2)| t1 == t2 && t1 != &Type::Error)
                    } else {
                        false
                    }
                }

                // We need to examine the previously defined symbol in detail to determine whether
                // or not this new declaration conflicts with the old one. This is because two
                // functions can have the same name as long as their signatures (except the return
                // values) are different.
                let old_sym = ctx.sdt.get_symbol(old_sym);
                let conflict_sym = match old_sym.node {
                    SymbolType::Fun(ref f) => if does_sig_conflict(ctx.tdt, &params, f.sig) {
                        Some(old_sym)
                    } else {
                        None
                    },
                    SymbolType::MultiFun(ref old_sym) => {
                        old_sym.funcs.borrow().iter()
                            .find(|&&(sig, _)| does_sig_conflict(ctx.tdt, &params, sig))
                            .map(|&(_, sym_id)| ctx.sdt.get_symbol(sym_id))
                    },
                    _ => Some(old_sym)
                };

                // If a conflicting symbol was found, determine the correct error message to print
                // depending on whether the previous definition was a function or something else.
                if let Some(conflict_sym) = conflict_sym {
                    if let SymbolType::Fun(_) = conflict_sym.node {
                        ctx.push_error(function_definition_conflict!(decl, conflict_sym));
                    } else {
                        ctx.push_error(name_already_defined!(name, decl.span, conflict_sym));
                    };
                    false
                } else {
                    true
                }
            } else {
                true
            };

            // Pre-create symbols for the function parameters. These will be added to the correct
            // symbol tables alongside the symbol for the new function.
            let params: Vec<_> = sig.params.into_iter().map(|p| Symbol::new(
                0,
                p.id,
                p.span,
                SymbolType::Param(ParamSymbol {
                    val_type: resolve_type(&p.val_type, ctx, symbols)
                }),
                ctx.function_id
            )).collect();

            // Only actually add the symbol if there were no conflicts. If there were conflicts, the
            // function declaration is type-checked but no symbol is actually added.
            if define {
                ctx.sdt.add_named_function_symbol(symbols, Symbol::new(
                    0,
                    name.to_string(),
                    decl.span,
                    SymbolType::Fun(FunSymbol {
                        sig: fn_type,
                        params: vec![],
                        body: RefCell::new(body)
                    }),
                    ctx.function_id
                ), params.into_iter());
            };
        },
        ast::DeclType::Var(spec, val_type) => {
            let val_type = resolve_type(&val_type, ctx, symbols);

            if expect_name_not_defined!(ctx, symbols, &spec.id, spec.span) {
                ctx.sdt.add_named_symbol(symbols, Symbol::new(
                    0,
                    spec.id,
                    spec.span,
                    SymbolType::Var(VarSymbol {
                        val_type: val_type,
                        dims: RefCell::new(spec.dims)
                    }),
                    ctx.function_id
                ));
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
    ctx: &AnalysisContext,
    symbols: &Rc<RefCell<SymbolTable>>
) {
    if params.len() != expected_types.len() {
        ctx.push_error(wrong_number_of_args!(params.len(), expected_types.len(), *span));
    };

    for (et, param) in expected_types.iter().zip(params.iter_mut()) {
        analyze_expression(param, ctx, symbols, Some(et));
        expect_convert_exact!(ctx, param, et);
    };
}

fn analyze_cases<T, U: Fn (&mut T) -> &Rc<RefCell<SymbolTable>>>(
    val: &mut ast::Expr,
    cases: &mut [ast::Case<T>],
    get_symbols: U,
    ctx: &AnalysisContext,
    symbols: &Rc<RefCell<SymbolTable>>
) -> () {
    fn analyze_case_error<T, U: Fn (&mut T) -> &Rc<RefCell<SymbolTable>>>(
        case: &mut ast::Case<T>,
        get_symbols: &U,
        ctx: &AnalysisContext,
        symbols: &Rc<RefCell<SymbolTable>>
    ) {
        // Something went wrong during type-checking. However, we still want the variables to be
        // declared on the branch to avoid further errors when examining the branch.
        let sub_symbols = &mut get_symbols(&mut case.branch).borrow_mut();
        sub_symbols.set_parent(symbols.clone());

        for (name, span) in case.vars.drain(..) {
            if expect_name_not_defined!(ctx, sub_symbols, &name, span) {
                case.var_bindings.push(ctx.sdt.add_named_symbol(sub_symbols, Symbol::new(
                    0,
                    name,
                    span,
                    SymbolType::Var(VarSymbol {
                        val_type: Type::Error,
                        dims: RefCell::new(vec![])
                    }),
                    ctx.function_id
                )));
            } else {
                case.var_bindings.push(!0);
            };
        };
    }

    fn analyze_case<T, U: Fn (&mut T) -> &Rc<RefCell<SymbolTable>>>(
        case: &mut ast::Case<T>,
        typedef: &DataTypeDefinition,
        get_symbols: &U,
        ctx: &AnalysisContext,
        symbols: &Rc<RefCell<SymbolTable>>
    ) {
        // Begin by trying to find the named constructor and doing some very basic validation. If
        // anything fails, fall back to analyze_case_error.
        let (ctor_id, ctor) = if let Some(ctor) = typedef.ctors.iter().enumerate().find(|&(_, ctor)| ctor.name == case.cid) {
            ctor
        } else {
            ctx.push_error(constructor_not_defined!(case.cid, case.span, typedef.name));

            analyze_case_error(case, get_symbols, ctx, symbols);
            return;
        };

        case.ctor_id = ctor_id;

        if ctor.args.len() != case.vars.len() {
            ctx.push_error(wrong_number_of_args!(case.vars.len(), ctor.args.len(), case.span));

            analyze_case_error(case, get_symbols, ctx, symbols);
            return;
        };

        // Now that we know what types each of the variables we've matched should be, we can go
        // through and declare them on the branch's symbol table.
        let sub_symbols = &mut get_symbols(&mut case.branch).borrow_mut();
        sub_symbols.set_parent(symbols.clone());

        for ((name, span), val_type) in case.vars.drain(..).zip(ctor.args.iter()) {
            if expect_name_not_defined!(ctx, sub_symbols, &name, span) {
                case.var_bindings.push(ctx.sdt.add_named_symbol(sub_symbols, Symbol::new(
                    0,
                    name,
                    span,
                    SymbolType::Var(VarSymbol {
                        val_type: val_type.clone(),
                        dims: RefCell::new(vec![])
                    }),
                    ctx.function_id
                )));
            } else {
                case.var_bindings.push(!0);
            };
        };
    }

    // Begin by examining the value we'll be pattern matching against. Then, try to unwrap that type
    // into its underlying data type definition, which we'll need to validate the constructors.
    let val_type = analyze_expression(val, ctx, symbols, None);
    let typedef = if let Type::Defined(ref type_id) = val_type {
        if let TypeDefinition::Data(ref typedef) = ctx.tdt.get_definition(*type_id) {
            Some(typedef)
        } else {
            None
        }
    } else {
        None
    };

    // Check that the type we found was a data type and then go through, analyzing each of the
    // branches and checking for any duplicates.
    if let Some(typedef) = typedef {
        let mut handled_cases: Vec<(usize, Span)> = Vec::new();
        for case in cases.iter_mut() {
            analyze_case(case, typedef, &get_symbols, ctx, symbols);

            if let Some((_, prev_span)) = handled_cases.iter().find(|&&(id, _)| id == case.ctor_id) {
                ctx.push_error(duplicate_case!(case.cid, case.span, prev_span));
            } else {
                handled_cases.push((case.ctor_id, case.span));
            };
        };
    } else {
        ctx.push_error(cannot_pattern_match!(ctx, val));

        for case in cases.iter_mut() {
            analyze_case_error(case, &get_symbols, ctx, symbols);
        };
    };
}

fn get_expression_symbols(
    expr: &mut ast::Expr
) -> &Rc<RefCell<SymbolTable>> {
    // Sometimes, we need to be able to add symbols to an arbitrary expression. To do this, we need
    // to turn the expression into a block expression if it wasn't already. We also mark it as
    // "synthetic" to avoid the analyzer complaining about a missing @feature(block_expr) annotation
    // later.
    if let ast::ExprType::Block(ref block, _) = expr.node {
        return &block.symbols;
    };

    let span = expr.span;
    let old_expr = mem::replace(expr, ast::Expr::block(ast::Block::new(vec![], vec![]), None).at(span));
    expr.synthetic = true;
    if let ast::ExprType::Block(ref block, ref mut result) = expr.node {
        *result = Some(Box::new(old_expr));
        &block.symbols
    } else {
        unreachable!();
    }
}

fn do_analyze_expression(
    expr: &mut ast::Expr,
    ctx: &AnalysisContext,
    symbols: &Rc<RefCell<SymbolTable>>,
    expected_type: Option<&Type>
) -> Type {
    match expr.node {
        ast::ExprType::BinaryOp(op, ref mut lhs, ref mut rhs, ref mut sym_op) => {
            let val_types = [
                analyze_expression(lhs, ctx, symbols, None),
                analyze_expression(rhs, ctx, symbols, None)
            ];

            if val_types[0] == Type::Error || val_types[1] == Type::Error {
                return Type::Error;
            };

            // Look through the available binary operator implementations and try to find one which
            // is applicable to the given types.
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

            // If a valid implementation was found, complete the analysis of the sub-expressions to
            // resolve any ambiguous references.
            if let Some(&(ref params, ref result, op)) = op_impl {
                analyze_expression(lhs, ctx, symbols, Some(&params[0]));
                analyze_expression(rhs, ctx, symbols, Some(&params[1]));
                *sym_op = op;

                return result.clone();
            } else {
                ctx.push_error(no_such_operator!(ctx, op, lhs, rhs, expr.span));
                return Type::Error;
            };
        },
        ast::ExprType::UnaryOp(op, ref mut val, ref mut sym_op) => {
            let val_types = [analyze_expression(val, ctx, symbols, None)];

            if val_types[0] == Type::Error {
                return Type::Error;
            };

            // Look through the available unary operator implementations and try to find one which
            // is applicable to the given type.
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

            // If a valid implementation was found, complete the analysis of the sub-expression to
            // resolve any ambiguous references.
            if let Some(&(ref params, ref result, op)) = op_impl {
                analyze_expression(val, ctx, symbols, Some(&params[0]));
                *sym_op = op;

                return result.clone();
            } else {
                ctx.push_error(no_such_operator!(ctx, op, val, expr.span));
                return Type::Error;
            };
        },
        ast::ExprType::Size(ref mut val, dim) => {
            let val_type = analyze_expression(val, ctx, symbols, None);

            if let Type::Array(_, val_dims) = val_type {
                if dim >= val_dims {
                    ctx.push_error(cannot_get_size!(ctx, val, dim, expr.span));
                };
            } else {
                ctx.push_error(cannot_get_size!(ctx, val, expr.span));
            };
            return Type::Int;
        },
        ast::ExprType::Id(ref name, ref mut sym_id) => {
            fn resolve_symbol<'a>(
                ctx: &'a AnalysisContext,
                symbols: &SymbolTable,
                name: &str,
                expected_type: Option<&Type>
            ) -> Option<&'a Symbol> {
                // First, try to simply look up the symbol in the symbol table.
                if let Some(sym) = symbols.find_named_symbol(name) {
                    let sym = ctx.sdt.get_symbol(sym);

                    // If the symbol was a multi-function and we know what type of function is
                    // expected, resolve the reference. Otherwise, keep the reference ambiguous and
                    // let a higher-level analysis routine take care of resolving the ambiguity and
                    // calling this routine again with an expected type.
                    if let SymbolType::MultiFun(ref mf) = sym.node {
                        if let Some(&Type::Defined(expected_type)) = expected_type {
                            mf.funcs.borrow().iter()
                                .find(|&&(type_id, _)| type_id == expected_type)
                                .map(|&(_, sym_id)| ctx.sdt.get_symbol(sym_id))
                        } else {
                            Some(sym)
                        }
                    } else {
                        Some(sym)
                    }
                } else {
                    None
                }
            }

            let symbols_borrow = &symbols.borrow();
            if let Some(sym) = resolve_symbol(ctx, symbols_borrow, name, expected_type) {
                // If the symbol is not local to the current function, set a flag saying the
                // variable has references from outside of the function in which it is declared.
                // This is used to suppress unsafe optimizations on these symbols.
                if sym.defining_fun != ctx.function_id {
                    sym.has_nonlocal_references.set(true);
                };

                expr.assignable = sym.is_assignable();
                *sym_id = sym.id;
                return sym.val_type();
            } else {
                expr.assignable = true;
                ctx.push_error(variable_not_defined!(name, expr.span));
                return Type::Error
            };
        },
        ast::ExprType::Call(ref mut func, ref mut params) => {
            fn get_function_typedefs<'a>(
                t: &Type,
                tdt: &'a TypeDefinitionTable
            ) -> Vec<(usize, &'a FunctionTypeDefinition)> {
                // Look through the given type and unwrap all available function signatures. If
                // multiple are returned, the caller will need to perform disambiguation and then
                // re-analyze the function expression.
                match *t {
                    Type::Defined(type_id) => {
                        if let TypeDefinition::Function(ref typedef) = tdt.get_definition(type_id) {
                            vec![(type_id, typedef)]
                        } else {
                            vec![]
                        }
                    },
                    Type::Unresolved(ref types) => {
                        types.iter().filter_map(|t| {
                            if let Type::Defined(type_id) = *t {
                                if let TypeDefinition::Function(ref typedef) = tdt.get_definition(type_id) {
                                    Some((type_id, typedef))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }).collect()
                    },
                    _ => vec![]
                }
            }

            // First, perform a loose analysis of the function and arguments to determine which
            // types they could potentially be.
            let func_type = analyze_expression(func, ctx, symbols, None);

            if expr.val_type == Type::Unknown {
                for param in params.iter_mut() {
                    analyze_expression(param, ctx, symbols, None);
                };
            };

            let typedefs = get_function_typedefs(&func_type, ctx.tdt);

            if typedefs.len() == 0 {
                // If the number of valid call signatures was 0, then the provided expression cannot
                // be called.
                ctx.push_error(cannot_call!(ctx, func, expr.span));
                return Type::Error;
            } else if typedefs.len() == 1 {
                // If there is only one valid call signature, just go through the analysis of the
                // arguments assuming that's the one we want.
                analyze_call_signature(&expr.span, &typedefs[0].1.params, params, ctx, symbols);
                return typedefs[0].1.return_type.clone();
            } else {
                // If there are multiple valid call signatures, we need to disambiguate which of
                // them should be used for this call.
                let param_types: Vec<_> = params.iter_mut()
                    .map(|p| analyze_expression(p, ctx, symbols, None))
                    .collect();
                let valid_typedefs = get_valid_call_signatures(
                    &param_types[..],
                    typedefs.iter().map(|&(_, td)| &td.params[..])
                );

                // Now, we go through the same analysis as above again, only looking at the set of
                // call signatures which matched the arguments.
                if valid_typedefs.len() == 0 {
                    ctx.push_error(no_function_match!(ctx, params, typedefs, expr.span));
                    return Type::Error;
                } else if valid_typedefs.len() == 1 {
                    analyze_expression(
                        func,
                        ctx,
                        symbols,
                        Some(&Type::Defined(typedefs[valid_typedefs[0]].0))
                    );
                    analyze_call_signature(
                        &expr.span,
                        &typedefs[valid_typedefs[0]].1.params,
                        params,
                        ctx,
                        symbols
                    );
                    return typedefs[valid_typedefs[0]].1.return_type.clone();
                } else {
                    return Type::union(
                        valid_typedefs.iter().map(|&i| typedefs[i].1.return_type.clone())
                    );
                };
            };
        },
        ast::ExprType::Index(ref mut val, ref mut index) => {
            let val_type = analyze_expression(val, ctx, symbols, None);

            analyze_expression(index, ctx, symbols, Some(&Type::Int));
            expect_convert_exact!(ctx, index, &Type::Int);

            if let Type::Array(inner_type, dims) = val_type {
                // We can only assign values to an array once we have fully dereferenced it, so set
                // the assignability of the expression and the type accordingly.
                return if dims == 1 {
                    expr.assignable = true;
                    *inner_type
                } else {
                    Type::Array(inner_type, dims - 1)
                };
            } else {
                expr.assignable = true;
                ctx.push_error(cannot_index!(ctx, val, expr.span));
                return Type::Error;
            };
        },
        ast::ExprType::Cons(ref name, ref mut params, ref mut expr_ctor_id) => {
            if expr.val_type == Type::Unknown {
                for param in params.iter_mut() {
                    analyze_expression(param, ctx, symbols, None);
                };
            };

            // First, go through and find the list of constructors with the given name
            let symbols_borrow = &symbols.borrow();
            let ctors = if let Some(ctors) = symbols_borrow.find_ctors(name) {
                ctors
            } else {
                ctx.push_error(constructor_not_defined!(name, expr.span));
                return Type::Error;
            };

            if ctors.len() == 1 {
                // If there was only one, analyze assuming that's the one we want to use.
                let ctor_id = ctors[0];
                let ctor = match ctx.tdt.get_definition(ctor_id.0) {
                    TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                    _ => panic!("invalid data type")
                };

                analyze_call_signature(&expr.span, &ctor.args, params, ctx, symbols);
                *expr_ctor_id = ctor_id.1;
                return Type::Defined(ctor_id.0);
            } else {
                // If there are multiple, but we know what the expected type of this expression was,
                // then analyze using the correct one.
                if let Some(Type::Defined(expected_type)) = expected_type {
                    let ctor_id = ctors.iter().find(|ctor_id| &ctor_id.0 == expected_type);

                    if let Some(ctor_id) = ctor_id {
                        let ctor = match ctx.tdt.get_definition(ctor_id.0) {
                            TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                            _ => panic!("invalid data type")
                        };

                        analyze_call_signature(&expr.span, &ctor.args, params, ctx, symbols);
                        *expr_ctor_id = ctor_id.1;
                        return Type::Defined(ctor_id.0);
                    };
                };

                // Otherwise, examine the list of constructors to see which ones match the types of
                // the number of arguments we have.
                let valid_ctors: Vec<_> = ctors.iter().map(|ctor_id| {
                    let ctor = match ctx.tdt.get_definition(ctor_id.0) {
                        TypeDefinition::Data(ref td) => &td.ctors[ctor_id.1],
                        _ => panic!("invalid data type")
                    };

                    (ctor, *ctor_id)
                }).filter(|(ctor, _)| ctor.args.len() == params.len()).collect();

                // If there is only 1 valid constructor, or no valid constructors at all, we can
                // stop disambiguation early.
                if valid_ctors.len() == 0 {
                    let param_types: Vec<_> = params.iter().map(|p| p.val_type.clone()).collect();

                    ctx.push_error(no_constructor_match!(ctx, name, param_types, ctors, expr.span));
                    return Type::Error;
                } else if valid_ctors.len() == 1 {
                    let (ctor, ctor_id) = valid_ctors[0];

                    analyze_call_signature(&expr.span, &ctor.args, params, ctx, symbols);
                    *expr_ctor_id = ctor_id.1;
                    return Type::Defined(ctor_id.0);
                }

                // Otherwise, we look at the types of the arguments to determine which constructor
                // should be used.
                let param_types: Vec<_> = params.iter().map(|p| p.val_type.clone()).collect();
                let valid_ctors: Vec<_> = valid_ctors.iter().filter(|&&(ctor, _)| {
                    for (et, at) in ctor.args.iter().zip(param_types.iter()) {
                        if !at.can_convert_to(et) {
                            return false;
                        };
                    };
                    true
                }).collect();

                if valid_ctors.len() == 0 {
                    ctx.push_error(no_constructor_match!(ctx, name, param_types, ctors, expr.span));
                    return Type::Error;
                } else if valid_ctors.len() == 1 {
                    let (ctor, ctor_id) = valid_ctors[0];
                    analyze_call_signature(&expr.span, &ctor.args, params, ctx, symbols);
                    *expr_ctor_id = ctor_id.1;
                    return Type::Defined(ctor_id.0);
                } else {
                    // If we still can't determine which constructor to use, then it's up to the
                    // higher-level analyzer to tell us what type it's expecting.
                    return Type::union(valid_ctors.into_iter().map(|&(_, (type_id, _))| {
                        Type::Defined(type_id)
                    }));
                };
            };
        },
        ast::ExprType::Int(_) => return Type::Int,
        ast::ExprType::Real(_) => return Type::Real,
        ast::ExprType::Bool(_) => return Type::Bool,
        ast::ExprType::Char(_) => return Type::Char,
        ast::ExprType::Block(ref mut block, ref mut result) => {
            // Contrary to the way that expressions work, examining blocks multiple times can have
            // bad consequences. A good way to check if this is the first time we're examining this
            // expression is to check whether the type is Type::Unknown. The type should only ever
            // have that value on the first analysis.
            if expr.val_type == Type::Unknown {
                if !expr.synthetic && !ctx.features.contains_key("block_expr") {
                    ctx.push_error(block_expr_disabled!(expr.span));
                };

                {
                    let sub_symbols = &mut block.symbols.borrow_mut();

                    if sub_symbols.parent.is_none() {
                        sub_symbols.set_parent(symbols.clone());
                    }
                };

                populate_block_symbol_table(block, ctx);
            };

            // Now that the block has been examined, we can just pass analysis through to the
            // expression we want to use as the result.
            if let Some(result) = result {
                return analyze_expression(
                    result,
                    ctx,
                    &block.symbols,
                    expected_type
                );
            } else if block.stmts.iter().any(|s| s.will_return) {
                return Type::Never;
            } else {
                ctx.push_error(block_expr_has_no_value!(expr.span));
                return Type::Error;
            };
        },
        ast::ExprType::Case(ref mut val, ref mut cases) => {
            // Again, we only want to examine the actual case expression itself once, so we check
            // whether we're performing the first analysis here.
            if expr.val_type == Type::Unknown {
                if !expr.synthetic && !ctx.features.contains_key("case_expr") {
                    ctx.push_error(case_expr_disabled!(expr.span));
                };

                analyze_cases(val, cases, get_expression_symbols, ctx, symbols);

                if !val.val_type.are_cases_exhaustive(ctx.tdt, cases) {
                    ctx.push_error(case_expr_not_exhaustive!(expr.span));
                };
            };

            // Once the cases have been examined, we want to examine the expressions on each of the
            // branches and try to find the type to which they should all be converted. If we know
            // what type this expression should be, we can just use that straight away. Otherwise,
            // we need to examine the possible types and find their least upper bound.
            let result_type = if let Some(expected_type) = expected_type {
                expected_type.clone()
            } else {
                let mut result_type = Type::Never;

                for c in cases.iter_mut() {
                    let next_result_type = analyze_expression(
                        &mut c.branch,
                        ctx,
                        symbols,
                        None
                    );

                    if let Some(next_result_type) = Type::least_upper_bound(&result_type, &next_result_type) {
                        result_type = next_result_type;
                    };
                };

                let next_result_type = if let Type::Unresolved(ref mut types) = result_type {
                    if types.len() == 1 {
                        Some(types.remove(0))
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(next_result_type) = next_result_type {
                    next_result_type
                } else {
                    result_type
                }
            };

            // If we found a valid type to convert everything to, go through and finalize the
            // analysis of each of the branches.
            if result_type.is_resolved() {
                for c in cases.iter_mut() {
                    analyze_expression(
                        &mut c.branch,
                        ctx,
                        symbols,
                        Some(&result_type)
                    );

                    expect_convert_exact!(ctx, c.branch, &result_type);
                };
            };

            return result_type;
        }
    };
}

fn analyze_expression(
    expr: &mut ast::Expr,
    ctx: &AnalysisContext,
    symbols: &Rc<RefCell<SymbolTable>>,
    expected_type: Option<&Type>
) -> Type {
    if expr.val_type.is_resolved() {
        return expr.val_type.clone();
    };

    expr.val_type = do_analyze_expression(expr, ctx, symbols, expected_type);
    expr.val_type.clone()
}

fn get_statement_symbols(
    stmt: &mut ast::Stmt
) -> &Rc<RefCell<SymbolTable>> {
    // Sometimes, we need to add symbols to an arbitrary statement. However, only block statements
    // have an associated symbol table. Thus, we need to turn any non-block statements into block
    // statements so that we can access their symbol table.

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
    ctx: &AnalysisContext,
    symbols: &Rc<RefCell<SymbolTable>>
) {
    match stmt.node {
        ast::StmtType::IfThenElse(ref mut cond, ref mut then_stmt, ref mut else_stmt) => {
            analyze_expression(cond, ctx, symbols, Some(&Type::Bool));
            expect_convert_exact!(ctx, cond, &Type::Bool);

            analyze_statement(then_stmt, ctx, symbols);

            if let Some(else_stmt) = else_stmt {
                analyze_statement(else_stmt, ctx, symbols);
                stmt.will_return = then_stmt.will_return && else_stmt.will_return;
            } else if !ctx.features.contains_key("optional_else") {
                ctx.push_error(missing_else!(stmt.span));
            };
        },
        ast::StmtType::WhileDo(ref mut cond, ref mut do_stmt) => {
            analyze_expression(cond, ctx, symbols, Some(&Type::Bool));
            expect_convert_exact!(ctx, cond, &Type::Bool);

            analyze_statement(do_stmt, ctx, symbols);
            stmt.will_return = if let ast::ExprType::Bool(true) = cond.node { true } else { false };
        },
        ast::StmtType::Read(ref mut loc) => {
            let val_type = analyze_expression(loc, ctx, symbols, None);
            let is_valid = match val_type {
                Type::Bool => true,
                Type::Char => true,
                Type::Int => true,
                Type::Real => true,
                Type::Error => true,
                _ => false
            };

            if !loc.assignable {
                ctx.push_error(cannot_assign!(loc));
            } else if !is_valid {
                ctx.push_error(cannot_read!(ctx, loc));
            };
        },
        ast::StmtType::Assign(ref mut loc, ref mut val) => {
            let expected_type = analyze_expression(loc, ctx, symbols, None);
            analyze_expression(val, ctx, symbols, if expected_type.is_resolved() {
                Some(&expected_type)
            } else {
                None
            });

            if !loc.assignable {
                ctx.push_error(cannot_assign!(loc));
            } else {
                expect_convert_exact!(ctx, val, &expected_type);
            };
        },
        ast::StmtType::Print(ref mut val) => {
            let val_type = analyze_expression(val, ctx, symbols, None);
            let is_valid = match val_type {
                Type::Bool => true,
                Type::Char => true,
                Type::Int => true,
                Type::Real => true,
                Type::Error => true,
                _ => false
            };

            if !is_valid {
                ctx.push_error(cannot_print!(ctx, val));
            };
        },
        ast::StmtType::Block(ref mut inner_block) => {
            if inner_block.symbols.borrow().parent.is_none() {
                inner_block.symbols.borrow_mut().set_parent(symbols.clone());
            };

            populate_block_symbol_table(inner_block, ctx);

            stmt.will_return = inner_block.stmts.iter().any(|s| s.will_return);
        },
        ast::StmtType::Case(ref mut val, ref mut cases) => {
            analyze_cases(val, cases, get_statement_symbols, ctx, symbols);

            for case in cases.iter_mut() {
                analyze_statement(&mut case.branch, ctx, symbols);
            };

            stmt.will_return = val.val_type.are_cases_exhaustive(ctx.tdt, cases)
                && cases.iter().all(|c| c.branch.will_return);
        },
        ast::StmtType::Return(ref mut val) => {
            analyze_expression(val, ctx, symbols, ctx.expected_return);

            if let Some(expected_return) = ctx.expected_return {
                expect_convert_exact!(ctx, val, expected_return);
            } else {
                ctx.push_error(cannot_return_outside_func!(stmt.span));
            };

            stmt.will_return = true;
        }
    };
}

fn populate_block_symbol_table(
    block: &mut ast::Block,
    ctx: &AnalysisContext
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
                *type_id = ctx.tdt.add_dummy_definition();
                if let Some((_, old_type_span)) = symbols.find_imm_named_type(&name) {
                    ctx.push_error(type_already_defined!(name, decl.span, old_type_span));
                } else {
                    symbols.add_type(name.clone(), *type_id, decl.span);
                };
            };
        };

        for decl in mem::replace(&mut block.decls, vec![]) {
            create_symbol_for_decl(decl, ctx, symbols);
        };
    };

    // Now that all symbols are fully declared, we can start examining any embedded expressions or
    // statements for these symbols. This means that function bodies and array dimensions are fully
    // examined at this point.
    for (_, &sym) in &block.symbols.borrow().symbol_names {
        let sym = ctx.sdt.get_symbol(sym);
        match sym.node {
            SymbolType::Fun(ref fs) => {
                populate_function_symbol_table(sym.id, &fs, &sym.span, ctx, &block.symbols);
            },
            SymbolType::Var(ref vs) => {
                for d in vs.dims.borrow_mut().iter_mut() {
                    analyze_expression(d, ctx, &block.symbols, Some(&Type::Int));
                    expect_convert_exact!(ctx, d, &Type::Int);
                };
            },
            _ => {}
        };
    };

    for stmt in &mut block.stmts {
        analyze_statement(stmt, ctx, &block.symbols);
    };
}

fn populate_function_symbol_table(
    id: usize,
    sym: &FunSymbol,
    span: &Span,
    ctx: &AnalysisContext,
    parent_symbols: &Rc<RefCell<SymbolTable>>,
) {
    let block = &mut sym.body.borrow_mut();
    let return_type = if let TypeDefinition::Function(ref sig) = ctx.tdt.get_definition(sym.sig) {
        sig.return_type.clone()
    } else {
        panic!("Invalid function signature")
    };

    block.symbols.borrow_mut().set_parent(parent_symbols.clone());

    populate_block_symbol_table(
        block,
        &ctx.inside_function(id, Some(&return_type))
    );

    // If an @feature(return_anywhere) annotation is present, then we allow return statements
    // anywhere inside the function body, so long as the function is guaranteed to always return.
    // Otherwise, we need to check that the last statement is a return statement and that none of
    // other statements in the function are return statements.
    if ctx.features.contains_key("return_anywhere") {
        if !block.stmts.iter().any(|s| s.will_return) {
            ctx.push_error(missing_return_anywhere!(*span));
        };
    } else {
        let has_final_return = if let Some(ast::StmtType::Return(_)) = block.stmts.last().map(|s| &s.node) {
            true
        } else {
            false
        };

        if !has_final_return {
            ctx.push_error(missing_return_at_end!(*span));
        } else {
            fn check_no_return(s: &ast::Stmt, ctx: &AnalysisContext) -> () {
                use ast::StmtType::*;

                match s.node {
                    IfThenElse(_, ref then_stmt, ref else_stmt) => {
                        check_no_return(then_stmt, ctx);

                        if let Some(else_stmt) = else_stmt {
                            check_no_return(else_stmt, ctx);
                        };
                    },
                    WhileDo(_, ref do_stmt) => {
                        check_no_return(do_stmt, ctx);
                    },
                    Block(ref block) => {
                        for s in &block.stmts {
                            check_no_return(s, ctx);
                        };
                    },
                    Case(_, ref cases) => {
                        for c in cases {
                            check_no_return(&c.branch, ctx);
                        };
                    },
                    Return(_) => {
                        ctx.push_error(cannot_return_except_at_end!(s.span));
                    },
                    _ => {}
                };
            }

            for s in &block.stmts[..(block.stmts.len() - 1)] {
                check_no_return(s, ctx);
            };
        };
    };
}

pub fn populate_symbol_tables(
    program: &mut ast::Program,
    errors: &mut Vec<(String, Span)>
) {
    for (feature, &span) in &program.features {
        if !VALID_FEATURES.contains(&feature[..]) {
            errors.push((
                format!("unknown feature {}", feature),
                span
            ));
        };
    };

    populate_block_symbol_table(
        &mut program.block,
        &AnalysisContext::new(!0, &program.types, &program.symbols, &program.features, None, errors)
    );
}
