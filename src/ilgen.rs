use std::collections::HashMap;
use std::io::Write;

use ast;
use il::{BasicBlock, FlowGraph, IlConst, IlFloat, IlInstruction, IlOperand, IlRegister, IlType, Program};
use symbol;
use util::PrettyDisplay;

struct IlGenContext<'a> {
    func_id: usize,
    tdt: &'a symbol::TypeDefinitionTable,
    sdt: &'a symbol::SymbolDefinitionTable
}

fn translate_type(
    t: &symbol::Type,
    ctx: &mut IlGenContext
) -> IlType {
    use symbol::Type::*;

    // TODO Support arrays
    match *t {
        Int => IlType::Int,
        Bool => IlType::Int,
        Char => IlType::Int,
        Real => IlType::Float,
        Defined(id) => match ctx.tdt.get_definition(id) {
            symbol::TypeDefinition::Data(_) => IlType::Addr,
            _ => panic!("unsupported type {}", t.pretty(ctx.tdt))
        },
        Array(_, _) => IlType::Addr,
        PartialArray(_, _) => IlType::Addr,
        ref t => panic!("unsupported type {}", t.pretty(ctx.tdt))
    }
}

fn append_cases<T, U: FnMut (&T, &mut IlGenContext, &mut BasicBlock, &mut FlowGraph, &mut Write)>(
    cases: &[ast::Case<T>],
    mut appender: U,
    val: &ast::Expr,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) {
    let val_reg = append_expr(val, ctx, block, g, w);
    let variant_reg = g.registers.alloc_temp(IlType::Int);
    block.instrs.push(IlInstruction::LoadInt(variant_reg, IlOperand::Register(val_reg)));

    let val_type = match val.val_type {
        symbol::Type::Defined(id) => match ctx.tdt.get_definition(id) {
            symbol::TypeDefinition::Data(ref td) => td,
            _ => unreachable!()
        },
        _ => unreachable!()
    };
    let mut branch_ends = vec![];

    for c in cases.iter() {
        let check_block = block.id;
        let check_reg = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::EqInt(
            check_reg,
            IlOperand::Register(variant_reg),
            IlOperand::Const(IlConst::Int(c.ctor_id as i32))
        ));
        block.instrs.push(IlInstruction::JumpZero(
            IlOperand::Register(check_reg),
            !0
        ));
        block.successor = Some(block.id + 1);
        g.append_block(block);

        let ctor = &val_type.ctors[c.ctor_id];
        let types: Vec<_> = ctor.args.iter().map(|a| translate_type(a, ctx)).collect();

        for (i, &val_sym) in c.var_bindings.iter().enumerate() {
            let t = &types[i];
            let addr_reg = g.registers.alloc_temp(IlType::Addr);
            let subval_reg = g.registers.get_or_alloc_local(ctx.func_id, ctx.sdt.get_symbol(val_sym), t.clone());

            block.instrs.push(IlInstruction::AddAddr(
                addr_reg,
                IlOperand::Register(val_reg),
                IlOperand::Const(IlConst::Addr(
                    4 + types.iter().take(i).map(|t| t.size()).sum::<u64>()
                ))
            ));

            block.instrs.push(match t {
                IlType::Int => IlInstruction::LoadInt(
                    subval_reg,
                    IlOperand::Register(addr_reg)
                ),
                IlType::Float => IlInstruction::LoadFloat(
                    subval_reg,
                    IlOperand::Register(addr_reg)
                ),
                IlType::Addr => IlInstruction::LoadAddr(
                    subval_reg,
                    IlOperand::Register(addr_reg)
                )
            });
        };

        appender(&c.branch, ctx, block, g, w);
        branch_ends.push(block.id);
        g.append_block(block);

        g.blocks.get_mut(&check_block).unwrap().relink_alt_successor(block.id);
    };

    for branch_end in branch_ends {
        g.blocks.get_mut(&branch_end).unwrap().successor = Some(block.id);
    };
}

fn append_index_calc(
    arr_expr: &ast::Expr,
    ind_expr: &ast::Expr,
    elem_size: u64,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> IlRegister {
    fn unwrap_array_expr<'a>(
        mut e: &'a ast::Expr,
        is: &mut Vec<&'a ast::Expr>
    ) -> &'a ast::Expr {
        while let ast::ExprType::Index(ref arr_expr, ref ind_expr) = e.node {
            is.push(ind_expr);
            e = arr_expr;
        };
        e
    }

    let mut ind_exprs = vec![ind_expr];
    let arr_reg = append_expr(unwrap_array_expr(arr_expr, &mut ind_exprs), ctx, block, g, w);

    let mut ind_reg = append_expr(ind_exprs.last().unwrap(), ctx, block, g, w);
    let mut size_mult = IlOperand::Const(IlConst::Int(1));

    for (i, &ind_expr) in ind_exprs.iter().rev().skip(1).enumerate() {
        let dim_ind = append_expr(ind_expr, ctx, block, g, w);
        let next_size_addr = g.registers.alloc_temp(IlType::Addr);

        block.instrs.push(IlInstruction::AddAddr(
            next_size_addr,
            IlOperand::Register(arr_reg),
            IlOperand::Const(IlConst::Addr(i as u64 * IlType::Int.size()))
        ));

        let next_size = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::LoadInt(
            next_size,
            IlOperand::Register(next_size_addr)
        ));

        let next_size_mult = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::MulInt(
            next_size_mult,
            size_mult,
            IlOperand::Register(next_size)
        ));

        let dim_ind_mult = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::MulInt(
            dim_ind_mult,
            IlOperand::Register(dim_ind),
            IlOperand::Register(next_size_mult)
        ));

        let next_ind = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::AddInt(
            next_ind,
            IlOperand::Register(ind_reg),
            IlOperand::Register(dim_ind_mult)
        ));

        size_mult = IlOperand::Register(next_size_mult);
        ind_reg = next_ind;
    };

    let ind_as_addr = g.registers.alloc_temp(IlType::Addr);

    block.instrs.push(IlInstruction::Int2Addr(
        ind_as_addr,
        IlOperand::Register(ind_reg)
    ));

    let off_reg = g.registers.alloc_temp(IlType::Addr);

    block.instrs.push(IlInstruction::MulAddr(
        off_reg,
        IlOperand::Register(ind_as_addr),
        IlOperand::Const(IlConst::Addr(elem_size))
    ));

    let total_off_reg = g.registers.alloc_temp(IlType::Addr);

    block.instrs.push(IlInstruction::AddAddr(
        total_off_reg,
        IlOperand::Register(off_reg),
        IlOperand::Const(IlConst::Addr(IlType::Int.size() * ind_exprs.len() as u64))
    ));

    let addr_reg = g.registers.alloc_temp(IlType::Addr);

    block.instrs.push(IlInstruction::AddAddr(
        addr_reg,
        IlOperand::Register(arr_reg),
        IlOperand::Register(total_off_reg)
    ));

    addr_reg
}

fn append_expr_to(
    expr: &ast::Expr,
    target: IlRegister,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> IlRegister {
    use ast::ExprType::*;

    writeln!(w, "@{} <= {} <- {}", block.id, target, expr.pretty()).unwrap();

    match expr.node {
        BinaryOp(_, ref lhs, ref rhs, symbol::BinaryOp::BoolOr) => {
            let start_block = block.id;

            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(1))));

            let lhs = IlOperand::Register(append_expr(lhs, ctx, block, g, w));
            let lhs_end_block = block.id;

            block.instrs.push(IlInstruction::JumpNonZero(lhs, !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            let rhs = IlOperand::Register(append_expr(rhs, ctx, block, g, w));
            let rhs_end_block = block.id;

            block.instrs.push(IlInstruction::JumpNonZero(rhs, !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            let false_block = block.id;

            writeln!(w, "@{} <= BoolOr(@{}) false path", block.id, start_block).unwrap();
            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(0))));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            writeln!(w, "@{} <= BoolOr(@{}) join", block.id, start_block).unwrap();
            g.blocks.get_mut(&lhs_end_block).unwrap().relink_alt_successor(block.id);
            g.blocks.get_mut(&rhs_end_block).unwrap().relink_alt_successor(block.id);
            g.blocks.get_mut(&false_block).unwrap().successor = Some(block.id);
        },
        BinaryOp(_, ref lhs, ref rhs, symbol::BinaryOp::BoolAnd) => {
            let start_block = block.id;

            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(0))));

            let lhs = IlOperand::Register(append_expr(lhs, ctx, block, g, w));
            let lhs_end_block = block.id;

            block.instrs.push(IlInstruction::JumpZero(lhs, !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            let rhs = IlOperand::Register(append_expr(rhs, ctx, block, g, w));
            let rhs_end_block = block.id;

            block.instrs.push(IlInstruction::JumpZero(rhs, !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            let true_block = block.id;

            writeln!(w, "@{} <= BoolAnd(@{}) true path", block.id, start_block).unwrap();
            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(1))));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            writeln!(w, "@{} <= BoolAnd(@{}) join", block.id, start_block).unwrap();
            g.blocks.get_mut(&lhs_end_block).unwrap().relink_alt_successor(block.id);
            g.blocks.get_mut(&rhs_end_block).unwrap().relink_alt_successor(block.id);
            g.blocks.get_mut(&true_block).unwrap().successor = Some(block.id);
        },
        BinaryOp(_, ref lhs, ref rhs, op) => {
            use symbol::BinaryOp;

            let lhs = IlOperand::Register(append_expr(lhs, ctx, block, g, w));
            let rhs = IlOperand::Register(append_expr(rhs, ctx, block, g, w));

            match op {
                BinaryOp::IntAdd => block.instrs.push(IlInstruction::AddInt(target, lhs, rhs)),
                BinaryOp::IntSub => block.instrs.push(IlInstruction::SubInt(target, lhs, rhs)),
                BinaryOp::IntMul => block.instrs.push(IlInstruction::MulInt(target, lhs, rhs)),
                BinaryOp::IntDiv => block.instrs.push(IlInstruction::DivInt(target, lhs, rhs)),
                BinaryOp::IntEq | BinaryOp::BoolEq | BinaryOp::CharEq => (
                    block.instrs.push(IlInstruction::EqInt(target, lhs, rhs))
                ),
                BinaryOp::IntLt => block.instrs.push(IlInstruction::LtInt(target, lhs, rhs)),
                BinaryOp::IntGt => block.instrs.push(IlInstruction::GtInt(target, lhs, rhs)),
                BinaryOp::IntLe => block.instrs.push(IlInstruction::LeInt(target, lhs, rhs)),
                BinaryOp::IntGe => block.instrs.push(IlInstruction::GeInt(target, lhs, rhs)),
                BinaryOp::RealEq => block.instrs.push(IlInstruction::EqFloat(target, lhs, rhs)),
                BinaryOp::RealLt => block.instrs.push(IlInstruction::LtFloat(target, lhs, rhs)),
                BinaryOp::RealGt => block.instrs.push(IlInstruction::GtFloat(target, lhs, rhs)),
                BinaryOp::RealLe => block.instrs.push(IlInstruction::LeFloat(target, lhs, rhs)),
                BinaryOp::RealGe => block.instrs.push(IlInstruction::GeFloat(target, lhs, rhs)),
                BinaryOp::RealAdd => block.instrs.push(IlInstruction::AddFloat(target, lhs, rhs)),
                BinaryOp::RealSub => block.instrs.push(IlInstruction::SubFloat(target, lhs, rhs)),
                BinaryOp::RealMul => block.instrs.push(IlInstruction::MulFloat(target, lhs, rhs)),
                BinaryOp::RealDiv => block.instrs.push(IlInstruction::DivFloat(target, lhs, rhs)),
                BinaryOp::BoolOr | BinaryOp::BoolAnd | BinaryOp::Unknown => unreachable!()
            };
        },
        UnaryOp(_, ref val, op) => {
            let val = IlOperand::Register(append_expr(val, ctx, block, g, w));

            use symbol::UnaryOp;

            match op {
                UnaryOp::BoolNot => block.instrs.push(IlInstruction::LogicNotInt(target, val)),
                UnaryOp::IntFloat => block.instrs.push(IlInstruction::Int2Float(target, val)),
                UnaryOp::IntNeg => block.instrs.push(IlInstruction::MulInt(
                    target,
                    val,
                    IlOperand::Const(IlConst::Int(-1))
                )),
                UnaryOp::RealNeg => block.instrs.push(IlInstruction::MulFloat(
                    target,
                    val,
                    IlOperand::Const(IlConst::Float(IlFloat::from_f64(-1.0)))
                )),
                op => panic!("not yet supported: {:?}", op)
            }
        },
        Size(ref val, dims) => {
            let val = IlOperand::Register(append_expr(val, ctx, block, g, w));
            let addr = g.registers.alloc_temp(IlType::Addr);

            block.instrs.push(IlInstruction::AddAddr(
                addr,
                val,
                IlOperand::Const(IlConst::Addr(dims as u64 * IlType::Int.size()))
            ));
            block.instrs.push(IlInstruction::LoadInt(
                target,
                IlOperand::Register(addr)
            ));
        },
        Id(_, sym_id) => {
            let sym = ctx.sdt.get_symbol(sym_id);

            match sym.node {
                symbol::SymbolType::Var(_) | symbol::SymbolType::Param(_) => {
                    block.instrs.push(IlInstruction::Copy(
                        target,
                        IlOperand::Register(g.registers.get_or_alloc_local(
                            ctx.func_id,
                            sym,
                            translate_type(&sym.val_type(), ctx)
                        ))
                    ))
                },
                _ => panic!("load from invalid symbol {}", sym_id)
            }
        },
        Call(box ast::Expr { node: Id(_, func), .. }, ref args) => {
            let args: Vec<_> = args.iter().map(|a| {
                IlOperand::Register(append_expr(a, ctx, block, g, w))
            }).collect();

            block.instrs.push(IlInstruction::CallDirect(target, func, args));
        },
        Call(_, _) => panic!("indirect calls are not yet supported"),
        Index(ref arr, ref ind) => {
            let addr_reg = append_index_calc(
                arr,
                ind,
                translate_type(&expr.val_type, ctx).size(),
                ctx,
                block,
                g,
                w
            );

            block.instrs.push(match translate_type(&expr.val_type, ctx) {
                IlType::Int => IlInstruction::LoadInt(
                    target,
                    IlOperand::Register(addr_reg)
                ),
                IlType::Float => IlInstruction::LoadFloat(
                    target,
                    IlOperand::Register(addr_reg)
                ),
                IlType::Addr => IlInstruction::LoadAddr(
                    target,
                    IlOperand::Register(addr_reg)
                )
            });
        },
        Cons(_, ref args, ctor_id) => {
            let types: Vec<_> = args.iter().map(|a| translate_type(&a.val_type, ctx)).collect();

            block.instrs.push(IlInstruction::AllocHeap(
                target,
                IlOperand::Const(IlConst::Addr(
                    IlType::Int.size() + types.iter().map(|t| t.size()).sum::<u64>()
                ))
            ));
            block.instrs.push(IlInstruction::StoreInt(
                IlOperand::Register(target),
                IlOperand::Const(IlConst::Int(ctor_id as i32))
            ));

            for (i, arg) in args.iter().enumerate() {
                let addr_reg = g.registers.alloc_temp(IlType::Addr);
                let val_reg = append_expr(arg, ctx, block, g, w);

                block.instrs.push(IlInstruction::AddAddr(
                    addr_reg,
                    IlOperand::Register(target),
                    IlOperand::Const(IlConst::Addr(
                        4 + types.iter().take(i).map(|t| t.size()).sum::<u64>()
                    ))
                ));

                block.instrs.push(match translate_type(&arg.val_type, ctx) {
                    IlType::Int => IlInstruction::StoreInt(
                        IlOperand::Register(addr_reg),
                        IlOperand::Register(val_reg)
                    ),
                    IlType::Float => IlInstruction::StoreFloat(
                        IlOperand::Register(addr_reg),
                        IlOperand::Register(val_reg)
                    ),
                    IlType::Addr => IlInstruction::StoreAddr(
                        IlOperand::Register(addr_reg),
                        IlOperand::Register(val_reg)
                    )
                });
            };
        },
        Int(val) => {
            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(val))));
        },
        Bool(val) => {
            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(if val {
                1
            } else {
                0
            }))));
        },
        Char(val) => {
            block.instrs.push(IlInstruction::Copy(target, IlOperand::Const(IlConst::Int(val as i32))));
        },
        Real(val) => {
            block.instrs.push(IlInstruction::Copy(
                target,
                IlOperand::Const(IlConst::Float(IlFloat::from_f64(val)))
            ));
        },
        Block(ref ast_block, ref result) => {
            append_block(ast_block, ctx, block, g, w);

            if let Some(ref result) = *result {
                append_expr_to(result, target, ctx, block, g, w);
            };
        },
        Case(ref val, ref cases) => {
            append_cases(
                &cases[..],
                |e, ctx, block, g, w| { append_expr_to(e, target, ctx, block, g, w); },
                val,
                ctx,
                block,
                g,
                w
            );
        }
    };

    target
}

fn append_expr(
    expr: &ast::Expr,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> IlRegister {
    let target = g.registers.alloc_temp(translate_type(&expr.val_type, ctx));

    append_expr_to(
        expr,
        target,
        ctx,
        block,
        g,
        w
    )
}

fn append_store_to_expr(
    expr: &ast::Expr,
    source: IlRegister,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> () {
    use ast::ExprType::*;

    writeln!(w, "@{} <= {} -> {}", block.id, source, expr.pretty()).unwrap();

    match expr.node {
        Id(_, sym_id) => {
            let sym = ctx.sdt.get_symbol(sym_id);

            match sym.node {
                symbol::SymbolType::Var(_) | symbol::SymbolType::Param(_) => {
                    block.instrs.push(IlInstruction::Copy(
                        g.registers.get_or_alloc_local(
                            ctx.func_id,
                            sym,
                            translate_type(&sym.val_type(), ctx)
                        ),
                        IlOperand::Register(source)
                    ));
                },
                _ => panic!("store to invalid symbol {}", sym_id)
            }
        },
        Index(ref arr, ref ind) => {
            let addr_reg = append_index_calc(
                arr,
                ind,
                translate_type(&expr.val_type, ctx).size(),
                ctx,
                block,
                g,
                w
            );

            block.instrs.push(match translate_type(&expr.val_type, ctx) {
                IlType::Int => IlInstruction::StoreInt(
                    IlOperand::Register(addr_reg),
                    IlOperand::Register(source)
                ),
                IlType::Float => IlInstruction::StoreFloat(
                    IlOperand::Register(addr_reg),
                    IlOperand::Register(source)
                ),
                IlType::Addr => IlInstruction::StoreAddr(
                    IlOperand::Register(addr_reg),
                    IlOperand::Register(source)
                )
            });
        },
        ref n => panic!("not supported as lvalue: {:?}", n)
    }
}

fn append_stmt(
    stmt: &ast::Stmt,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> () {
    use ast::StmtType::*;

    writeln!(w, "@{} <= {}", block.id, stmt.pretty()).unwrap();

    match stmt.node {
        IfThenElse(ref cond, ref then_stmt, ref else_stmt) => {
            let start_block = block.id;
            let cond = append_expr(cond, ctx, block, g, w);
            let cond_end_block = block.id;

            block.instrs.push(IlInstruction::JumpZero(IlOperand::Register(cond), !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            append_stmt(then_stmt, ctx, block, g, w);
            let then_end_block = g.append_block(block);

            let else_start_block = block.id;
            if let Some(ref else_stmt) = *else_stmt {
                append_stmt(else_stmt, ctx, block, g, w);
                block.successor = Some(block.id + 1);
                g.append_block(block);
            };

            g.blocks.get_mut(&cond_end_block).unwrap().relink_alt_successor(else_start_block);
            g.blocks.get_mut(&then_end_block).unwrap().successor = Some(block.id);

            writeln!(w, "@{} <= If(@{}) join", block.id, start_block).unwrap();
        },
        WhileDo(ref cond, ref do_stmt) => {
            let start_block = block.id;

            block.successor = Some(block.id + 1);
            g.append_block(block);

            let cond_start_block = block.id;
            let cond = append_expr(cond, ctx, block, g, w);
            let cond_end_block = block.id;

            block.instrs.push(IlInstruction::JumpZero(IlOperand::Register(cond), !0));
            block.successor = Some(block.id + 1);
            g.append_block(block);

            append_stmt(do_stmt, ctx, block, g, w);
            block.successor = Some(cond_start_block);
            g.append_block(block);

            g.blocks.get_mut(&cond_end_block).unwrap().relink_alt_successor(block.id);

            writeln!(w, "@{} <= While(@{}) join", block.id, start_block).unwrap();
        },
        Read(ref loc) => {
            let reg = g.registers.alloc_temp(translate_type(&loc.val_type, ctx));

            block.instrs.push(match loc.val_type.clone() {
                symbol::Type::Int => IlInstruction::ReadInt(reg),
                symbol::Type::Bool => IlInstruction::ReadBool(reg),
                symbol::Type::Char => IlInstruction::ReadChar(reg),
                symbol::Type::Real => IlInstruction::ReadFloat(reg),
                _ => unreachable!()
            });

            append_store_to_expr(loc, reg, ctx, block, g, w);
        },
        Assign(ref loc, ref val) => {
            append_store_to_expr(
                loc,
                append_expr(val, ctx, block, g, w),
                ctx,
                block,
                g,
                w
            );
        },
        Print(ref val) => {
            let val_reg = append_expr(val, ctx, block, g, w);
            block.instrs.push(match val.val_type.clone() {
                symbol::Type::Int => IlInstruction::PrintInt(IlOperand::Register(val_reg)),
                symbol::Type::Bool => IlInstruction::PrintBool(IlOperand::Register(val_reg)),
                symbol::Type::Char => IlInstruction::PrintChar(IlOperand::Register(val_reg)),
                symbol::Type::Real => IlInstruction::PrintFloat(IlOperand::Register(val_reg)),
                _ => unreachable!()
            });
        },
        Block(ref ast_block) => {
            append_block(ast_block, ctx, block, g, w);
        },
        Case(ref val, ref cases) => {
            append_cases(
                &cases[..],
                append_stmt,
                val,
                ctx,
                block,
                g,
                w
            );
        },
        Return(ref val) => {
            let val_reg = append_expr(val, ctx, block, g, w);
            block.instrs.push(IlInstruction::Return(IlOperand::Register(val_reg)));

            // Note that this basic block is intentionally *not* connected to the next one. This
            // way, if any statements appear after the return, they will be properly detected as
            // dead code.
            g.append_block(block);
        }
    }
}

fn append_size_calc(
    dims: &[IlRegister],
    val_type: IlType,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> IlRegister {
    let mut num_reg = dims[0];

    for &d in dims.iter().skip(1) {
        let next_reg = g.registers.alloc_temp(IlType::Int);

        block.instrs.push(IlInstruction::MulInt(
            next_reg,
            IlOperand::Register(num_reg),
            IlOperand::Register(d)
        ));
        num_reg = next_reg;
    };

    let size_reg = g.registers.alloc_temp(IlType::Int);

    block.instrs.push(IlInstruction::MulInt(
        size_reg,
        IlOperand::Register(num_reg),
        IlOperand::Const(IlConst::Int(val_type.size() as i32))
    ));

    let size_with_overhead_reg = g.registers.alloc_temp(IlType::Addr);

    block.instrs.push(IlInstruction::AddInt(
        size_with_overhead_reg,
        IlOperand::Register(size_reg),
        IlOperand::Const(IlConst::Int(IlType::Int.size() as i32 * dims.len() as i32))
    ));

    size_with_overhead_reg
}

fn append_block(
    ast_block: &ast::Block,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> () {
    let mut total_size_reg = None;
    for (_, &sym_id) in ast_block.symbols.borrow().symbol_names.iter() {
        let sym = ctx.sdt.get_symbol(sym_id);
        match sym {
            symbol::Symbol { node: symbol::SymbolType::Var(ref vsym), .. } if vsym.dims.borrow().len() > 0 => {
                let sizes: Vec<_> = vsym.dims.borrow().iter()
                    .map(|d| append_expr(d, ctx, block, g, w))
                    .collect();
                let size_reg = append_size_calc(
                    &sizes[..],
                    translate_type(if let symbol::Type::Array(ref inner_type, _) = vsym.val_type {
                        inner_type
                    } else {
                        unreachable!()
                    }, ctx),
                    ctx,
                    block,
                    g,
                    w
                );

                let reg = g.registers.get_or_alloc_local(ctx.func_id, sym, IlType::Addr);
                block.instrs.push(IlInstruction::AllocStack(
                    reg,
                    IlOperand::Register(size_reg)
                ));

                for (i, &size) in sizes.iter().enumerate() {
                    let i = i as u64;
                    let addr_reg = g.registers.alloc_temp(IlType::Addr);

                    block.instrs.push(IlInstruction::AddAddr(
                        addr_reg,
                        IlOperand::Register(reg),
                        IlOperand::Const(IlConst::Addr(IlType::Int.size() * i))
                    ));
                    block.instrs.push(IlInstruction::StoreInt(
                        IlOperand::Register(addr_reg),
                        IlOperand::Register(size)
                    ));
                };

                if let Some(old_total_size_reg) = total_size_reg {
                    let next_reg = g.registers.alloc_temp(IlType::Int);

                    block.instrs.push(IlInstruction::AddInt(
                        next_reg,
                        IlOperand::Register(old_total_size_reg),
                        IlOperand::Register(size_reg)
                    ));
                    total_size_reg = Some(next_reg);
                } else {
                    total_size_reg = Some(size_reg);
                };
            },
            _ => {}
        };
    };

    for stmt in &ast_block.stmts {
        append_stmt(stmt, ctx, block, g, w);
    };

    if let Some(total_size_reg) = total_size_reg {
        block.instrs.push(IlInstruction::FreeStack(IlOperand::Register(total_size_reg)));
    };
}

fn generate_function_il(
    func: &ast::Block,
    args: Option<&[usize]>,
    ctx: &mut IlGenContext,
    w: &mut Write
) -> FlowGraph {
    let mut g = FlowGraph::new();
    let mut b = BasicBlock::new(0);

    if ctx.func_id == !0 {
        writeln!(w, "========== GENERATING IL FOR MAIN BLOCK ==========\n").unwrap();
    } else {
        writeln!(w, "========== GENERATING IL FOR FUNCTION #{} ==========\n", ctx.func_id).unwrap();
    };

    if let Some(args) = args {
        g.registers.args = args.iter().map(|&sym_id| {
            let sym = ctx.sdt.get_symbol(sym_id);

            g.registers.get_or_alloc_local(
                ctx.func_id,
                sym,
                translate_type(&sym.val_type(), ctx)
            )
        }).collect();
    };

    append_block(func, ctx, &mut b, &mut g, w);
    g.append_block(&mut b);

    writeln!(w, "\n========== GENERATED IL ==========\n{}", g).unwrap();

    g
}

pub fn generate_il(program: &ast::Program, w: &mut Write) -> Program {
    writeln!(w, "========== IL GENERATION ==========\n").unwrap();

    Program {
        main_block: generate_function_il(&program.block, None, &mut IlGenContext {
            func_id: !0,
            tdt: &program.types,
            sdt: &program.symbols
        }, w),
        funcs: program.symbols.iter().filter_map(|(_, sym)| if let symbol::SymbolType::Fun(ref f) = sym.node {
            Some((sym.id, generate_function_il(&f.body.borrow(), Some(&f.params), &mut IlGenContext {
                func_id: sym.id,
                tdt: &program.types,
                sdt: &program.symbols
            }, w)))
        } else {
            None
        }).collect(),
        ipa: HashMap::new()
    }
}
