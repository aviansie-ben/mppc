use std::io::Write;

use ast;
use il::{BasicBlock, FlowGraph, IlConst, IlInstruction, IlOperand, IlRegister, IlType, Program};
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

    // TODO Support non-integers
    match *t {
        Int => IlType::Int,
        Bool => IlType::Int,
        Char => IlType::Int,
        ref t => panic!("unsupported type {}", t)
    }
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
                BinaryOp::BoolOr | BinaryOp::BoolAnd => unreachable!(),
                op => panic!("not yet supported: {:?}", op)
            };
        },
        UnaryOp(_, ref val, op) => {
            let val = IlOperand::Register(append_expr(val, ctx, block, g, w));

            use symbol::UnaryOp;

            match op {
                UnaryOp::BoolNot => block.instrs.push(IlInstruction::LogicNotInt(target, val)),
                UnaryOp::IntNeg => block.instrs.push(IlInstruction::MulInt(
                    target,
                    val,
                    IlOperand::Const(IlConst::Int(-1))
                )),
                op => panic!("not yet supported: {:?}", op)
            }
        },
        Id(_, sym_id) => {
            let sym = ctx.sdt.get_symbol(sym_id);

            match sym.node {
                symbol::SymbolType::Var(_) | symbol::SymbolType::Param(_) => {
                    // TODO Handle globals and nonlocals
                    assert_eq!(sym.defining_fun, ctx.func_id, "Nonlocals are not yet supported");

                    block.instrs.push(IlInstruction::Copy(
                        target,
                        IlOperand::Register(g.registers.get_or_alloc_local(
                            sym_id,
                            translate_type(&sym.val_type(), ctx)
                        ))
                    ))
                },
                _ => panic!("load from invalid symbol {}", sym_id)
            }
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
        Block(ref ast_block, ref result) => {
            append_block(ast_block, ctx, block, g, w);

            if let Some(ref result) = *result {
                append_expr_to(result, target, ctx, block, g, w);
            };
        },
        ref n => panic!("not yet supported: {:?}", n)
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
                    // TODO Handle globals and nonlocals
                    assert_eq!(sym.defining_fun, ctx.func_id, "Nonlocals are not yet supported");

                    block.instrs.push(IlInstruction::Copy(
                        g.registers.get_or_alloc_local(
                            sym_id,
                            translate_type(&sym.val_type(), ctx)
                        ),
                        IlOperand::Register(source)
                    ));
                },
                _ => panic!("store to invalid symbol {}", sym_id)
            }
        },
        ref n => panic!("not yet supported: {:?}", n)
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
                _ => unreachable!()
            });
        },
        Block(ref ast_block) => {
            append_block(ast_block, ctx, block, g, w);
        },
        ref n => panic!("not yet supported: {:?}", n)
    }
}

fn append_block(
    ast_block: &ast::Block,
    ctx: &mut IlGenContext,
    block: &mut BasicBlock,
    g: &mut FlowGraph,
    w: &mut Write
) -> () {
    // TODO Allocate arrays

    for stmt in &ast_block.stmts {
        append_stmt(stmt, ctx, block, g, w);
    };

    // TODO Deallocate arrays
}

fn generate_function_il(
    func: &ast::Block,
    ctx: &mut IlGenContext,
    w: &mut Write
) -> FlowGraph {
    let mut g = FlowGraph::new();
    let mut b = BasicBlock::new(0);

    if ctx.func_id == !0 {
        writeln!(w, "========== GENERATING IL FOR MAIN BLOCK ==========\n").unwrap();
    } else {
        writeln!(w, "========== GENERATING IL FOR FUNCTION {} ==========\n", ctx.func_id).unwrap();
    };

    append_block(func, ctx, &mut b, &mut g, w);
    g.append_block(&mut b);

    writeln!(w, "\n========== GENERATED IL ==========\n{}", g).unwrap();

    g
}

pub fn generate_il(program: &ast::Program, w: &mut Write) -> Program {
    writeln!(w, "========== IL GENERATION ==========\n").unwrap();

    Program {
        main_block: generate_function_il(&program.block, &mut IlGenContext {
            func_id: !0,
            tdt: &program.types,
            sdt: &program.symbols
        }, w),
        funcs: vec![]
    }
}
