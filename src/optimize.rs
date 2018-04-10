use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::mem;

use il::{BasicBlock, FlowGraph, IlRegister, IlOperand, IlInstruction, IpaStats, Program, RegisterAlloc};
use util;

fn precompute_liveness(
    block: &BasicBlock,
    ipa: &HashMap<usize, IpaStats>,
    registers: &RegisterAlloc,
    w: &mut Write,
    gen: &mut HashSet<IlRegister>,
    kill: &mut HashSet<IlRegister>
) {
    use il::IlOperand::*;

    // Find the set of registers whose values are used in this block before being assigned (gen) and
    // whose values are set before being used (kill).
    for i in block.instrs.iter().rev() {
        if let Some(reg) = i.target_register() {
            writeln!(w, "@{} - overwriting {}", block.id, reg).unwrap();
            gen.remove(&reg);
            kill.insert(reg);
        };

        if let IlInstruction::CallDirect(_, func_id, _) = *i {
            for &sym_id in ipa[&func_id].nonlocal_refs.iter() {
                if let Some(&reg) = registers.locals.get(&sym_id) {
                    writeln!(w, "@{} - using {} (due to nonlocal access to #{})", block.id, reg, sym_id).unwrap();
                    gen.insert(reg);
                    kill.remove(&reg);
                };
            };
        };

        i.for_operands(|o| {
            if let Register(reg) = o {
                writeln!(w, "@{} - using {}", block.id, reg).unwrap();
                gen.insert(*reg);
                kill.remove(reg);
            };
        });
    };
}

struct LivenessInfo {
    live_vars_begin: HashSet<IlRegister>,
    live_vars_end: HashSet<IlRegister>,
    gen: HashSet<IlRegister>,
    kill: HashSet<IlRegister>,
    checked: bool
}

fn update_liveness(
    block: &BasicBlock,
    g: &FlowGraph,
    w: &mut Write,
    nonlocals: &Vec<IlRegister>,
    all_liveness: &mut HashMap<u32, LivenessInfo>
) -> bool {
    // Each basic block can only be updated once per pass through the CFG. This prevents infinite
    // recursion of calls to update_liveness in the presence of loops in the CFG.
    if all_liveness[&block.id].checked {
        return false;
    };

    all_liveness.get_mut(&block.id).unwrap().checked = true;

    let mut updated = false;
    let mut new_live_vars: HashSet<IlRegister> = HashSet::new();

    // If this block has a successor, then propagate its liveness information to this block.
    // Otherwise, we need to make sure that all registers corresponding to nonlocals are live, since
    // their values can be observed after the function returns.
    if let Some(target) = block.successor {
        updated = update_liveness(
            &g.blocks[&target],
            g,
            w,
            &nonlocals,
            all_liveness
        ) || updated;

        for reg in &all_liveness[&target].live_vars_begin {
            new_live_vars.insert(*reg);
        }
    } else {
        for &reg in nonlocals.iter() {
            new_live_vars.insert(reg);
        };
    };

    // If this block has an alternate successor, then propagate its liveness information to this
    // block.
    if let Some(target) = block.alt_successor() {
        updated = update_liveness(
            &g.blocks[&target],
            g,
            w,
            &nonlocals,
            all_liveness
        ) || updated;

        for reg in &all_liveness[&target].live_vars_begin {
            new_live_vars.insert(*reg);
        }
    };

    // Find this block's saved liveness information, overwrite the liveness information at the end
    // with the newly found liveness information, then apply the gen and kill sets to find the
    // liveness information at the start of the block.
    let liveness = all_liveness.get_mut(&block.id).unwrap();
    liveness.live_vars_end.clone_from(&mut new_live_vars);

    for reg in &liveness.kill {
        new_live_vars.remove(reg);
    }

    for reg in &liveness.gen {
        new_live_vars.insert(*reg);
    }

    if new_live_vars != liveness.live_vars_begin {
        liveness.live_vars_begin = new_live_vars;
        updated = true;

        write!(w, "@{} - propagation changed live vars [", block.id).unwrap();

        let mut first = true;

        for reg in &liveness.live_vars_begin {
            if first {
                write!(w, "{}", reg).unwrap();
                first = false;
            } else {
                write!(w, ", {}", reg).unwrap();
            }
        }

        writeln!(w, "]").unwrap();
    };

    updated
}

fn build_liveness_graph(
    g: &FlowGraph,
    ipa: &HashMap<usize, IpaStats>,
    w: &mut Write
) -> HashMap<u32, HashSet<IlRegister>> {
    writeln!(w, "========== LIVENESS GRAPH CONSTRUCTION ==========\n").unwrap();

    let mut liveness: HashMap<u32, LivenessInfo> = HashMap::new();
    for (_, b) in g.blocks.iter() {
        // Find a gen/kill set that is equivalent (in terms of liveness) to the instructions in the
        // current basic block. See precompute_liveness for more information.
        let mut gen: HashSet<IlRegister> = HashSet::new();
        let mut kill: HashSet<IlRegister> = HashSet::new();

        precompute_liveness(b, ipa, &g.registers, w, &mut gen, &mut kill);

        liveness.insert(b.id, LivenessInfo {
            live_vars_begin: gen.clone(),
            live_vars_end: HashSet::new(),
            gen: gen,
            kill: kill,
            checked: false
        });
    };

    let nonlocals: Vec<_> = g.registers.reg_meta.iter().filter_map(|(&reg, reg_meta)| {
        if reg_meta.reg_type.is_nonlocal() {
            writeln!(w, "@end - implicitly using {} since it is nonlocal", reg).unwrap();
            Some(reg)
        } else {
            None
        }
    }).collect();

    loop {
        writeln!(w, "Propagating between basic blocks...").unwrap();

        // Go through and propagate liveness information between basic blocks. If no liveness
        // information was changed, we're done.
        if !update_liveness(&g.blocks[&g.start_block], g, w, &nonlocals, &mut liveness) {
            break;
        };

        // Otherwise, reset the checked flag of all basic blocks in preparation for the next pass.
        for (_, liveness) in &mut liveness {
            liveness.checked = false;
        };
    };

    writeln!(w).unwrap();

    // Remove unnecessary information, returning only the liveness information at the end of each
    // basic block.
    let liveness: HashMap<_, _>
        = liveness.into_iter().map(|(id, l)| (id, l.live_vars_end)).collect();

    liveness
}

fn do_dead_store_elimination(
    g: &mut FlowGraph,
    w: &mut Write,
    l: &HashMap<u32, HashSet<IlRegister>>
) -> u32 {
    writeln!(w, "========== DEAD STORE ELIMINATION ==========\n").unwrap();

    let mut num_eliminated: u32 = 0;
    for (_, b) in &mut g.blocks {
        let mut live_vars = l[&b.id].clone();

        // Look through instructions in reverse order, since liveness information propagates
        // backwards through a basic block.
        for i in b.instrs.iter_mut().rev() {
            if let Some(reg) = i.target_register() {
                // If the register written by this instruction is dead at this point *and* the
                // instruction has no side effects (we can't remove dead input instructions since
                // that would change how prompts for values from the user work).
                if !i.has_side_effect() && !live_vars.contains(&reg) {
                    writeln!(w, "@{} - eliminating {}", b.id, i).unwrap();

                    mem::replace(i, IlInstruction::Nop);
                    num_eliminated += 1;

                    continue;
                };

                // Update the liveness information to reflect the fact that the target register is
                // dead above this point in the basic block.
                live_vars.remove(&reg);
            };

            // Update the liveness information to reflect the fact that any registers used as
            // operands by this instruction are live above this point in the basic block.
            i.for_operands(|o| {
                if let IlOperand::Register(reg) = o {
                    live_vars.insert(*reg);
                }
            });
        };

        // Remove any no-ops injected into the instruction stream by eliminated instructions. These
        // instructions cannot be removed immediately since doing so would require taking a mutable
        // borrow of g.blocks, which we can't do in the above loop.
        b.instrs.retain(|i| *i != IlInstruction::Nop);
    };

    writeln!(w).unwrap();
    num_eliminated
}

fn try_fold_constant(instr: &mut IlInstruction) -> Option<(IlRegister, IlOperand)> {
    use il::IlConst::*;
    use il::IlInstruction::*;
    use il::IlOperand::*;

    // If the value of this instruction can be calculated, do so. Note that division by 0 is special
    // cased since it causes a runtime error when attempted.
    match *instr {
        AddInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int(r + l)))),
        AddInt(reg, Register(r), Const(Int(0))) => Some((reg, Register(r))),
        AddInt(reg, Const(Int(0)), Register(l)) => Some((reg, Register(l))),
        SubInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int(r - l)))),
        SubInt(reg, Register(r), Const(Int(0))) => Some((reg, Register(r))),
        SubInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(0)))),
        MulInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int(r * l)))),
        MulInt(reg, Register(_), Const(Int(0))) => Some((reg, Const(Int(0)))),
        MulInt(reg, Const(Int(0)), Register(_)) => Some((reg, Const(Int(0)))),
        MulInt(reg, Register(r), Const(Int(1))) => Some((reg, Register(r))),
        MulInt(reg, Const(Int(1)), Register(l)) => Some((reg, Register(l))),
        DivInt(_, _, Const(Int(0))) => None,
        DivInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int(r / l)))),
        DivInt(reg, Register(r), Const(Int(1))) => Some((reg, Register(r))),
        DivInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(1)))),
        LogicNotInt(reg, Const(Int(r))) => Some((reg, Const(Int((r == 0) as i32)))),
        EqInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int((r == l) as i32)))),
        EqInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(1)))),
        LtInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int((r < l) as i32)))),
        LtInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(0)))),
        GtInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int((r > l) as i32)))),
        GtInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(0)))),
        LeInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int((r <= l) as i32)))),
        LeInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(1)))),
        GeInt(reg, Const(Int(r)), Const(Int(l))) => Some((reg, Const(Int((r >= l) as i32)))),
        GeInt(reg, Register(r), Register(l)) if r == l => Some((reg, Const(Int(1)))),
        _ => None
    }
}

fn do_constant_fold(
    g: &mut FlowGraph,
    ipa: &HashMap<usize, IpaStats>,
    w: &mut Write,
    input: &HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>,
    assigns: &mut HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>
) -> u32 {
    use il::IlInstruction::*;
    use il::IlOperand::*;

    writeln!(w, "========== CONSTANT FOLDING AND PROPAGATION ==========\n").unwrap();

    let mut num_substitutions = 0;

    for (_, b) in g.blocks.iter_mut() {
        let input = input.get(&b.id);
        let assigns = assigns.entry(b.id).or_insert_with(HashMap::new);
        assigns.clear();

        for i in &mut b.instrs {
            let old = i.clone();

            // Look at the operands of the current instruction. If the operand is a register whose
            // value is known (either to be a constant or to be a copy of another register), replace
            // the operand.
            i.mutate_operands(|o| {
                if let Register(reg) = o {
                    if let Some(Some(val)) = assigns.get(reg) {
                        mem::replace(o, val.clone());
                    } else if let Some(Some(val)) = input.and_then(|input| input.get(reg)) {
                        mem::replace(o, val.clone());
                    };
                };
            });

            // If this instruction is a call to a function, we need to make sure to invalidate the
            // known values of any nonlocals that the function might write to.
            if let IlInstruction::CallDirect(_, func_id, _) = *i {
                for &sym_id in ipa[&func_id].nonlocal_writes.iter() {
                    if let Some(&reg) = g.registers.locals.get(&sym_id) {
                        assigns.insert(reg, None);
                    };
                };
            };

            // Now look at the instruction itself. If the instruction's result can be computed at
            // compile-time, do so using try_fold_constant. If the instruction is a simple copy of
            // a value (whether a constant or register), keep track of that so that its value can be
            // substituted for uses of the target register later. Otherwise, mark the value of the
            // target register as being unknown at this point in the basic block.
            if let Some((r, c)) = try_fold_constant(i) {
                mem::replace(i, Copy(r, c.clone()));
                assigns.insert(r, Some(c));
            } else if let Copy(r, ref v) = *i {
                assigns.insert(r, Some(v.clone()));
            } else if let Some(r) = i.target_register() {
                assigns.insert(r, None);
            }

            if old != *i {
                writeln!(w, "@{} - {} => {}", b.id, old, i).unwrap();
                num_substitutions += 1;
            }
        }

        for (r, v) in assigns {
            if let Some(v) = v {
                writeln!(w, "@{} - {} <- {}", b.id, r, v).unwrap();
            } else {
                writeln!(w, "@{} - {} <- ???", b.id, r).unwrap();
            }
        }
    }

    writeln!(w).unwrap();
    num_substitutions
}

fn get_jump_constant(
    b: &BasicBlock,
    is_alt: bool
) -> Option<(IlRegister, IlOperand)> {
    use il::IlConst::*;
    use il::IlOperand::*;
    use il::IlInstruction::*;

    // Look at the last instruction of the block to see if we can deduce the value of a register
    // based on the fact that a particular jump was taken.
    if is_alt {
        match b.instrs.last() {
            Some(JumpZero(Register(reg), _)) => Some((*reg, Const(Int(0)))),
            _ => None
        }
    } else {
        match b.instrs.last() {
            Some(JumpNonZero(Register(reg), _)) => Some((*reg, Const(Int(0)))),
            _ => None
        }
    }
}

fn propagate_single_constant(
    next_consts: &mut HashMap<IlRegister, Option<IlOperand>>,
    next_assigns: &HashMap<IlRegister, Option<IlOperand>>,
    reg: &IlRegister,
    val: &Option<IlOperand>
) -> bool {
    if !next_assigns.contains_key(reg) {
        let result = if let Some(val) = val {
            if let Some(old_val) = next_consts.get(reg) {
                if old_val != &None && old_val.as_ref() != Some(val) {
                    // The value coming in through this path was constant, but is different from the
                    // value coming in from another path to this block. Thus, the value of the
                    // register must be marked as unknown.
                    Some(None)
                } else {
                    // The value coming in through this path was consistent with the existing value
                    // coming in from other paths, so no update is required.
                    None
                }
            } else {
                // The value coming in through this path was constant and no other path has given a
                // value for this register. Thus, the value of the register in this block can be
                // temporarily guessed to be the constant we found.
                Some(Some(val.clone()))
            }
        } else {
            if let Some(None) = next_consts.get(reg) {
                // The value coming in through this path was unknown, but this block already has
                // this register marked as unknown, so no update is required.
                None
            } else {
                // The value coming in through this path was unknown and this was inconsistent with
                // the existing value of the register coming into this block. Thus, the value of the
                // register must be marked as unknown.
                Some(None)
            }
        };

        if let Some(new_val) = result {
            next_consts.insert(*reg, new_val);
            true
        } else {
            false
        }
    } else {
        false
    }
}

fn propagate_block_constants(
    g: &FlowGraph,
    output: &mut HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>,
    assigns: &HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>,
    prev: u32,
    next: u32,
    is_alt: bool
) -> bool {
    if prev == next {
        return false;
    };

    let mut changed = false;

    let mut next_consts = output.remove(&next).unwrap_or_else(HashMap::new);
    let next_assigns = &assigns[&next];

    {
        let prev_consts = &output[&prev];
        let prev_assigns = &assigns[&prev];
        let (jmpc_reg, jmpc_val) = get_jump_constant(&g.blocks[&prev], is_alt)
            .map_or((None, None), |(r, v)| (Some(r), Some(v)));

        if let (Some(reg), Some(val)) = (jmpc_reg, jmpc_val) {
            changed = propagate_single_constant(&mut next_consts, next_assigns, &reg, &Some(val)) || changed;
        };

        // Propagate constants from assignments in the preceeding block.
        for (reg, val) in prev_assigns {
            // Do not overwrite the value of jmpc_reg, since its value is known *after* any
            // assignments in this block.
            if jmpc_reg == Some(*reg) {
                continue;
            };

            changed = propagate_single_constant(&mut next_consts, next_assigns, reg, val) || changed;
        };

        for (reg, val) in prev_consts {
            // Do not overwrite the values of any registers whose values are known from the block
            // itself; these values overwrite previous values in the registers, making past
            // information about these registers incorrect.
            if prev_assigns.contains_key(reg) || jmpc_reg == Some(*reg) {
                continue;
            };

            changed = propagate_single_constant(&mut next_consts, next_assigns, reg, val) || changed;
        };
    }

    output.insert(next, next_consts);

    changed
}

fn do_global_propagation(
    g: &FlowGraph,
    w: &mut Write,
    output: &mut HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>,
    assigns: &HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>
) {
    writeln!(w, "========== GLOBAL PROPAGATION ==========\n").unwrap();

    let mut worklist: VecDeque<u32> = VecDeque::new();

    // Start by marking all blocks as inconsistent, since they haven't had a chance to propagate
    // their values yet.
    for (id, _) in &g.blocks {
        worklist.push_back(*id);
    };

    for (_, ref mut consts) in output.iter_mut() {
        consts.clear();
    };

    // Repeatedly propagate known values forward through the CFG until all blocks are consistent.
    // This is guaranteed to happen since the values of registers form a partial ordering and all
    // operations that change them are monotonic.
    while worklist.len() != 0 {
        let b = &g.blocks[&worklist.pop_front().unwrap()];

        writeln!(w, "@{} - propagating constants", b.id).unwrap();

        if let Some(target) = b.successor {
            write!(w, "  to @{}", target).unwrap();
            if propagate_block_constants(g, output, assigns, b.id, target, false) {
                write!(w, " [changed]").unwrap();
                if !worklist.contains(&target) {
                    worklist.push_back(target);
                };
            };
            writeln!(w).unwrap();
        };

        if let Some(target) = b.alt_successor() {
            write!(w, "  to @{}", target).unwrap();
            if propagate_block_constants(g, output, assigns, b.id, target, true) {
                write!(w, " [changed]").unwrap();
                if !worklist.contains(&target) {
                    worklist.push_back(target);
                };
            };
            writeln!(w).unwrap();
        }
    };

    writeln!(w).unwrap();
}

fn do_propagation(g: &mut FlowGraph, ipa: &HashMap<usize, IpaStats>, w: &mut Write) {
    let mut assigns = HashMap::new();
    let mut constants: HashMap<u32, HashMap<IlRegister, Option<IlOperand>>>
        = g.blocks.iter().map(|(id, _)| (*id, HashMap::new())).collect();

    // Start with local constant folding and propagation so that we know the values assigned to each
    // variable in all basic blocks.
    do_constant_fold(g, ipa, w, &constants, &mut assigns);

    // Perform global constant propagation followed by local constant folding and propagation until
    // no more constants are found to propagate.
    loop {
        do_global_propagation(g, w, &mut constants, &assigns);
        if do_constant_fold(g, ipa, w, &constants, &mut assigns) == 0 {
            break;
        };
    };
}

fn do_dead_jump_elimination(g: &mut FlowGraph, w: &mut Write) -> u32 {
    use il::IlConst::*;
    use il::IlInstruction::*;
    use il::IlOperand::*;

    writeln!(w, "========== DEAD JUMP ELIMINATION ==========\n").unwrap();

    let mut num_removed: u32 = 0;

    for (_, b) in &mut g.blocks {
        // Find jumps that are either always taken or are never taken. These include jumps whose
        // values are constants and jumps whose targets are the same as their basic block's
        // fallthrough target (the same target is reached whether or not the jump is taken).
        let jump_info = match b.instrs.last() {
            Some(&JumpZero(Const(Int(0)), target)) => Some(Some(target)),
            Some(&JumpZero(Const(Int(_)), _)) => Some(b.successor),
            Some(&JumpZero(_, target)) if Some(target) == b.successor => Some(b.successor),
            Some(&JumpNonZero(Const(Int(0)), _)) => Some(b.successor),
            Some(&JumpNonZero(Const(Int(_)), target)) => Some(Some(target)),
            Some(&JumpNonZero(_, target)) if Some(target) == b.successor => Some(b.successor),
            _ => None
        };

        // If the jump target of a block is known, eliminate the conditional jump and let the block
        // always fall through to the correct target.
        if let Some(jump_taken) = jump_info {
            num_removed += 1;

            if let Some(jump_taken) = jump_taken {
                writeln!(w, "@{} - always jumps to @{}", b.id, jump_taken).unwrap();
            } else {
                writeln!(w, "@{} - always jumps to @end", b.id).unwrap();
            };

            b.instrs.pop();
            b.successor = jump_taken;
        };
    };

    writeln!(w).unwrap();
    num_removed
}

fn do_dead_block_elimination(g: &mut FlowGraph, w: &mut Write) {
    writeln!(w, "========== DEAD BLOCK ELIMINATION ==========\n").unwrap();

    let mut dead_blocks: HashSet<u32> = g.blocks.keys().map(|id| *id).collect();

    // Perform a simple reachability analysis to find any reachable blocks and remove them from the
    // list of dead blocks.
    fn mark_not_dead(id: u32, g: &FlowGraph, dead_blocks: &mut HashSet<u32>) {
        if dead_blocks.remove(&id) {
            let (s1, s2) = {
                let block = &g.blocks[&id];
                (block.successor, block.alt_successor())
            };

            if let Some(s1) = s1 {
                mark_not_dead(s1, g, dead_blocks);
            };

            if let Some(s2) = s2 {
                mark_not_dead(s2, g, dead_blocks);
            };
        }
    }

    mark_not_dead(g.start_block, g, &mut dead_blocks);

    // Once we've identified all reachable blocks, all remaining blocks in dead_blocks are
    // unreachable and can be removed from the CFG.
    for id in dead_blocks {
        writeln!(w, "@{} - unreachable block eliminated", id).unwrap();
        g.blocks.remove(&id);
    }

    writeln!(w).unwrap();
}

fn do_empty_block_elision(
    g: &mut FlowGraph,
    w: &mut Write
) -> usize {
    writeln!(w, "========== EMPTY BLOCK ELISION ==========\n").unwrap();

    let mut empty_blocks: HashMap<u32, u32> = HashMap::new();

    // Look for any empty blocks and mark them for removal.
    for (_, b) in &g.blocks {
        if b.instrs.len() == 0 {
            if let Some(successor) = b.successor {
                writeln!(w, "@{} - detected empty block, redirecting to @{}", b.id, successor).unwrap();

                empty_blocks.insert(b.id, successor);
            };
        };
    };

    // Look for any jumps targetting an empty block and change them so that they point to the block
    // immediately after the empty block.
    g.start_block = *util::follow(&empty_blocks, &g.start_block);

    for (_, b) in &mut g.blocks {
        if let Some(target) = b.successor {
            b.successor = Some(*util::follow(&empty_blocks, &target));
        };

        if let Some(target) = b.alt_successor() {
            b.relink_alt_successor(*util::follow(&empty_blocks, &target));
        };
    };

    writeln!(w).unwrap();

    empty_blocks.len()
}

fn optimize_function(
    g: &mut FlowGraph,
    ipa: &HashMap<usize, IpaStats>,
    w: &mut Write,
    optimizations: &HashSet<&'static str>
) {
    if optimizations.len() == 0 {
        return;
    };

    // IMPORTANT: The order in which these optimizations are applied can affect the number of
    // optimization passes required, since the application of one optimization can create new
    // opportunities for other optimizations.
    loop {
        let mut updated = false;

        if optimizations.contains("const") {
            do_propagation(g, ipa, w);
        };

        if optimizations.contains("dead-store") {
            let l = build_liveness_graph(g, ipa, w);
            updated = do_dead_store_elimination(g, w, &l) != 0 || updated;
        };

        if optimizations.contains("dead-code") {
            updated = do_empty_block_elision(g, w) != 0 || updated;
            updated = do_dead_jump_elimination(g, w) != 0 || updated;
            do_dead_block_elimination(g, w);
        };

        writeln!(w, "========== CURRENT IL ==========").unwrap();
        writeln!(w, "{}", g).unwrap();

        if !updated {
            break;
        };
    }
}

pub fn optimize_il(program: &mut Program, w: &mut Write, optimizations: &HashSet<&'static str>) {
    perform_ipa(program, w);
    optimize_function(&mut program.main_block, &program.ipa, w, optimizations);

    for &mut (_, ref mut g) in &mut program.funcs {
        optimize_function(g, &program.ipa, w, optimizations);
    };
}

pub fn perform_ipa(program: &mut Program, w: &mut Write) {
    writeln!(w, "========== INTERPROCEDURAL ANALYSIS ==========\n").unwrap();

    fn analyze_function(
        g: &mut FlowGraph,
        stats: &mut IpaStats,
        w: &mut Write
    ) {
        stats.calls.clear();
        stats.nonlocal_refs.clear();
        stats.nonlocal_writes.clear();

        for (_, b) in g.blocks.iter_mut() {
            for i in b.instrs.iter() {
                match *i {
                    IlInstruction::CallDirect(_, func_id, _) => {
                        writeln!(w, "  Found a call to #{}", func_id).unwrap();
                        stats.calls.insert(func_id);
                    },
                    _ => {}
                };

                if let Some(target) = i.target_register() {
                    let reg_meta = &g.registers.reg_meta[&target];

                    if reg_meta.reg_type.is_nonlocal() {
                        let sym_id = reg_meta.sym_id.unwrap();

                        writeln!(w, "  Found a write to nonlocal #{}", sym_id).unwrap();
                        stats.nonlocal_refs.insert(sym_id);
                        stats.nonlocal_writes.insert(sym_id);
                    };
                };

                {
                    let registers = &g.registers;
                    i.for_operands(|o| if let IlOperand::Register(r) = o {
                        let reg_meta = &registers.reg_meta[&r];

                        if reg_meta.reg_type.is_nonlocal() {
                            let sym_id = reg_meta.sym_id.unwrap();

                            writeln!(w, "  Found a read from nonlocal #{}", sym_id).unwrap();
                            stats.nonlocal_refs.insert(sym_id);
                        };
                    });
                };
            };
        }
    }

    for &mut (func_id, ref mut func) in program.funcs.iter_mut() {
        writeln!(w, "Analyzing #{}...", func_id).unwrap();
        analyze_function(
            func,
            program.ipa.entry(func_id).or_insert_with(IpaStats::new),
            w
        );
    };

    writeln!(w, "Propagating calls between functions...").unwrap();
    let func_ids: Vec<_> = program.funcs.iter()
        .map(|&(id, _)| id)
        .collect();

    for &func_id in func_ids.iter() {
        fn append_calls(
            calls: &mut HashSet<usize>,
            next_func: usize,
            ipa: &HashMap<usize, IpaStats>
        ) {
            if calls.insert(next_func) {
                for &called_id in ipa[&next_func].calls.iter() {
                    append_calls(calls, called_id, ipa);
                };
            };
        }

        let mut calls = HashSet::new();

        for called_id in mem::replace(&mut program.ipa.get_mut(&func_id).unwrap().calls, HashSet::new()) {
            append_calls(&mut calls, called_id, &program.ipa);
        };

        writeln!(
            w,
            "  #{} ->{}",
            func_id,
            util::DeferredDisplay(|f| {
                if calls.len() == 0 {
                    write!(f, " none")?;
                } else {
                    for &called_id in calls.iter() {
                        write!(f, " #{}", called_id)?;
                    };
                };
                Result::Ok(())
            })
        ).unwrap();

        mem::replace(&mut program.ipa.get_mut(&func_id).unwrap().calls, calls);
    };

    writeln!(w, "Propagating nonlocal usage between functions...").unwrap();

    for &func_id in func_ids.iter() {
        let mut ipa = program.ipa.remove(&func_id).unwrap();

        for &called_id in ipa.calls.iter() {
            if called_id == func_id { continue; };

            for &nonlocal_ref in program.ipa[&called_id].nonlocal_refs.iter() {
                ipa.nonlocal_refs.insert(nonlocal_ref);
            };

            for &nonlocal_write in program.ipa[&called_id].nonlocal_writes.iter() {
                ipa.nonlocal_writes.insert(nonlocal_write);
            };
        };

        writeln!(
            w,
            "  #{} references{}, writes to{}",
            func_id,
            util::DeferredDisplay(|f| {
                if ipa.nonlocal_refs.len() == 0 {
                    write!(f, " none")?;
                } else {
                    for &sym_id in ipa.nonlocal_refs.iter() {
                        write!(f, " #{}", sym_id)?;
                    };
                };
                Result::Ok(())
            }),
            util::DeferredDisplay(|f| {
                if ipa.nonlocal_writes.len() == 0 {
                    write!(f, " none")?;
                } else {
                    for &sym_id in ipa.nonlocal_writes.iter() {
                        write!(f, " #{}", sym_id)?;
                    };
                };
                Result::Ok(())
            })
        ).unwrap();

        program.ipa.insert(func_id, ipa);
    };

    writeln!(w).unwrap();
}
