use itertools::Itertools;
use std::collections::HashMap;
use std::io::{self, Write};

use il::*;

enum RegLoc {
    Local(i64),
    NonLocal(i64, i64),
    Global(usize)
}

struct Registers<'a> {
    alloc: &'a RegisterAlloc,
    off: HashMap<IlRegister, RegLoc>,
    nonlocals: HashMap<usize, i64>,
    next_local: i64
}

fn emit_register_addr(
    reg: IlRegister,
    dst: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    match registers.off[&reg] {
        RegLoc::Local(off) => {
            if off < 0 {
                writeln!(w, "lea {}, [rbp - {}]", dst, -off)?;
            } else {
                writeln!(w, "lea {}, [rbp + {}]", dst, off)?;
            };
        },
        RegLoc::NonLocal(addr_off, off) => {
            if addr_off < 0 {
                writeln!(w, "mov {}, [rbp - {}]", dst, -addr_off)?;
            } else {
                writeln!(w, "mov {}, [rbp + {}]", dst, addr_off)?;
            };
            writeln!(w, "add {}, {}", dst, off)?;
        },
        RegLoc::Global(sym_id) => {
            writeln!(w, "mov {}, __mpp_global_{}", dst, sym_id)?;
        }
    };
    Result::Ok(())
}

fn emit_operand_read_i32(
    op: &IlOperand,
    dst: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    match *op {
        IlOperand::Const(IlConst::Int(i)) => writeln!(w, "mov {}, {}", dst, i)?,
        IlOperand::Const(ref c) => panic!("invalid constant: {}", c),
        IlOperand::Register(r) => {
            emit_register_addr(r, "rdx", registers, w)?;
            writeln!(w, "mov {}, [rdx]", dst)?;
        }
    }
    Result::Ok(())
}

fn emit_operand_read_f64(
    op: &IlOperand,
    dst: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    match *op {
        IlOperand::Const(IlConst::Float(f)) => {
            writeln!(w, "mov rdx, {}", f.0)?;
            writeln!(w, "movq {}, rdx", dst)?
        },
        IlOperand::Const(ref c) => panic!("invalid constant: {}", c),
        IlOperand::Register(r) => {
            emit_register_addr(r, "rdx", registers, w)?;
            writeln!(w, "movsd {}, [rdx]", dst)?;
        }
    }
    Result::Ok(())
}

fn emit_operand_read_addr(
    op: &IlOperand,
    dst: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    match *op {
        IlOperand::Const(IlConst::Addr(i)) => writeln!(w, "mov {}, {}", dst, i)?,
        IlOperand::Const(ref c) => panic!("invalid constant: {}", c),
        IlOperand::Register(r) => {
            emit_register_addr(r, "rdx", registers, w)?;
            writeln!(w, "mov {}, [rdx]", dst)?;
        }
    }
    Result::Ok(())
}

fn emit_register_write_i32(
    reg: IlRegister,
    src: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    emit_register_addr(reg, "rdx", registers, w)?;
    writeln!(w, "mov [rdx], {}", src)?;

    Result::Ok(())
}

fn emit_register_write_f64(
    reg: IlRegister,
    src: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    emit_register_addr(reg, "rdx", registers, w)?;
    writeln!(w, "movsd [rdx], {}", src)?;

    Result::Ok(())
}

fn emit_register_write_addr(
    reg: IlRegister,
    src: &str,
    registers: &Registers,
    w: &mut Write
) -> io::Result<()> {
    emit_register_addr(reg, "rdx", registers, w)?;
    writeln!(w, "mov [rdx], {}", src)?;

    Result::Ok(())
}

fn emit_instruction(
    i: &IlInstruction,
    n: (u32, usize),
    registers: &Registers,
    all_registers: &HashMap<usize, Registers>,
    w: &mut Write
) -> io::Result<()> {
    use il::IlInstruction::*;

    writeln!(w, "; {}", i)?;

    match *i {
        JumpNonZero(ref o, block) => {
            emit_operand_read_i32(o, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jnz .L{}", block)?;
        },
        JumpZero(ref o, block) => {
            emit_operand_read_i32(o, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jz .L{}", block)?;
        },
        Return(ref o) => {
            match o.get_type(registers.alloc) {
                IlType::Int => emit_operand_read_i32(o, "eax", registers, w)?,
                IlType::Float => emit_operand_read_f64(o, "xmm0", registers, w)?,
                IlType::Addr => emit_operand_read_addr(o, "rax", registers, w)?
            };
        },
        CallDirect(reg, func, ref args) => {
            let other_registers = &all_registers[&func];
            let mut stack_room: i64 = (args.len() + other_registers.nonlocals.len()) as i64 * 8;
            let mut curr_off: i64 = 0;

            if stack_room % 16 != 0 {
                stack_room += 16 - stack_room % 16;
            };

            writeln!(w, "sub rsp, {}", stack_room)?;

            for a in args.iter().rev() {
                match a.get_type(registers.alloc) {
                    IlType::Int => {
                        emit_operand_read_i32(a, "eax", registers, w)?;
                        writeln!(w, "mov [rsp + {}], rax", curr_off)?;
                    },
                    IlType::Float => {
                        emit_operand_read_f64(a, "xmm0", registers, w)?;
                        writeln!(w, "movsd [rsp + {}], xmm0", curr_off)?;
                    },
                    IlType::Addr => {
                        emit_operand_read_addr(a, "rax", registers, w)?;
                        writeln!(w, "mov [rsp + {}], rax", curr_off)?;
                    }
                };
                curr_off += 8;
            };

            for (&nl, &off) in other_registers.nonlocals.iter() {
                if nl == n.1 {
                    writeln!(w, "mov [rsp + {}], rbp", off - 16)?;
                } else {
                    writeln!(w, "mov rdx, [rbp + {}]", registers.nonlocals[&nl])?;
                    writeln!(w, "mov [rsp + {}], rdx", off - 16)?;
                };
            };

            writeln!(w, "call __mpp_func_{}", func)?;

            match registers.alloc.reg_meta[&reg].val_type {
                IlType::Int => emit_register_write_i32(reg, "eax", registers, w)?,
                IlType::Float => emit_register_write_f64(reg, "xmm0", registers, w)?,
                IlType::Addr => emit_register_write_addr(reg, "rax", registers, w)?
            };

            if stack_room > 0 {
                writeln!(w, "add rsp, {}", stack_room)?;
            };
        },
        Copy(reg, ref o) => {
            match o.get_type(registers.alloc) {
                IlType::Int => {
                    emit_operand_read_i32(o, "eax", registers, w)?;
                    emit_register_write_i32(reg, "eax", registers, w)?;
                },
                IlType::Float => {
                    emit_operand_read_f64(o, "xmm0", registers, w)?;
                    emit_register_write_f64(reg, "xmm0", registers, w)?;
                },
                IlType::Addr => {
                    emit_operand_read_addr(o, "rax", registers, w)?;
                    emit_register_write_addr(reg, "rax", registers, w)?;
                }
            };
        },
        AddInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "add eax, ecx")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        SubInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "sub eax, ecx")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        MulInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "imul ecx")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        DivInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "idiv ecx")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LogicNotInt(reg, ref o) => {
            emit_operand_read_i32(o, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "setz al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        AddFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "addsd xmm0, xmm1")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        SubFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "subsd xmm0, xmm1")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        MulFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "mulsd xmm0, xmm1")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        DivFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "divsd xmm0, xmm1")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        AddAddr(reg, ref o1, ref o2) => {
            emit_operand_read_addr(o1, "rax", registers, w)?;
            emit_operand_read_addr(o2, "rcx", registers, w)?;
            writeln!(w, "add rax, rcx")?;
            emit_register_write_addr(reg, "rax", registers, w)?;
        },
        MulAddr(reg, ref o1, ref o2) => {
            emit_operand_read_addr(o1, "rax", registers, w)?;
            emit_operand_read_addr(o2, "rcx", registers, w)?;
            writeln!(w, "mul rcx")?;
            emit_register_write_addr(reg, "rax", registers, w)?;
        },
        CeilFloat(reg, ref o) => {
            emit_operand_read_f64(o, "xmm0", registers, w)?;
            writeln!(w, "call ceil")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        FloorFloat(reg, ref o) => {
            emit_operand_read_f64(o, "xmm0", registers, w)?;
            writeln!(w, "call floor")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        EqInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "cmp eax, ecx")?;
            writeln!(w, "sete al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LtInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "cmp eax, ecx")?;
            writeln!(w, "setb al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        GtInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "cmp eax, ecx")?;
            writeln!(w, "seta al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LeInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "cmp eax, ecx")?;
            writeln!(w, "setbe al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        GeInt(reg, ref o1, ref o2) => {
            emit_operand_read_i32(o1, "eax", registers, w)?;
            emit_operand_read_i32(o2, "ecx", registers, w)?;
            writeln!(w, "cmp eax, ecx")?;
            writeln!(w, "setae al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        EqFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "ucomisd xmm0, xmm1")?;
            writeln!(w, "sete al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LtFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "ucomisd xmm0, xmm1")?;
            writeln!(w, "setb al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        GtFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "ucomisd xmm0, xmm1")?;
            writeln!(w, "seta al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LeFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "ucomisd xmm0, xmm1")?;
            writeln!(w, "setbe al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        GeFloat(reg, ref o1, ref o2) => {
            emit_operand_read_f64(o1, "xmm0", registers, w)?;
            emit_operand_read_f64(o2, "xmm1", registers, w)?;
            writeln!(w, "ucomisd xmm0, xmm1")?;
            writeln!(w, "setae al")?;
            writeln!(w, "and eax, 1")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        Int2Addr(reg, ref o) => {
            emit_operand_read_i32(o, "eax", registers, w)?;
            emit_register_write_addr(reg, "rax", registers, w)?;
        },
        Int2Float(reg, ref o) => {
            emit_operand_read_i32(o, "eax", registers, w)?;
            writeln!(w, "cvtsi2sd xmm0, eax")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        AllocStack(reg, ref size) => {
            emit_operand_read_i32(size, "eax", registers, w)?;
            writeln!(w, "add rax, 23")?;
            writeln!(w, "and rax, -16")?;
            writeln!(w, "sub rsp, rax")?;
            emit_register_write_addr(reg, "rsp", registers, w)?;
        },
        FreeStack(ref size) => {
            emit_operand_read_i32(size, "eax", registers, w)?;
            writeln!(w, "add rax, 23")?;
            writeln!(w, "and rax, -16")?;
            writeln!(w, "add rsp, rax")?;
        },
        AllocHeap(reg, ref size) => {
            emit_operand_read_addr(size, "edi", registers, w)?;
            writeln!(w, "call malloc")?;
            emit_register_write_addr(reg, "rax", registers, w)?;
        },
        LoadInt(reg, ref addr) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            writeln!(w, "mov eax, [rax]")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        LoadFloat(reg, ref addr) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            writeln!(w, "movsd xmm0, [rax]")?;
            emit_register_write_f64(reg, "xmm0", registers, w)?;
        },
        LoadAddr(reg, ref addr) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            writeln!(w, "mov rax, [rax]")?;
            emit_register_write_addr(reg, "rax", registers, w)?;
        },
        StoreInt(ref addr, ref val) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            emit_operand_read_i32(val, "ecx", registers, w)?;
            writeln!(w, "mov [rax], ecx")?;
        },
        StoreFloat(ref addr, ref val) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            emit_operand_read_f64(val, "xmm0", registers, w)?;
            writeln!(w, "movsd [rax], xmm0")?;
        },
        StoreAddr(ref addr, ref val) => {
            emit_operand_read_addr(addr, "rax", registers, w)?;
            emit_operand_read_addr(val, "rcx", registers, w)?;
            writeln!(w, "mov [rax], rcx")?;
        },
        PrintInt(ref val) => {
            emit_operand_read_i32(val, "esi", registers, w)?;
            writeln!(w, "mov rdi, __mpp_print_i32")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call printf")?;
        },
        PrintBool(ref val) => {
            emit_operand_read_i32(val, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "mov rsi, __mpp_true")?;
            writeln!(w, "mov rcx, __mpp_false")?;
            writeln!(w, "cmove rsi, rcx")?;
            writeln!(w, "mov rdi, __mpp_print_str")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call printf")?;
        },
        PrintFloat(ref val) => {
            emit_operand_read_f64(val, "xmm0", registers, w)?;
            writeln!(w, "mov rdi, __mpp_print_f64")?;
            writeln!(w, "mov eax, 1")?;
            writeln!(w, "call printf")?;
        },
        PrintChar(ref val) => {
            emit_operand_read_i32(val, "edi", registers, w)?;
            writeln!(w, "call putchar")?;
        },
        ReadInt(reg) => {
            writeln!(w, ".L{}_{}_scanf:", n.0, n.1)?;
            emit_register_addr(reg, "rsi", registers, w)?;
            writeln!(w, "mov rdi, __mpp_read_i32")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call scanf")?;
            writeln!(w, "mov ebx, eax")?;
            writeln!(w, ".L{}_{}_getchar:", n.0, n.1)?;
            writeln!(w, "call getchar")?;
            writeln!(w, "cmp eax, 10")?;
            writeln!(w, "jne .L{}_{}_getchar", n.0, n.1)?;
            writeln!(w, "cmp ebx, 1")?;
            writeln!(w, "je .L{}_{}_end", n.0, n.1)?;
            writeln!(w, "mov rdi, __mpp_invalid_input")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call printf")?;
            writeln!(w, "jmp .L{}_{}_scanf", n.0, n.1)?;
            writeln!(w, ".L{}_{}_end:", n.0, n.1)?;
        },
        ReadBool(reg) => {
            writeln!(w, "sub rsp, 20")?;
            writeln!(w, "mov rbx, rsp")?;
            writeln!(w, ".L{}_{}_fgets:", n.0, n.1)?;
            writeln!(w, "mov rdx, [stdin]")?;
            writeln!(w, "mov rdi, rbx")?;
            writeln!(w, "mov esi, 20")?;
            writeln!(w, "call fgets")?;
            writeln!(w, "mov byte [rbx + 19], 0")?;
            writeln!(w, "mov rdi, rbx")?;
            writeln!(w, "mov rsi, __mpp_true")?;
            writeln!(w, "call strcmp")?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jnz .L{}_{}_not_true", n.0, n.1)?;
            writeln!(w, "mov eax, 1")?;
            writeln!(w, "jmp .L{}_{}_end", n.0, n.1)?;
            writeln!(w, ".L{}_{}_not_true:", n.0, n.1)?;
            writeln!(w, "mov rdi, rbx")?;
            writeln!(w, "mov rsi, __mpp_false")?;
            writeln!(w, "call strcmp")?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jz .L{}_{}_false", n.0, n.1)?;
            writeln!(w, "mov rdi, __mpp_invalid_input")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call printf")?;
            writeln!(w, "jmp .L{}_{}_fgets", n.0, n.1)?;
            writeln!(w, ".L{}_{}_false:", n.0, n.1)?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, ".L{}_{}_end:", n.0, n.1)?;
            writeln!(w, "add rsp, 20")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        ReadChar(reg) => {
            writeln!(w, "call getchar")?;
            emit_register_write_i32(reg, "eax", registers, w)?;
        },
        ReadFloat(reg) => {
            writeln!(w, ".L{}_{}_scanf:", n.0, n.1)?;
            emit_register_addr(reg, "rsi", registers, w)?;
            writeln!(w, "mov rdi, __mpp_read_f64")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call scanf")?;
            writeln!(w, "mov ebx, eax")?;
            writeln!(w, ".L{}_{}_getchar:", n.0, n.1)?;
            writeln!(w, "call getchar")?;
            writeln!(w, "cmp eax, 10")?;
            writeln!(w, "jne .L{}_{}_getchar", n.0, n.1)?;
            writeln!(w, "cmp ebx, 1")?;
            writeln!(w, "je .L{}_{}_end", n.0, n.1)?;
            writeln!(w, "mov rdi, __mpp_invalid_input")?;
            writeln!(w, "xor eax, eax")?;
            writeln!(w, "call printf")?;
            writeln!(w, "jmp .L{}_{}_scanf", n.0, n.1)?;
            writeln!(w, ".L{}_{}_end:", n.0, n.1)?;
        },
        AssertNonZero(ref val) => {
            emit_operand_read_i32(val, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jnz .L{}_{}_okay", n.0, n.1)?;
            writeln!(w, "call abort")?;
            writeln!(w, ".L{}_{}_okay:", n.0, n.1)?;
        },
        AssertZero(ref val) => {
            emit_operand_read_i32(val, "eax", registers, w)?;
            writeln!(w, "test eax, eax")?;
            writeln!(w, "jz .L{}_{}_okay", n.0, n.1)?;
            writeln!(w, "call abort")?;
            writeln!(w, ".L{}_{}_okay:", n.0, n.1)?;
        },
        Nop => {}
    };

    Result::Ok(())
}

fn emit_basic_block(
    b: &BasicBlock,
    needs_jump: bool,
    registers: &Registers,
    all_registers: &HashMap<usize, Registers>,
    w: &mut Write
) -> io::Result<()> {
    writeln!(w, ".L{}:", b.id)?;

    for (n_instr, i) in b.instrs.iter().enumerate() {
        emit_instruction(i, (b.id, n_instr), registers, all_registers, w)?;
    };

    if needs_jump {
        if let Some(successor) = b.successor {
            writeln!(w, "jmp .L{}", successor)?;
        } else {
            writeln!(w, "jmp .Lend")?;
        };
    };

    Result::Ok(())
}

fn add_register(reg: IlRegister, regs: &mut Registers) {
    use std::collections::hash_map::Entry;
    match regs.off.entry(reg) {
        Entry::Occupied(_) => {},
        Entry::Vacant(e) => {
            let reg_meta = &regs.alloc.reg_meta[&reg];

            match reg_meta.reg_type {
                IlRegisterType::Temporary | IlRegisterType::Local(_) => {
                    if let Some((n, _)) = regs.alloc.args.iter().find_position(|&&r| r == reg) {
                        let off: i64 = (regs.alloc.args.len() - n) as i64 * 8;
                        e.insert(RegLoc::Local(off + 8));
                    } else {
                        regs.next_local -= regs.alloc.reg_meta[&reg].val_type.size() as i64;
                        e.insert(RegLoc::Local(regs.next_local));
                    };
                },
                IlRegisterType::NonLocal(def_fun) => {
                    e.insert(RegLoc::NonLocal(regs.nonlocals[&def_fun], 0));
                },
                IlRegisterType::Global => {
                    e.insert(RegLoc::Global(reg_meta.sym_id.unwrap()));
                }
            };
        }
    }
}

fn find_function_registers<'a>(
    g: &'a FlowGraph,
    ipa: Option<&IpaStats>
) -> Registers<'a> {
    let mut registers = Registers {
        alloc: &g.registers,
        off: HashMap::new(),
        nonlocals: HashMap::new(),
        next_local: 0
    };

    if let Some(ipa) = ipa {
        let mut next_nonlocals = registers.alloc.args.len() as i64 * 8 + 16;
        let nonlocals_from = ipa.nonlocals_from.iter().sorted();

        for &nl in nonlocals_from {
            registers.nonlocals.insert(nl, next_nonlocals);
            next_nonlocals += 8;
        };
    };

    for (_, b) in g.blocks.iter() {
        for i in b.instrs.iter() {
            if let Some(reg) = i.target_register() {
                add_register(reg, &mut registers);
            };

            i.for_operands(|o| if let IlOperand::Register(reg) = *o {
                add_register(reg, &mut registers);
            });
        };
    };

    registers
}

fn populate_nonlocal_offsets(
    registers: &mut Registers,
    all_registers: &mut HashMap<usize, Registers>
) {
    for (&reg, loc) in registers.off.iter_mut() {
        match *loc {
            RegLoc::NonLocal(_, ref mut off) => {
                let reg_meta = &registers.alloc.reg_meta[&reg];
                let sym_id = reg_meta.sym_id.unwrap();
                let def_fun = match reg_meta.reg_type {
                    IlRegisterType::NonLocal(def_fun) => def_fun,
                    _ => unreachable!()
                };

                let other_registers = all_registers.get_mut(&def_fun).unwrap();
                let other_reg = other_registers.alloc.locals[&sym_id];

                add_register(other_reg, other_registers);
                *off = match other_registers.off[&other_reg] {
                    RegLoc::Local(off) => off,
                    _ => unreachable!()
                };
            },
            _ => {}
        };
    };
}

fn emit_function(
    g: &FlowGraph,
    id: usize,
    all_registers: &HashMap<usize, Registers>,
    globals: &mut HashMap<usize, IlType>,
    w: &mut Write
) -> io::Result<()> {
    let registers = &all_registers[&id];
    let blocks = g.blocks.iter().map(|(&id, _)| id).sorted();

    for &r in registers.off.keys() {
        let reg_meta = &registers.alloc.reg_meta[&r];

        match reg_meta.reg_type {
            IlRegisterType::Global => {
                globals.insert(reg_meta.sym_id.unwrap(), reg_meta.val_type);
            },
            _ => {}
        };
    };

    let mut stack_space = -(registers.next_local + 8);

    if stack_space % 16 != 0 {
        stack_space += 16 - stack_space % 16;
    };

    if id == !0 {
        writeln!(w, "main:")?;
    } else {
        writeln!(w, "__mpp_func_{}:", id)?;
    };
    writeln!(w, "push rbp")?;
    writeln!(w, "mov rbp, rsp")?;
    writeln!(w, "sub rsp, {}", stack_space)?;

    for i in 0..blocks.len() {
        let block = &g.blocks[&blocks[i]];
        let needs_jump = block.successor != blocks.get(i + 1).map(|&s| s);

        emit_basic_block(block, needs_jump, registers, all_registers, w)?;
    };

    writeln!(w, ".Lend:")?;
    writeln!(w, "mov rsp, rbp")?;
    writeln!(w, "pop rbp")?;
    writeln!(w, "ret")?;

    Result::Ok(())
}

pub fn emit_program(p: &Program, w: &mut Write) -> io::Result<()> {
    let mut globals = HashMap::new();
    let mut registers = HashMap::new();

    writeln!(w, "[bits 64]")?;
    writeln!(w, "[global main]")?;
    writeln!(w, "[extern printf]")?;
    writeln!(w, "[extern putchar]")?;
    writeln!(w, "[extern getchar]")?;
    writeln!(w, "[extern scanf]")?;
    writeln!(w, "[extern stdin]")?;
    writeln!(w, "[extern fgets]")?;
    writeln!(w, "[extern strcmp]")?;
    writeln!(w, "[extern malloc]")?;
    writeln!(w, "[extern ceil]")?;
    writeln!(w, "[extern floor]")?;
    writeln!(w, "[extern abort]")?;

    registers.insert(!0, find_function_registers(&p.main_block, None));

    for &(id, ref g) in p.funcs.iter() {
        registers.insert(id, find_function_registers(g, p.ipa.get(&id)));
    };

    emit_function(&p.main_block, !0, &registers, &mut globals, w)?;

    for &(id, ref g) in p.funcs.iter() {
        let mut regs = registers.remove(&id).unwrap();
        populate_nonlocal_offsets(&mut regs, &mut registers);
        registers.insert(id, regs);

        emit_function(g, id, &registers, &mut globals, w)?;
    };

    writeln!(w, "[section .rodata]")?;
    writeln!(w, r"__mpp_print_i32: db `%d\n\0`")?;
    writeln!(w, r"__mpp_print_f64: db `%lf\n\0`")?;
    writeln!(w, r"__mpp_print_str: db `%s\0`")?;
    writeln!(w, r"__mpp_true: db `true\n\0`")?;
    writeln!(w, r"__mpp_false: db `false\n\0`")?;
    writeln!(w, r"__mpp_read_i32: db `%d\0`")?;
    writeln!(w, r"__mpp_read_f64: db `%lf\0`")?;
    writeln!(w, r"__mpp_invalid_input: db `error: invalid input\n\0`")?;

    writeln!(w, "[section .data]")?;
    for (sym, t) in globals {
        match t {
            IlType::Int => writeln!(w, "__mpp_global_{}: dd 0", sym)?,
            IlType::Float | IlType::Addr => writeln!(w, "__mpp_global_{}: dq 0", sym)?
        };
    };

    Result::Ok(())
}
