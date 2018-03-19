use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::mem;

use ast;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IlConst {
    Int(i32)
}

impl fmt::Display for IlConst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlConst::*;
        match *self {
            Int(v) => write!(f, "i32:{}", v)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IlRegister(pub u32);

impl fmt::Display for IlRegister {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let IlRegister(n) = self;
        write!(f, "${}", n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IlOperand {
    Register(IlRegister),
    Const(IlConst)
}

impl fmt::Display for IlOperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlOperand::*;
        match *self {
            Register(ref reg) => write!(f, "{}", reg),
            Const(ref val) => write!(f, "{}", val)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IlInstruction {
    Input(IlRegister),
    TapeWrite(IlOperand, IlOperand),
    Write(IlOperand),

    JumpNonZero(IlOperand, u32),
    JumpZero(IlOperand, u32),

    Copy(IlRegister, IlOperand),
    Add(IlRegister, IlOperand, IlOperand),
    Sub(IlRegister, IlOperand, IlOperand),
    Mul(IlRegister, IlOperand, IlOperand),
    Div(IlRegister, IlOperand, IlOperand),

    TapeRead(IlRegister, IlOperand),

    Nop
}

impl IlInstruction {
    pub fn relink_jump(&mut self, target: u32) {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(_, ref mut old_target) => mem::replace(old_target, target),
            JumpZero(_, ref mut old_target) => mem::replace(old_target, target),
            _ => panic!("Attempt to relink non-jump instruction {}", &self)
        };
    }

    pub fn jump_target(&self) -> Option<u32> {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(_, ref target) => Some(*target),
            JumpZero(_, ref target) => Some(*target),
            _ => None
        }
    }

    pub fn mutate_operands<TFn>(&mut self, mut f: TFn) where TFn: FnMut (&mut IlOperand) -> () {
        use il::IlInstruction::*;
        match *self {
            Input(_) => {},
            TapeWrite(ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            Write(ref mut o) => f(o),
            JumpNonZero(ref mut o, _) => f(o),
            JumpZero(ref mut o, _) => f(o),
            Copy(_, ref mut o) => f(o),
            Add(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            Sub(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            Mul(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            Div(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            TapeRead(_, ref mut o) => f(o),
            Nop => {}
        }
    }

    pub fn for_operands<TFn>(&self, mut f: TFn) where TFn: FnMut (&IlOperand) -> () {
        use il::IlInstruction::*;
        match *self {
            Input(_) => {},
            TapeWrite(ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            Write(ref o) => f(o),
            JumpNonZero(ref o, _) => f(o),
            JumpZero(ref o, _) => f(o),
            Copy(_, ref o) => f(o),
            Add(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            Sub(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            Mul(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            Div(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            TapeRead(_, ref o) => f(o),
            Nop => {}
        }
    }

    pub fn target_register(&self) -> Option<IlRegister> {
        use il::IlInstruction::*;
        match *self {
            Input(r) => Some(r),
            TapeWrite(_, _) => None,
            Write(_) => None,
            JumpNonZero(_, _) => None,
            JumpZero(_, _) => None,
            Copy(r, _) => Some(r),
            Add(r, _, _) => Some(r),
            Sub(r, _, _) => Some(r),
            Mul(r, _, _) => Some(r),
            Div(r, _, _) => Some(r),
            TapeRead(r, _) => Some(r),
            Nop => None
        }
    }

    pub fn relink_target(&mut self, target: IlRegister) {
        use il::IlInstruction::*;
        match *self {
            Input(ref mut old_target) => mem::replace(old_target, target),
            Copy(ref mut old_target, _) => mem::replace(old_target, target),
            Add(ref mut old_target, _, _) => mem::replace(old_target, target),
            Sub(ref mut old_target, _, _) => mem::replace(old_target, target),
            Mul(ref mut old_target, _, _) => mem::replace(old_target, target),
            Div(ref mut old_target, _, _) => mem::replace(old_target, target),
            TapeRead(ref mut old_target, _) => mem::replace(old_target, target),
            _ => panic!("Attempt to relink target register of non-writing instruction {}", self)
        };
    }

    pub fn has_side_effect(&self) -> bool {
        use il::IlInstruction::*;
        match *self {
            Input(_) => true,
            TapeWrite(_, _) => true,
            Write(_) => true,
            JumpNonZero(_, _) => true,
            JumpZero(_, _) => true,
            Copy(_, _) => false,
            Add(_, _, _) => false,
            Sub(_, _, _) => false,
            Mul(_, _, _) => false,
            Div(_, _, _) => false,
            TapeRead(_, _) => false,
            Nop => false
        }
    }
}

impl fmt::Display for IlInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlInstruction::*;
        match *self {
            Input(ref reg) => write!(f, "input {}", reg),
            TapeWrite(ref addr, ref val) => write!(f, "tape_write {} {}", addr, val),
            Write(ref val) => write!(f, "write {}", val),
            JumpNonZero(ref cond, ref target) => write!(f, "jnz {} @{}", cond, target),
            JumpZero(ref cond, ref target) => write!(f, "jz {} @{}", cond, target),
            Copy(ref reg, ref val) => write!(f, "copy {} {}", reg, val),
            Add(ref reg, ref lhs, ref rhs) => write!(f, "add {} {} {}", reg, lhs, rhs),
            Sub(ref reg, ref lhs, ref rhs) => write!(f, "sub {} {} {}", reg, lhs, rhs),
            Mul(ref reg, ref lhs, ref rhs) => write!(f, "mul {} {} {}", reg, lhs, rhs),
            Div(ref reg, ref lhs, ref rhs) => write!(f, "div {} {} {}", reg, lhs, rhs),
            TapeRead(ref reg, ref addr) => write!(f, "tape_read {} {}", reg, addr),
            Nop => write!(f, "nop")
        }
    }
}

pub struct BasicBlock {
    pub instrs: Vec<IlInstruction>,
    pub successor: Option<u32>,
    pub id: u32
}

impl BasicBlock {
    pub fn new(id: u32) -> BasicBlock {
        BasicBlock {
            instrs: Vec::new(),
            successor: None,
            id: id
        }
    }

    pub fn alt_successor(&self) -> Option<u32> {
        self.instrs.last().and_then(|i| i.jump_target())
    }

    pub fn relink_alt_successor(&mut self, target: u32) {
        self.instrs.last_mut().unwrap().relink_jump(target);
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "block @{}", self.id)?;

        for instr in &self.instrs {
            writeln!(f, "{}", instr)?;
        };

        if let Some(successor) = self.successor {
            write!(f, "j @{}", successor)?;
        } else {
            write!(f, "j @end")?;
        };

        Result::Ok(())
    }
}

pub struct FlowGraph {
    pub blocks: HashMap<u32, BasicBlock>,
    pub locals: HashMap<String, IlRegister>,
    pub start_block: u32,
    next_block: u32,
    next_reg: u32
}

impl FlowGraph {
    pub fn new() -> FlowGraph {
        FlowGraph {
            blocks: HashMap::new(),
            locals: HashMap::new(),
            start_block: 0,
            next_block: 0,
            next_reg: 0
        }
    }

    pub fn append_block(&mut self, b: &mut BasicBlock) -> u32 {
        let mut b2 = BasicBlock::new(self.next_block + 1);
        mem::swap(b, &mut b2);

        assert!(self.next_block == b2.id);
        self.blocks.insert(self.next_block, b2);
        self.next_block += 1;
        self.next_block - 1
    }

    pub fn alloc_reg(&mut self) -> IlRegister {
        self.next_reg += 1;
        IlRegister(self.next_reg - 1)
    }

    pub fn get_or_alloc_local(&mut self, name: &str) -> IlRegister {
        if let Some(reg) = self.locals.get(name).map(|r| *r) {
            reg
        } else {
            let reg = self.alloc_reg();
            self.locals.insert(name.to_string(), reg);
            reg
        }
    }
}

impl fmt::Display for FlowGraph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for id in 0..self.next_block {
            if let Some(block) = self.blocks.get(&id) {
                writeln!(f, "\n{}", block)?;
            }
        };
        Result::Ok(())
    }
}

pub fn generate_il(program: &ast::Program, w: &mut Write) -> FlowGraph {
    let mut g = FlowGraph::new();
    let mut b = BasicBlock::new(0);

    writeln!(w, "========== IL GENERATION ==========\n").unwrap();

    // TODO Generate IL
    g.append_block(&mut b);

    writeln!(w, "\n========== GENERATED IL ==========\n{}", g).unwrap();

    g
}
