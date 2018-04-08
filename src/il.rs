use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::iter::FromIterator;
use std::mem;

use util::DeferredDisplay;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IlFloat(pub u64);

impl IlFloat {
    pub fn as_f64(&self) -> f64 {
        unsafe { mem::transmute::<u64, f64>(self.0) }
    }

    pub fn from_f64(val: f64) -> IlFloat {
        IlFloat(unsafe { mem::transmute::<f64, u64>(val) })
    }
}

impl fmt::Display for IlFloat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_f64())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IlType {
    Int,
    Float,
    Addr
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlConst {
    Int(i32),
    Float(IlFloat),
    AddrSym(usize, u64),
    Addr(u64)
}

impl IlConst {
    pub fn get_type(&self) -> IlType {
        use il::IlConst::*;
        match *self {
            Int(_) => IlType::Int,
            Float(_) => IlType::Float,
            AddrSym(_, _) => IlType::Addr,
            Addr(_) => IlType::Addr
        }
    }
}

impl fmt::Display for IlConst {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlConst::*;
        match *self {
            Int(v) => write!(f, "i32:{}", v),
            Float(v) => write!(f, "f64:{}", v),
            AddrSym(id, off) => if off == 0 {
                write!(f, "a:${}", id)
            } else {
                write!(f, "a:${}+{}", id, off)
            },
            Addr(v) => write!(f, "a:{}", v)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IlRegister(u32, IlType);

impl IlRegister {
    pub fn id(&self) -> u32 {
        self.0
    }

    pub fn reg_type(&self) -> IlType {
        self.1
    }
}

impl fmt::Display for IlRegister {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let IlRegister(n, _) = self;
        write!(f, "${}", n)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlInstruction {
    JumpNonZero(IlOperand, u32),
    JumpZero(IlOperand, u32),
    Return(IlOperand),
    CallDirect(IlRegister, usize, Vec<IlOperand>),

    Copy(IlRegister, IlOperand),
    AddInt(IlRegister, IlOperand, IlOperand),
    SubInt(IlRegister, IlOperand, IlOperand),
    MulInt(IlRegister, IlOperand, IlOperand),
    DivInt(IlRegister, IlOperand, IlOperand),
    LogicNotInt(IlRegister, IlOperand),

    EqInt(IlRegister, IlOperand, IlOperand),
    LtInt(IlRegister, IlOperand, IlOperand),
    GtInt(IlRegister, IlOperand, IlOperand),
    LeInt(IlRegister, IlOperand, IlOperand),
    GeInt(IlRegister, IlOperand, IlOperand),

    PrintInt(IlOperand),
    PrintBool(IlOperand),
    PrintChar(IlOperand),

    ReadInt(IlRegister),
    ReadBool(IlRegister),
    ReadChar(IlRegister),

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
            JumpNonZero(ref mut o, _) => f(o),
            JumpZero(ref mut o, _) => f(o),
            Return(ref mut o) => f(o),
            CallDirect(_, _, ref mut os) => for o in os {
                f(o);
            },
            Copy(_, ref mut o) => f(o),
            AddInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            SubInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            MulInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            DivInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            LogicNotInt(_, ref mut o) => f(o),
            EqInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            LtInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            GtInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            LeInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            GeInt(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            PrintInt(ref mut o) => f(o),
            PrintBool(ref mut o) => f(o),
            PrintChar(ref mut o) => f(o),
            ReadInt(_) => {},
            ReadBool(_) => {},
            ReadChar(_) => {},
            Nop => {}
        }
    }

    pub fn for_operands<TFn>(&self, mut f: TFn) where TFn: FnMut (&IlOperand) -> () {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(ref o, _) => f(o),
            JumpZero(ref o, _) => f(o),
            Return(ref o) => f(o),
            CallDirect(_, _, ref os) => for o in os {
                f(o);
            },
            Copy(_, ref o) => f(o),
            AddInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            SubInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            MulInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            DivInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            LogicNotInt(_, ref o) => f(o),
            EqInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            LtInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            GtInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            LeInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            GeInt(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            PrintInt(ref o) => f(o),
            PrintBool(ref o) => f(o),
            PrintChar(ref o) => f(o),
            ReadInt(_) => {},
            ReadBool(_) => {},
            ReadChar(_) => {},
            Nop => {}
        }
    }

    pub fn target_register(&self) -> Option<IlRegister> {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(_, _) => None,
            JumpZero(_, _) => None,
            Return(_) => None,
            CallDirect(r, _, _) => Some(r),
            Copy(r, _) => Some(r),
            AddInt(r, _, _) => Some(r),
            SubInt(r, _, _) => Some(r),
            MulInt(r, _, _) => Some(r),
            DivInt(r, _, _) => Some(r),
            LogicNotInt(r, _) => Some(r),
            EqInt(r, _, _) => Some(r),
            LtInt(r, _, _) => Some(r),
            GtInt(r, _, _) => Some(r),
            LeInt(r, _, _) => Some(r),
            GeInt(r, _, _) => Some(r),
            PrintInt(_) => None,
            PrintBool(_) => None,
            PrintChar(_) => None,
            ReadInt(r) => Some(r),
            ReadBool(r) => Some(r),
            ReadChar(r) => Some(r),
            Nop => None
        }
    }

    pub fn relink_target(&mut self, target: IlRegister) {
        use il::IlInstruction::*;
        match *self {
            CallDirect(ref mut old_target, _, _) => mem::replace(old_target, target),
            Copy(ref mut old_target, _) => mem::replace(old_target, target),
            AddInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            SubInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            MulInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            DivInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            LogicNotInt(ref mut old_target, _) => mem::replace(old_target, target),
            EqInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            LtInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            GtInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            LeInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            GeInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            ReadInt(ref mut old_target) => mem::replace(old_target, target),
            ReadBool(ref mut old_target) => mem::replace(old_target, target),
            ReadChar(ref mut old_target) => mem::replace(old_target, target),
            _ => panic!("Attempt to relink target register of non-writing instruction {}", self)
        };
    }

    pub fn has_side_effect(&self) -> bool {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(_, _) => true,
            JumpZero(_, _) => true,
            Return(_) => true,
            CallDirect(_, _, _) => true,
            PrintInt(_) => true,
            PrintBool(_) => true,
            PrintChar(_) => true,
            ReadInt(_) => true,
            ReadBool(_) => true,
            ReadChar(_) => true,
            _ => false
        }
    }
}

impl fmt::Display for IlInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlInstruction::*;
        match *self {
            JumpNonZero(ref cond, ref target) => write!(f, "jnz {} @{}", cond, target),
            JumpZero(ref cond, ref target) => write!(f, "jz {} @{}", cond, target),
            Return(ref val) => write!(f, "ret {}", val),
            CallDirect(ref reg, ref fun, ref args) => write!(
                f,
                "call {} #{}{}",
                reg,
                fun,
                DeferredDisplay(|f| Result::from_iter(args.iter().map(|a| {
                    write!(f, " {}", a)
                })))
            ),
            Copy(ref reg, ref val) => write!(f, "copy {} {}", reg, val),
            AddInt(ref reg, ref lhs, ref rhs) => write!(f, "add.i32 {} {} {}", reg, lhs, rhs),
            SubInt(ref reg, ref lhs, ref rhs) => write!(f, "sub.i32 {} {} {}", reg, lhs, rhs),
            MulInt(ref reg, ref lhs, ref rhs) => write!(f, "mul.i32 {} {} {}", reg, lhs, rhs),
            DivInt(ref reg, ref lhs, ref rhs) => write!(f, "div.i32 {} {} {}", reg, lhs, rhs),
            LogicNotInt(ref reg, ref val) => write!(f, "lnot.i32 {} {}", reg, val),
            EqInt(ref reg, ref lhs, ref rhs) => write!(f, "eq.i32 {} {} {}", reg, lhs, rhs),
            LtInt(ref reg, ref lhs, ref rhs) => write!(f, "lt.i32 {} {} {}", reg, lhs, rhs),
            GtInt(ref reg, ref lhs, ref rhs) => write!(f, "gt.i32 {} {} {}", reg, lhs, rhs),
            LeInt(ref reg, ref lhs, ref rhs) => write!(f, "le.i32 {} {} {}", reg, lhs, rhs),
            GeInt(ref reg, ref lhs, ref rhs) => write!(f, "ge.i32 {} {} {}", reg, lhs, rhs),
            PrintInt(ref val) => write!(f, "print.i32 {}", val),
            PrintBool(ref val) => write!(f, "print_b.i32 {}", val),
            PrintChar(ref val) => write!(f, "print_c.i32 {}", val),
            ReadInt(ref reg) => write!(f, "read.i32 {}", reg),
            ReadBool(ref reg) => write!(f, "read_b.i32 {}", reg),
            ReadChar(ref reg) => write!(f, "read_c.i32 {}", reg),
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

pub struct RegisterAlloc {
    pub locals: HashMap<usize, IlRegister>,
    next_reg: u32
}

impl RegisterAlloc {
    pub fn new() -> RegisterAlloc {
        RegisterAlloc {
            locals: HashMap::new(),
            next_reg: 0
        }
    }

    pub fn get_or_alloc_local(&mut self, id: usize, reg_type: IlType) -> IlRegister {
        match self.locals.entry(id) {
            Entry::Occupied(e) => {
                assert_eq!(reg_type, e.get().1);
                *e.get()
            },
            Entry::Vacant(e) => {
                let reg = IlRegister(self.next_reg, reg_type);

                e.insert(reg);
                self.next_reg += 1;

                reg
            }
        }
    }

    pub fn alloc_temp(&mut self, reg_type: IlType) -> IlRegister {
        let reg = IlRegister(self.next_reg, reg_type);

        self.next_reg += 1;

        reg
    }
}

pub struct FlowGraph {
    pub blocks: HashMap<u32, BasicBlock>,
    pub start_block: u32,
    next_block: u32,
    pub registers: RegisterAlloc
}

impl FlowGraph {
    pub fn new() -> FlowGraph {
        FlowGraph {
            blocks: HashMap::new(),
            start_block: 0,
            next_block: 0,
            registers: RegisterAlloc::new()
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

pub struct Program {
    pub main_block: FlowGraph,
    pub funcs: Vec<(usize, FlowGraph)>
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{}", self.main_block)?;

        for &(id, ref g) in &self.funcs {
            writeln!(f, "\nFUNCTION #{}:\n{}", id, g)?;
        };

        Result::Ok(())
    }
}
