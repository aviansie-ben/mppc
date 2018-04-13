use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::fmt;
use std::iter::FromIterator;
use std::mem;

use symbol;
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

impl IlType {
    pub fn size(&self) -> u64 {
        use il::IlType::*;
        match *self {
            Int => 4,
            Float => 8,
            Addr => 8
        }
    }
}

impl fmt::Display for IlType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use il::IlType::*;
        match *self {
            Int => write!(f, "i32"),
            Float => write!(f, "f64"),
            Addr => write!(f, "a")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IlRegisterType {
    Temporary,
    Local(bool),
    NonLocal(usize),
    Global
}

impl IlRegisterType {
    pub fn has_nonlocal_references(&self) -> bool {
        use il::IlRegisterType::*;
        match *self {
            Temporary => false,
            Local(has_nonlocal_references) => has_nonlocal_references,
            NonLocal(_) => true,
            Global => true
        }
    }

    pub fn is_nonlocal(&self) -> bool {
        use il::IlRegisterType::*;
        match *self {
            Temporary => false,
            Local(_) => false,
            NonLocal(_) => true,
            Global => true
        }
    }
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
                write!(f, "a:#{}", id)
            } else {
                write!(f, "a:#{}+{}", id, off)
            },
            Addr(v) => write!(f, "a:{}", v)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IlRegister(usize);

impl IlRegister {
    pub fn id(&self) -> usize {
        self.0
    }
}

impl fmt::Display for IlRegister {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.0)
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
    AddFloat(IlRegister, IlOperand, IlOperand),
    SubFloat(IlRegister, IlOperand, IlOperand),
    MulFloat(IlRegister, IlOperand, IlOperand),
    DivFloat(IlRegister, IlOperand, IlOperand),
    AddAddr(IlRegister, IlOperand, IlOperand),
    MulAddr(IlRegister, IlOperand, IlOperand),

    EqInt(IlRegister, IlOperand, IlOperand),
    LtInt(IlRegister, IlOperand, IlOperand),
    GtInt(IlRegister, IlOperand, IlOperand),
    LeInt(IlRegister, IlOperand, IlOperand),
    GeInt(IlRegister, IlOperand, IlOperand),
    EqFloat(IlRegister, IlOperand, IlOperand),
    LtFloat(IlRegister, IlOperand, IlOperand),
    GtFloat(IlRegister, IlOperand, IlOperand),
    LeFloat(IlRegister, IlOperand, IlOperand),
    GeFloat(IlRegister, IlOperand, IlOperand),

    Int2Addr(IlRegister, IlOperand),

    AllocStack(IlRegister, IlOperand),
    FreeStack(IlOperand),
    AllocHeap(IlRegister, IlOperand),

    LoadInt(IlRegister, IlOperand),
    LoadFloat(IlRegister, IlOperand),
    LoadAddr(IlRegister, IlOperand),

    StoreInt(IlOperand, IlOperand),
    StoreFloat(IlOperand, IlOperand),
    StoreAddr(IlOperand, IlOperand),

    PrintInt(IlOperand),
    PrintBool(IlOperand),
    PrintChar(IlOperand),
    PrintFloat(IlOperand),

    ReadInt(IlRegister),
    ReadBool(IlRegister),
    ReadChar(IlRegister),
    ReadFloat(IlRegister),

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
            AddFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            SubFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            MulFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            DivFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            AddAddr(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            MulAddr(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
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
            EqFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            LtFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            GtFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            LeFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            GeFloat(_, ref mut o1, ref mut o2) => {
                f(o1);
                f(o2);
            },
            Int2Addr(_, ref mut o) => f(o),
            AllocStack(_, ref mut o) => f(o),
            FreeStack(ref mut o) => f(o),
            AllocHeap(_, ref mut o) => f(o),
            LoadInt(_, ref mut o) => f(o),
            LoadFloat(_, ref mut o) => f(o),
            LoadAddr(_, ref mut o) => f(o),
            StoreInt(ref mut a, ref mut v) => {
                f(a);
                f(v);
            },
            StoreFloat(ref mut a, ref mut v) => {
                f(a);
                f(v);
            },
            StoreAddr(ref mut a, ref mut v) => {
                f(a);
                f(v);
            },
            PrintInt(ref mut o) => f(o),
            PrintBool(ref mut o) => f(o),
            PrintChar(ref mut o) => f(o),
            PrintFloat(ref mut o) => f(o),
            ReadInt(_) => {},
            ReadBool(_) => {},
            ReadChar(_) => {},
            ReadFloat(_) => {},
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
            AddFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            SubFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            MulFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            DivFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            AddAddr(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            MulAddr(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
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
            EqFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            LtFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            GtFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            LeFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            GeFloat(_, ref o1, ref o2) => {
                f(o1);
                f(o2);
            },
            Int2Addr(_, ref o) => f(o),
            AllocStack(_, ref o) => f(o),
            FreeStack(ref o) => f(o),
            AllocHeap(_, ref o) => f(o),
            LoadInt(_, ref o) => f(o),
            LoadFloat(_, ref o) => f(o),
            LoadAddr(_, ref o) => f(o),
            StoreInt(ref a, ref v) => {
                f(a);
                f(v);
            },
            StoreFloat(ref a, ref v) => {
                f(a);
                f(v);
            },
            StoreAddr(ref a, ref v) => {
                f(a);
                f(v);
            },
            PrintInt(ref o) => f(o),
            PrintBool(ref o) => f(o),
            PrintChar(ref o) => f(o),
            PrintFloat(ref o) => f(o),
            ReadInt(_) => {},
            ReadBool(_) => {},
            ReadChar(_) => {},
            ReadFloat(_) => {},
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
            AddFloat(r, _, _) => Some(r),
            SubFloat(r, _, _) => Some(r),
            MulFloat(r, _, _) => Some(r),
            DivFloat(r, _, _) => Some(r),
            AddAddr(r, _, _) => Some(r),
            MulAddr(r, _, _) => Some(r),
            EqInt(r, _, _) => Some(r),
            LtInt(r, _, _) => Some(r),
            GtInt(r, _, _) => Some(r),
            LeInt(r, _, _) => Some(r),
            GeInt(r, _, _) => Some(r),
            EqFloat(r, _, _) => Some(r),
            LtFloat(r, _, _) => Some(r),
            GtFloat(r, _, _) => Some(r),
            LeFloat(r, _, _) => Some(r),
            GeFloat(r, _, _) => Some(r),
            Int2Addr(r, _) => Some(r),
            AllocStack(r, _) => Some(r),
            FreeStack(_) => None,
            AllocHeap(r, _) => Some(r),
            LoadInt(r, _) => Some(r),
            LoadFloat(r, _) => Some(r),
            LoadAddr(r, _) => Some(r),
            StoreInt(_, _) => None,
            StoreFloat(_, _) => None,
            StoreAddr(_, _) => None,
            PrintInt(_) => None,
            PrintBool(_) => None,
            PrintChar(_) => None,
            PrintFloat(_) => None,
            ReadInt(r) => Some(r),
            ReadBool(r) => Some(r),
            ReadChar(r) => Some(r),
            ReadFloat(r) => Some(r),
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
            AddFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            SubFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            MulFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            DivFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            AddAddr(ref mut old_target, _, _) => mem::replace(old_target, target),
            MulAddr(ref mut old_target, _, _) => mem::replace(old_target, target),
            EqInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            LtInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            GtInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            LeInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            GeInt(ref mut old_target, _, _) => mem::replace(old_target, target),
            EqFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            LtFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            GtFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            LeFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            GeFloat(ref mut old_target, _, _) => mem::replace(old_target, target),
            Int2Addr(ref mut old_target, _) => mem::replace(old_target, target),
            AllocStack(ref mut old_target, _) => mem::replace(old_target, target),
            AllocHeap(ref mut old_target, _) => mem::replace(old_target, target),
            LoadInt(ref mut old_target, _) => mem::replace(old_target, target),
            LoadFloat(ref mut old_target, _) => mem::replace(old_target, target),
            LoadAddr(ref mut old_target, _) => mem::replace(old_target, target),
            ReadInt(ref mut old_target) => mem::replace(old_target, target),
            ReadBool(ref mut old_target) => mem::replace(old_target, target),
            ReadChar(ref mut old_target) => mem::replace(old_target, target),
            ReadFloat(ref mut old_target) => mem::replace(old_target, target),
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
            StoreInt(_, _) => true,
            StoreFloat(_, _) => true,
            StoreAddr(_, _) => true,
            AllocStack(_, _) => true,
            FreeStack(_) => true,
            AllocHeap(_, _) => true,
            PrintInt(_) => true,
            PrintBool(_) => true,
            PrintChar(_) => true,
            PrintFloat(_) => true,
            ReadInt(_) => true,
            ReadBool(_) => true,
            ReadChar(_) => true,
            ReadFloat(_) => true,
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
            AddFloat(ref reg, ref lhs, ref rhs) => write!(f, "add.f64 {} {} {}", reg, lhs, rhs),
            SubFloat(ref reg, ref lhs, ref rhs) => write!(f, "sub.f64 {} {} {}", reg, lhs, rhs),
            MulFloat(ref reg, ref lhs, ref rhs) => write!(f, "mul.f64 {} {} {}", reg, lhs, rhs),
            DivFloat(ref reg, ref lhs, ref rhs) => write!(f, "div.f64 {} {} {}", reg, lhs, rhs),
            AddAddr(ref reg, ref lhs, ref rhs) => write!(f, "add.a {} {} {}", reg, lhs, rhs),
            MulAddr(ref reg, ref lhs, ref rhs) => write!(f, "mul.a {} {} {}", reg, lhs, rhs),
            EqInt(ref reg, ref lhs, ref rhs) => write!(f, "eq.i32 {} {} {}", reg, lhs, rhs),
            LtInt(ref reg, ref lhs, ref rhs) => write!(f, "lt.i32 {} {} {}", reg, lhs, rhs),
            GtInt(ref reg, ref lhs, ref rhs) => write!(f, "gt.i32 {} {} {}", reg, lhs, rhs),
            LeInt(ref reg, ref lhs, ref rhs) => write!(f, "le.i32 {} {} {}", reg, lhs, rhs),
            GeInt(ref reg, ref lhs, ref rhs) => write!(f, "ge.i32 {} {} {}", reg, lhs, rhs),
            EqFloat(ref reg, ref lhs, ref rhs) => write!(f, "eq.f64 {} {} {}", reg, lhs, rhs),
            LtFloat(ref reg, ref lhs, ref rhs) => write!(f, "lt.f64 {} {} {}", reg, lhs, rhs),
            GtFloat(ref reg, ref lhs, ref rhs) => write!(f, "gt.f64 {} {} {}", reg, lhs, rhs),
            LeFloat(ref reg, ref lhs, ref rhs) => write!(f, "le.f64 {} {} {}", reg, lhs, rhs),
            GeFloat(ref reg, ref lhs, ref rhs) => write!(f, "ge.f64 {} {} {}", reg, lhs, rhs),
            Int2Addr(ref reg, ref val) => write!(f, "cvt_a.i32 {} {}", reg, val),
            AllocStack(ref reg, ref size) => write!(f, "alloc_stack {} {}", reg, size),
            FreeStack(ref size) => write!(f, "free_stack {}", size),
            AllocHeap(ref reg, ref size) => write!(f, "alloc_heap {} {}", reg, size),
            LoadInt(ref reg, ref addr) => write!(f, "ld.i32 {} {}", reg, addr),
            LoadFloat(ref reg, ref addr) => write!(f, "ld.f64 {} {}", reg, addr),
            LoadAddr(ref reg, ref addr) => write!(f, "ld.a {} {}", reg, addr),
            StoreInt(ref addr, ref val) => write!(f, "st.i32 {} {}", addr, val),
            StoreFloat(ref addr, ref val) => write!(f, "st.f64 {} {}", addr, val),
            StoreAddr(ref addr, ref val) => write!(f, "st.a {} {}", addr, val),
            PrintInt(ref val) => write!(f, "print.i32 {}", val),
            PrintBool(ref val) => write!(f, "print_b.i32 {}", val),
            PrintChar(ref val) => write!(f, "print_c.i32 {}", val),
            PrintFloat(ref val) => write!(f, "print.f64 {}", val),
            ReadInt(ref reg) => write!(f, "read.i32 {}", reg),
            ReadBool(ref reg) => write!(f, "read_b.i32 {}", reg),
            ReadChar(ref reg) => write!(f, "read_c.i32 {}", reg),
            ReadFloat(ref reg) => write!(f, "read.f64 {}", reg),
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

pub struct RegisterMeta {
    pub val_type: IlType,
    pub reg_type: IlRegisterType,
    pub sym_id: Option<usize>
}

pub struct RegisterAlloc {
    pub args: Vec<IlRegister>,
    pub locals: HashMap<usize, IlRegister>,
    pub reg_meta: HashMap<IlRegister, RegisterMeta>,
    next_reg: usize
}

impl RegisterAlloc {
    pub fn new() -> RegisterAlloc {
        RegisterAlloc {
            args: vec![],
            locals: HashMap::new(),
            reg_meta: HashMap::new(),
            next_reg: 0
        }
    }

    pub fn get_or_alloc_local(&mut self, fun: usize, sym: &symbol::Symbol, val_type: IlType) -> IlRegister {
        match self.locals.entry(sym.id) {
            Entry::Occupied(e) => {
                let reg = *e.get();

                assert_eq!(val_type, self.reg_meta[&reg].val_type);
                reg
            },
            Entry::Vacant(e) => {
                let reg = IlRegister(self.next_reg);

                e.insert(reg);
                self.reg_meta.insert(reg, RegisterMeta {
                    val_type: val_type,
                    reg_type: if sym.defining_fun == !0 && sym.has_nonlocal_references.get() {
                        IlRegisterType::Global
                    } else if sym.defining_fun == fun {
                        IlRegisterType::Local(sym.has_nonlocal_references.get())
                    } else {
                        IlRegisterType::NonLocal(sym.defining_fun)
                    },
                    sym_id: Some(sym.id)
                });
                self.next_reg += 1;

                reg
            }
        }
    }

    pub fn alloc_temp(&mut self, val_type: IlType) -> IlRegister {
        let reg = IlRegister(self.next_reg);

        self.reg_meta.insert(reg, RegisterMeta {
            val_type: val_type,
            reg_type: IlRegisterType::Temporary,
            sym_id: None
        });
        self.next_reg += 1;

        reg
    }
}

pub struct FlowGraph {
    pub blocks: HashMap<u32, BasicBlock>,
    pub start_block: u32,
    next_block: u32,
    pub registers: RegisterAlloc,
    pub calls: HashSet<usize>
}

impl FlowGraph {
    pub fn new() -> FlowGraph {
        FlowGraph {
            blocks: HashMap::new(),
            start_block: 0,
            next_block: 0,
            registers: RegisterAlloc::new(),
            calls: HashSet::new()
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
        for (reg, reg_meta) in &self.registers.reg_meta {
            match reg_meta.reg_type {
                IlRegisterType::Temporary => {},
                IlRegisterType::Local(false) => {
                    write!(f, "\n.local #{} {} {}", reg_meta.sym_id.unwrap(), reg, reg_meta.val_type)?;
                },
                IlRegisterType::Local(true) => {
                    write!(f, "\n.local #{} {} captured {}", reg_meta.sym_id.unwrap(), reg, reg_meta.val_type)?;
                },
                IlRegisterType::NonLocal(func_id) => {
                    write!(f, "\n.nonlocal #{}:#{} {} {}", func_id, reg_meta.sym_id.unwrap(), reg, reg_meta.val_type)?;
                },
                IlRegisterType::Global => {
                    write!(f, "\n.global #{} {} {}", reg_meta.sym_id.unwrap(), reg, reg_meta.val_type)?;
                }
            };
        };

        if self.registers.args.len() > 0 {
            write!(f, "\n.args")?;

            for a in self.registers.args.iter() {
                write!(f, " {}", a)?;
            };
        };

        for id in 0..self.next_block {
            if let Some(block) = self.blocks.get(&id) {
                writeln!(f, "\n{}", block)?;
            }
        };
        Result::Ok(())
    }
}

pub struct IpaStats {
    pub calls: HashSet<usize>,
    pub nonlocal_refs: HashSet<usize>,
    pub nonlocal_writes: HashSet<usize>
}

impl IpaStats {
    pub fn new() -> IpaStats {
        IpaStats {
            calls: HashSet::new(),
            nonlocal_refs: HashSet::new(),
            nonlocal_writes: HashSet::new()
        }
    }
}

pub struct Program {
    pub main_block: FlowGraph,
    pub funcs: Vec<(usize, FlowGraph)>,
    pub ipa: HashMap<usize, IpaStats>
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
