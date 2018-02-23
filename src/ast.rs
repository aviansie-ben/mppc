#[derive(Debug, Clone)]
pub struct Block {
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>
}

impl Block {
    pub fn new(decls: Vec<Decl>, stmts: Vec<Stmt>) -> Block {
        Block { decls: decls, stmts: stmts }
    }
}

#[derive(Debug, Clone)]
pub enum Type {
    Int,
    Real,
    Bool,
    Char,
    Id(String)
}

#[derive(Debug, Clone)]
pub struct VarSpec {
    pub id: String,
    pub dims: Vec<Expr>
}

#[derive(Debug, Clone)]
pub struct FunParam {
    pub id: String,
    pub val_type: Type,
    pub dims: u32
}

impl FunParam {
    pub fn new(id: String, val_type: Type, dims: u32) -> FunParam {
        FunParam { id: id, val_type: val_type, dims: dims }
    }
}

#[derive(Debug, Clone)]
pub struct FunSig {
    pub params: Vec<FunParam>,
    pub return_type: Type
}

impl FunSig {
    pub fn new(params: Vec<FunParam>, return_type: Type) -> FunSig {
        FunSig { params: params, return_type: return_type }
    }
}

#[derive(Debug, Clone)]
pub struct DataCons {
    pub cid: String,
    pub types: Vec<Type>
}

impl DataCons {
    pub fn new(cid: String, types: Vec<Type>) -> DataCons {
        DataCons { cid: cid, types: types }
    }
}

#[derive(Debug, Clone)]
pub enum DeclType {
    Var(Vec<VarSpec>, Type),
    Fun(String, FunSig, Block),
    Data(String, Vec<DataCons>)
}

#[derive(Debug, Clone)]
pub struct Decl {
    pub node: DeclType
}

impl Decl {
    pub fn new(node: DeclType) -> Decl {
        Decl { node: node }
    }

    pub fn var(specs: Vec<VarSpec>, val_type: Type) -> Decl {
        Decl::new(DeclType::Var(specs, val_type))
    }

    pub fn func(name: String, sig: FunSig, body: Block) -> Decl {
        Decl::new(DeclType::Fun(name, sig, body))
    }

    pub fn data(name: String, ctors: Vec<DataCons>) -> Decl {
        Decl::new(DeclType::Data(name, ctors))
    }
}

#[derive(Debug, Clone)]
pub enum ExprType {
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Equal(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Size(String, u32),
    Float(Box<Expr>),
    Floor(Box<Expr>),
    Ceil(Box<Expr>),
    Id(String),
    Call(Box<Expr>, Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Cons(String, Vec<Expr>),
    Int(i32),
    Real(f64),
    Bool(bool),
    Char(char),
    Neg(Box<Expr>)
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub node: ExprType
}

impl Expr {
    pub fn new(node: ExprType) -> Expr {
        Expr { node: node }
    }

    pub fn or(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Or(Box::new(lhs), Box::new(rhs)))
    }

    pub fn and(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::And(Box::new(lhs), Box::new(rhs)))
    }

    pub fn not(val: Expr) -> Expr {
        Expr::new(ExprType::Not(Box::new(val)))
    }

    pub fn equal_to(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Equal(Box::new(lhs), Box::new(rhs)))
    }

    pub fn less_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Lt(Box::new(lhs), Box::new(rhs)))
    }

    pub fn greater_than(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Ge(Box::new(lhs), Box::new(rhs)))
    }

    pub fn less_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Le(Box::new(lhs), Box::new(rhs)))
    }

    pub fn greater_than_or_equal(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Ge(Box::new(lhs), Box::new(rhs)))
    }

    pub fn add(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Add(Box::new(lhs), Box::new(rhs)))
    }

    pub fn sub(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Sub(Box::new(lhs), Box::new(rhs)))
    }

    pub fn mul(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Mul(Box::new(lhs), Box::new(rhs)))
    }

    pub fn div(lhs: Expr, rhs: Expr) -> Expr {
        Expr::new(ExprType::Div(Box::new(lhs), Box::new(rhs)))
    }

    pub fn size_of(id: String, array_derefs: u32) -> Expr {
        Expr::new(ExprType::Size(id, array_derefs))
    }

    pub fn float(val: Expr) -> Expr {
        Expr::new(ExprType::Float(Box::new(val)))
    }

    pub fn floor(val: Expr) -> Expr {
        Expr::new(ExprType::Floor(Box::new(val)))
    }

    pub fn ceil(val: Expr) -> Expr {
        Expr::new(ExprType::Ceil(Box::new(val)))
    }

    pub fn identifier(id: String) -> Expr {
        Expr::new(ExprType::Id(id))
    }

    pub fn call(func: Expr, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Call(Box::new(func), args))
    }

    pub fn index(arr: Expr, ind: Expr) -> Expr {
        Expr::new(ExprType::Index(Box::new(arr), Box::new(ind)))
    }

    pub fn cons(cid: String, args: Vec<Expr>) -> Expr {
        Expr::new(ExprType::Cons(cid, args))
    }

    pub fn int(val: i32) -> Expr {
        Expr::new(ExprType::Int(val))
    }

    pub fn real(val: f64) -> Expr {
        Expr::new(ExprType::Real(val))
    }

    pub fn bool(val: bool) -> Expr {
        Expr::new(ExprType::Bool(val))
    }

    pub fn char(val: char) -> Expr {
        Expr::new(ExprType::Char(val))
    }

    pub fn negate(val: Expr) -> Expr {
        Expr::new(ExprType::Neg(Box::new(val)))
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub cid: String,
    pub vars: Vec<String>,
    pub stmt: Stmt
}

impl Case {
    pub fn new(cid: String, vars: Vec<String>, stmt: Stmt) -> Case {
        Case { cid: cid, vars: vars, stmt: stmt }
    }
}

#[derive(Debug, Clone)]
pub enum StmtType {
    IfThenElse(Expr, Box<Stmt>, Box<Stmt>),
    WhileDo(Expr, Box<Stmt>),
    Read(Expr),
    Assign(Expr, Expr),
    Print(Expr),
    Block(Block),
    Case(Expr, Vec<Case>),
    Return(Expr)
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub node: StmtType
}

impl Stmt {
    pub fn new(node: StmtType) -> Stmt {
        Stmt { node: node }
    }

    pub fn if_then_else(cond: Expr, true_stmt: Stmt, false_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::IfThenElse(cond, Box::new(true_stmt), Box::new(false_stmt)))
    }

    pub fn while_do(cond: Expr, do_stmt: Stmt) -> Stmt {
        Stmt::new(StmtType::WhileDo(cond, Box::new(do_stmt)))
    }

    pub fn read(target: Expr) -> Stmt {
        Stmt::new(StmtType::Read(target))
    }

    pub fn assign(target: Expr, val: Expr) -> Stmt {
        Stmt::new(StmtType::Assign(target, val))
    }

    pub fn print(val: Expr) -> Stmt {
        Stmt::new(StmtType::Print(val))
    }

    pub fn block(block: Block) -> Stmt {
        Stmt::new(StmtType::Block(block))
    }

    pub fn case(val: Expr, cases: Vec<Case>) -> Stmt {
        Stmt::new(StmtType::Case(val, cases))
    }

    pub fn return_value(val: Expr) -> Stmt {
        Stmt::new(StmtType::Return(val))
    }
}
