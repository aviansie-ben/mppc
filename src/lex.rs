use std::cmp as cmp;
use std::fmt as fmt;

use util;

#[derive(Debug, Clone)]
pub enum Token {
    Add,
    Sub,
    Mul,
    Div,
    Arrow,
    And,
    Or,
    Not,
    Equal,

    Lt,
    Gt,
    Le,
    Ge,

    Assign,

    LPar,
    RPar,
    CLPar,
    CRPar,
    SLPar,
    SRPar,
    Slash,
    Colon,
    Semicolon,
    Comma,

    If,
    Then,
    While,
    Do,
    Read,
    Else,
    Begin,
    End,
    Case,
    Of,
    Print,
    Int,
    Bool,
    Char,
    Real,
    Var,
    Data,
    Size,
    Float,
    Floor,
    Ceil,
    Fun,
    Return,

    CId(String),
    Id(String),
    RVal(f64),
    IVal(i32),
    BVal(bool),
    CVal(char),

    LineComment,
    BlockCommentStart,
    BlockCommentText,
    BlockCommentEnd,

    Whitespace,

    BadChar(char),
    Error(String),
    EndOfFile
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Add => write!(f, "'+'"),
            Token::Sub => write!(f, "'-'"),
            Token::Mul => write!(f, "'*'"),
            Token::Div => write!(f, "'/'"),
            Token::Arrow => write!(f, "'=>'"),
            Token::And => write!(f, "'&&'"),
            Token::Or => write!(f, "'||'"),
            Token::Not => write!(f, "'not'"),

            Token::Equal => write!(f, "'='"),
            Token::Lt => write!(f, "'<'"),
            Token::Gt => write!(f, "'>'"),
            Token::Le => write!(f, "'=<'"),
            Token::Ge => write!(f, "'>='"),

            Token::Assign => write!(f, "':='"),

            Token::LPar => write!(f, "'('"),
            Token::RPar => write!(f, "')'"),
            Token::CLPar => write!(f, "'{{'"),
            Token::CRPar => write!(f, "'}}'"),
            Token::SLPar => write!(f, "'['"),
            Token::SRPar => write!(f, "']'"),
            Token::Slash => write!(f, "'|'"),

            Token::Colon => write!(f, "':'"),
            Token::Semicolon => write!(f, "';'"),
            Token::Comma => write!(f, "','"),

            Token::If => write!(f, "'if'"),
            Token::Then => write!(f, "'then'"),
            Token::While => write!(f, "'while'"),
            Token::Do => write!(f, "'do'"),
            Token::Read => write!(f, "'read'"),
            Token::Else => write!(f, "'else'"),
            Token::Begin => write!(f, "'begin'"),
            Token::End => write!(f, "'end'"),
            Token::Case => write!(f, "'case'"),
            Token::Of => write!(f, "'of'"),
            Token::Print => write!(f, "'print'"),
            Token::Int => write!(f, "'int'"),
            Token::Bool => write!(f, "'bool'"),
            Token::Char => write!(f, "'char'"),
            Token::Real => write!(f, "'real'"),
            Token::Var => write!(f, "'var'"),
            Token::Data => write!(f, "'data'"),
            Token::Size => write!(f, "'size'"),
            Token::Float => write!(f, "'float'"),
            Token::Floor => write!(f, "'floor'"),
            Token::Ceil => write!(f, "'ceil'"),
            Token::Fun => write!(f, "'fun'"),
            Token::Return => write!(f, "'return'"),

            Token::CId(_) => write!(f, "a constructor identifier"),
            Token::Id(_) => write!(f, "an identifier"),
            Token::RVal(_) => write!(f, "a real literal"),
            Token::IVal(_) => write!(f, "an integer literal"),
            Token::BVal(_) => write!(f, "a boolean literal"),
            Token::CVal(_) => write!(f, "a character literal"),

            Token::LineComment | Token::BlockCommentStart | Token::BlockCommentText
                | Token::BlockCommentEnd => write!(f, "a comment"),

            Token::Whitespace => write!(f, "whitespace"),

            Token::Error(ref err) => write!(f, "{}", err),
            Token::BadChar(c) => write!(f, "'{}'", c),
            Token::EndOfFile => write!(f, "end of file")
        }
    }
}

#[macro_export]
macro_rules! bad_token {
    ($actual:expr, $expected:expr) => {{
        let (ref tok, span) = $actual;
        return Result::Err((format!("expected {} but found {}", $expected, tok), span))
    }}
}

#[macro_export]
macro_rules! expect_token {
    ($stream:expr, $expected_name:expr, $expected_pat:pat) => {{
        let tok = $stream.peek().clone();

        if let $expected_pat = tok {
            $stream.pop();
            tok
        } else {
            bad_token!(tok, $expected_name)
        }
    }};
    ($stream:expr, $expected_name:expr, $expected_pat:pat => $result:expr) => {{
        let tok = $stream.peek().clone();

        if let $expected_pat = tok {
            $stream.pop();
            $result
        } else {
            bad_token!(tok, $expected_name)
        }
    }}
}

#[macro_export]
macro_rules! match_pop_token {
    ($stream:expr, $expected_name:expr $(, $expected_pat:pat => $result:expr)+) => {{
        match $stream.peek().clone() {
            $($expected_pat => { $stream.pop(); $result },)+
            t => bad_token!(t, $expected_name)
        }
    }}
}

#[macro_export]
macro_rules! match_peek_token {
    ($stream:expr, $expected_name:expr $(, $expected_pat:pat => $result:expr)+) => {{
        match $stream.peek().clone() {
            $($expected_pat => $result,)+
            t => bad_token!(t, $expected_name)
        }
    }}
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pos {
    pub line: usize,
    pub col: usize
}

impl Pos {
    pub fn dummy() -> Pos {
        Pos { line: usize::max_value(), col: usize::max_value() }
    }
}

impl fmt::Debug for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self == &Pos::dummy() {
            write!(f, "Pos::dummy()")
        } else {
            write!(f, "Pos(line {:?}, col {:?})", self.line, self.col)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    pub lo: Pos,
    pub hi: Pos
}

impl Span {
    pub fn dummy() -> Span {
        Span { lo: Pos::dummy(), hi: Pos::dummy() }
    }

    pub fn combine(s1: Span, s2: Span) -> Span {
        Span {
            lo: cmp::min(s1.lo, s2.lo),
            hi: cmp::max(s1.hi, s2.hi)
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self == &Span::dummy() {
            write!(f, "Span::dummy()")
        } else {
            write!(f, "Span({:?}, {:?})", self.lo, self.hi)
        }
    }
}

lexer! {
    fn next_token_standard(text: 'a) -> (Token, &'a str);

    r#"\+"# => (Token::Add, text),
    r#"-"# => (Token::Sub, text),
    r#"\*"# => (Token::Mul, text),
    r#"/"# => (Token::Div, text),
    r#"=>"# => (Token::Arrow, text),
    r#"\&\&"# => (Token::And, text),
    r#"\|\|"# => (Token::Or, text),
    r#"not"# => (Token::Not, text),

    r#"="# => (Token::Equal, text),
    r#"<"# => (Token::Lt, text),
    r#">"# => (Token::Gt, text),
    r#"=<"# => (Token::Le, text),
    r#">="# => (Token::Ge, text),

    r#":="# => (Token::Assign, text),

    r#"\("# => (Token::LPar, text),
    r#"\)"# => (Token::RPar, text),
    r#"{"# => (Token::CLPar, text),
    r#"}"# => (Token::CRPar, text),
    r#"\["# => (Token::SLPar, text),
    r#"\]"# => (Token::SRPar, text),
    r#"\|"# => (Token::Slash, text),

    r#":"# => (Token::Colon, text),
    r#";"# => (Token::Semicolon, text),
    r#","# => (Token::Comma, text),

    r#"if"# => (Token::If, text),
    r#"then"# => (Token::Then, text),
    r#"while"# => (Token::While, text),
    r#"do"# => (Token::Do, text),
    r#"read"# => (Token::Read, text),
    r#"else"# => (Token::Else, text),
    r#"begin"# => (Token::Begin, text),
    r#"end"# => (Token::End, text),
    r#"case"# => (Token::Case, text),
    r#"of"# => (Token::Of, text),
    r#"print"# => (Token::Print, text),
    r#"int"# => (Token::Int, text),
    r#"bool"# => (Token::Bool, text),
    r#"char"# => (Token::Char, text),
    r#"real"# => (Token::Real, text),
    r#"var"# => (Token::Var, text),
    r#"data"# => (Token::Data, text),
    r#"size"# => (Token::Size, text),
    r#"float"# => (Token::Float, text),
    r#"floor"# => (Token::Floor, text),
    r#"ceil"# => (Token::Ceil, text),
    r#"fun"# => (Token::Fun, text),
    r#"return"# => (Token::Return, text),

    r#"#[a-zA-Z0-9_]*"# => (Token::CId(text[1..].to_owned()), text),
    r#"[a-zA-Z][a-zA-Z0-9_]*"# => (Token::Id(text.to_owned()), text),
    r#"[0-9]+\.[0-9]+"# => if let Result::Ok(val) = text.parse::<f64>() {
        (Token::RVal(val), text)
    } else {
        (Token::Error("invalid real literal".to_string()), text)
    },
    r#"[0-9]+"# => if let Result::Ok(val) = text.parse::<i32>() {
        (Token::IVal(val), text)
    } else {
        (Token::Error("invalid integer literal".to_string()), text)
    },
    r#"true"# => (Token::BVal(true), text),
    r#"false"# => (Token::BVal(false), text),
    r#""\\n""# => (Token::CVal('\n'), text),
    r#""\\t""# => (Token::CVal('\t'), text),
    r#""\\"""# => (Token::CVal('"'), text),
    r#""\\\\""# => (Token::CVal('\\'), text),
    r#""[^"\n\\]""# => (Token::CVal(text.chars().skip(1).next().unwrap()), text),
    r#""[^"\n]*""# => (Token::Error("invalid character literal".to_string()), text),
    r#""[^"\n]*"# => (Token::Error("unterminated character literal".to_string()), text),

    r#"%[^\n]*"# => (Token::LineComment, text),
    r#"/\*"# => (Token::BlockCommentStart, text),

    r#"[ \t\r\n]+"# => (Token::Whitespace, text),

    r#"."# => (Token::BadChar(text.chars().next().unwrap()), text),
}

lexer! {
    fn next_token_block_comment(text: 'a) -> (Token, &'a str);

    r#"%[^\n]*"# => (Token::LineComment, text),
    r#"/\*"# => (Token::BlockCommentStart, text),
    r#"\*/"# => (Token::BlockCommentEnd, text),
    r#"[\*/]"# => (Token::BlockCommentText, text),
    r#"[^%\*/]+"# => (Token::BlockCommentText, text),
}

pub struct Lexer<'a> {
    line_reader: Box<FnMut () -> Option<String> + 'a>,

    line: Option<String>,
    line_offset: usize,

    line_number: usize,
    comment_stack: Vec<Pos>,

    panic_mode: bool,
}

impl <'a> Lexer<'a> {
    pub fn new(mut reader: Box<FnMut () -> Option<String> + 'a>) -> Lexer<'a> {
        let line = reader();
        Lexer {
            line_reader: reader,

            line: line,
            line_offset: 0,

            line_number: 1,
            comment_stack: Vec::new(),

            panic_mode: false,
        }
    }

    fn next_line_token(&mut self) -> Option<(Token, Span)> {
        if let Some(ref line) = self.line {
            let mut rem = &line[self.line_offset..];

            // In order to handle nested block comments, we use a context-sensitive lexer. To
            // accomplish this using only regular expressions, a different set of regexes is used
            // depending on whether we're currently inside a nested comment or not.
            let maybe_tok = if self.comment_stack.is_empty() {
                next_token_standard(&mut rem)
            } else {
                next_token_block_comment(&mut rem)
            };

            if let Some((tok, tok_text)) = maybe_tok {
                // Find the position at which the returned token starts in the input. Since the
                // returned token's text is guaranteed to be a string slice within the current line,
                // we can just use pointer arithmetic to find the offset from the start of the line.
                let offset = tok_text.as_ptr() as usize - line.as_ptr() as usize;
                let lo = Pos {
                    line: self.line_number,
                    col: offset + 1
                };
                let hi = Pos {
                    line: self.line_number,
                    col: offset + tok_text.len() + 1
                };

                // Update the saved line offset so that the next call to next_line_token returns the
                // next token instead of constantly returning the same token.
                self.line_offset = offset + tok_text.len();

                // Update the lexer context if we're entering or exiting a block comment.
                match tok {
                    Token::BlockCommentStart => {
                        self.comment_stack.push(lo);
                    },
                    Token::BlockCommentEnd => {
                        self.comment_stack.pop();
                    },
                    _ => { /* no state changes needed */ }
                };

                Some((tok, Span { lo: lo, hi: hi }))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn eof_token(&mut self) -> Option<(Token, Span)> {
        if let Some(start) = self.comment_stack.pop() {
            Some((
                Token::Error("unterminated block comment".to_string()),
                Span { lo: start, hi: Pos { line: start.line, col: start.col + 2 } }
            ))
        } else {
            Some((
                Token::EndOfFile,
                Span {
                    lo: Pos { line: self.line_number, col: self.line_offset + 1 },
                    hi: Pos { line: self.line_number, col: self.line_offset + 2 }
                }
            ))
        }
    }

    fn next_token(&mut self) -> Option<(Token, Span)> {
        loop {
            // If there are no tokens left in the current line, keep reading lines until we either
            // reach EOF or we find a line containing a token.
            if let Some(tok) = self.next_line_token() {
                break Some(tok);
            } else if self.line == None {
                break self.eof_token();
            } else {
                self.line = (self.line_reader)();

                if self.line == None {
                    break self.eof_token();
                }

                self.line_offset = 0;
                self.line_number += 1;
            };
        }
    }
}

impl <'a> Iterator for Lexer<'a> {
    type Item = (Token, Span);

    fn next(&mut self) -> Option<(Token, Span)> {
        loop {
            if let Some((tok, span)) = self.next_token() {
                match tok {
                    Token::LineComment | Token::BlockCommentStart | Token::BlockCommentText
                        | Token::BlockCommentEnd | Token::Whitespace => {
                        // These tokens are internally generated and not relevant outside the lexer,
                        // so we ignore them and continue to the next token.
                        self.panic_mode = false;
                        continue;
                    },
                    Token::BadChar(_) => {
                        // When encountering a bad character, we want to return only a single error
                        // token. To accomplish this, we enter a "panic mode" after an error and
                        // only exit when we find the next valid token; while in panic mode, we
                        // ignore all further error tokens.
                        if !self.panic_mode {
                            self.panic_mode = true;
                            return Some((tok, span));
                        } else {
                            continue;
                        }
                    },
                    _ => {
                        self.panic_mode = false;
                        return Some((tok, span));
                    }
                }
            } else {
                return None;
            }
        }
    }
}

pub struct TokenStream<'a> {
    iter: util::LookAheadIterator<Lexer<'a>>,
    errors: Vec<(String, Span)>
}

pub struct TokenStreamIter<'a, 'b>(&'a mut TokenStream<'b>) where 'b: 'a;

fn skip_errors(ts: &mut util::LookAheadIterator<Lexer>, errors: &mut Vec<(String, Span)>) {
    // Pop error tokens from the token stream until the next token is no longer an error
    while let &Some((Token::Error(_), _)) = ts.peek() {
        if let Some((Token::Error(err), span)) = ts.pop() {
            errors.push((err, span));
        } else {
            panic!("peek and pop results inconsistent")
        }
    }
}

impl <'a> TokenStream<'a> {
    pub fn new(lexer: Lexer) -> TokenStream {
        let mut errors: Vec<(String, Span)> = Vec::new();
        let mut iter = util::LookAheadIterator::new(lexer);

        skip_errors(&mut iter, &mut errors);

        TokenStream {
            iter: iter,
            errors: errors
        }
    }

    pub fn peek(&self) -> &(Token, Span) {
        self.iter.peek().as_ref().expect("missing eof token")
    }

    pub fn pop(&mut self) -> (Token, Span) {
        let tok = self.iter.pop().expect("missing eof token");
        skip_errors(&mut self.iter, &mut self.errors);

        tok
    }

    pub fn drain_tokens(&mut self) {
        loop {
            if let (Token::EndOfFile, _) = self.pop() {
                break;
            };
        };
    }

    pub fn finish(mut self, errors: &mut Vec<(String, Span)>) -> () {
        match self.pop() {
            (Token::EndOfFile, _) => {}
            (tok, span) => {
                // If there were unused tokens, signal an error...
                self.errors.push((format!("expected end of file but found {}", tok), span));

                // ...and drain all remaining tokens to ensure that any lexical errors are properly
                // detected and added to self.errors
                self.drain_tokens();
            }
        };

        for err in self.errors.drain(..) {
            errors.push(err);
        }
    }

    pub fn iter<'b>(&'b mut self) -> TokenStreamIter<'b, 'a> where 'a: 'b {
        TokenStreamIter(self)
    }
}

impl <'a, 'b> Iterator for TokenStreamIter<'a, 'b> {
    type Item = (Token, Span);

    fn next(&mut self) -> Option<Self::Item> {
        let TokenStreamIter(stream) = self;

        match stream.pop() {
            (Token::EndOfFile, _) => None,
            tok => Some(tok)
        }
    }
}
