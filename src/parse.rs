#![allow(unused_mut)]

use std::fmt;

use lex;
use lex::{Span, Token, TokenStream};

type Error = (String, Span);

pub struct Block {
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "block")
    }
}

parser! {
    fn _parse(Token, Span);

    (a, b) { lex::combine_spans(a, b) }

    block: Block {
        => Block {}
    }
}

fn build_error((bad_tok, bad_span): (Token, Span), err: &'static str) -> Error {
    (format!("{}, but found {}", err, bad_tok), bad_span)
}

pub fn parse_block<'a: 'b, 'b>(tokens: &'b mut TokenStream<'a>)
    -> Result<Block, Error> {
    match _parse(tokens.iter()) {
        Result::Ok(block) => Result::Ok(block),
        Result::Err((None, err)) => Result::Err(build_error(tokens.pop(), err)),
        Result::Err((Some(bad_tok), err)) => Result::Err(build_error(bad_tok, err))
    }
}
