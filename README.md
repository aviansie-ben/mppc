# CPSC 411 - Assignment 3

Author: Benjamin Thomas (10125343)

This is a very basic compiler for M++ written in Rust as specified
[here](http://pages.cpsc.ucalgary.ca/%7Erobin/class/411/M+/Mspec.pdf). Thus far, this compiler
includes a lexer and parser for M++. A partially implemented semantic analyzer also exists, but is
not yet complete and thus should not be used.

The lexer and parser are implemented using a lightly modified version of a library called
[plex](https://github.com/goffrie/plex). Modifications were made to bring that library up-to-date
with a couple of breaking changes in the newer nightly versions of Rust and fix some minor bugs.

## Compiling

In order to properly compile the M++ compiler, the nightly version of Rust must be installed.
While it should theoretically be possible to use future nightly versions, the compiler has thus far
been tested only with the `2018-01-25` version of the Rust compiler and standard library on Linux.

To install the correct version of Rust:

1. Install `rustup` either using your distro's package manager or using the instructions found
   [here](https://www.rust-lang.org/en-US/install.html)
2. Run `rustup install nightly-2018-01-25` to install the correct version of Rust
3. Run `rustup default nightly-2018-01-25` to set the nightly installation as the default (as
   opposed to the stable version)
4. Run `cargo build` inside this directory to build the compiler

Note that there are currently several warnings emitted by the Rust compiler. These are related to
the incomplete implementation of semantic analysis and code generation; these warnings do not affect
any code that will be run when only performing lexing/parsing.

## Running

The compiler can be run in lexer-only mode by running `cargo run -- lex <file>` inside this
directory. Additionally, if no input file is specified, input from stdin will be used. This will
print each token on a separate line, similar to the following:

```
Id("x")
Assign
Id("x")
Add
IVal(1)
Semicolon
```

Note that some tokens are made internal to the lexer and will not be included in the output,
including whitespace and comments. If any characters are found for which there is no valid token,
a single `BadChar` token will be returned. If the lexer encounters other errors while processing
tokens (out-of-range integer, unterminated block comment, etc.), those errors will be displayed once
all other tokens have been printed.

Similarly, the compiler can be run in parser-only mode by running `cargo run -- parse <file>`.
Again, stdin is used by default if no input file is provided. This will print the AST, similar to
the following:

```
Block
 Assign
   Id x
   Add
    Id x
    Int 1
```

Note that internally, the AST has some additional fields that will be used during semantic analysis
to annotate the AST with extra information (e.g. symbols and types). This does not make the AST not
an AST!

## Test Cases

Several test cases have been provided in the `tests/` directory. All of these should parse and lex
properly, though some are designed to fail semantic analysis; these should be noted as comments in
the test file.
