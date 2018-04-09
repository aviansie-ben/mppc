# CPSC 411 - Assignment 4

Author: Benjamin Thomas (10125343)

This is a very basic compiler for M++ written in Rust as specified
[here](http://pages.cpsc.ucalgary.ca/%7Erobin/class/411/M+/Mspec.pdf). Thus far, this compiler
includes a lexer, parser, and semantic analyzer. Experimental support for IL generation for simple
programs is also currently implemented, however it is very much in development and does not factor
into this assignment.

The lexer and parser are implemented using a lightly modified version of a library called
[plex](https://github.com/goffrie/plex). Modifications were made to bring that library up-to-date
with a couple of breaking changes in the newer nightly versions of Rust and fix some minor bugs.
Note that the original version has diverged in the meantime with the addition of procedural macros.

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
 VarDecl int
  VarSpec x
 Assign
  Id x
  Add
   Id x
   Int 1
```

Note that internally, the AST has some additional fields that will be used during semantic analysis
to annotate the AST with extra information (e.g. symbols and types). This does not make the AST not
an AST!

Semantic analysis can be performed by running `cargo run -- analyze <file>`. This will print an
annotated AST with a lot of extra information:

```
Symbol x 0
 Var int
Block
 SymRef x 0
 Assign
  Id x [TYPE: int] [ASSIGNABLE] [SYM: 0]
  Add [TYPE: int] [OP: IntAdd]
   Id x [TYPE: int] [ASSIGNABLE] [SYM: 0]
   Int 1 [TYPE: int]
```

Note that in the event of an error, the analyzer will continue to the best of its ability to analyze
the remainder of the program. It will then print out all error messages, followed by printing the
annotated AST, despite the fact that the annotated AST would not be valid for IL generation. Also,
note that nodes in the annotated AST of a semantically invalid program may be given the _error
type_, represented in the AST as `(error type)`. This type is used to silence further errors that
arise from the same problem in the code and a value of the error type can be used basically
anywhere, but a program containing it will never be valid.

## The Annotated AST

To perform semantic analysis, the existing AST is annotated rather than being used to construct an
entirely new representation. The only part of the AST that changes drastically is that variable,
function, and type declarations are removed from the AST and replaced with definitions in the
relevant tables.

The actual definitions for symbols and types are printed at the top level of the AST. This is done
since they are actually stored in completely separate tables from the ones that control the mapping
of names to symbol and type IDs. Instead, `TypeRef` and `SymRef` notes are printed that refer to
symbols and types previously printed.

All expression nodes contain two mandatory annotations: their result type and whether or not the
expression can be assigned to. This allows the left and right hand sides of an assignment to be
treated uniformly. Note that data and function types are printed as IDs referencing previous type
definitions, like the following: (`(typedef 0)` references the type defined by `TypeDef 0`)

```
TypeDef 0
 FunctionType
  Params
  Returns int
Symbol f 0
 Fun [sig 0]
  Block
   Return
    Int 3 [TYPE: int]
Block
 SymRef f 0
 Print
  Call [TYPE: int]
   Id f [TYPE: (typeref 0)] [SYM: 0]
```

Additionally, a couple specific types of nodes have special annotations that are specific to those
types. For instance, an `Id` expression node has an annotation for the ID of the symbol being
referred to and a number of expression and statement nodes have annotations relating to the ID of a
constructor on a data type. These are generally relatively self-explanatory.

## M++ Extensions

This compiler includes several extensions to the M++ language that can be enabled by putting a
special annotation at the beginning of the file.

Adding `@feature(return_anywhere)` allows return statements to appear anywhere in a function body.
The analyzer will check to make sure that all possible paths return a value. This includes checking
the exhaustiveness of case statements where applicable.

Adding `@feature(optional_else)` allows if statements to omit the else clause. The dangling else
problem is resolved by binding the else clause to the closest available if statement.

Adding `@feature(block_expr)` allows a block of statements followed by an expression to be used
where a simple expression is expected. Such a block would look similar to the following:

```
print ({
    var x : int;
    read x;
    x
});
```

Note that the final expression in the block is **not** terminated by a semicolon. This is on
purpose to differentiate the final expression from a statement. Note also that all statements that
would be valid in the body of the current function are valid inside a block expression, including
return statements.

Adding `@feature(case_expr)` extends pattern matching support to allow case expressions in addition
to case statements. Case expressions are almost identical to case statements, except that they have
expressions as their branches and can appear anywhere an expression is expected. Additionally, they
must be exhaustive (i.e. all constructors must be handled) to avoid an expression not having a valid
value. A case expression might look like the following:

```
data t = #A | #B | #C;
print case #B of {
    #A => 1 |
    #B => 2 |
    #C => 3
};
```

Failing to specify the correct annotation for a feature while using that feature will result in an
error during semantic analysis. These features are heavily experimental and not guaranteed to work
correctly under all circumstances.

## Test Cases

Several test cases have been provided in the `tests/` directory. All of these should parse and lex
properly, though some are designed to fail semantic analysis; these should be noted as comments in
the test file.

## Plumbing Diagrams

Unfortunately, due to the complexity of the logic used during type-checking and the extensive error
handling capabilities of the semantic analyzer of this compiler, I found that it was practically
impossible to draw diagrams for the semantic analyzer I designed.

The main factor that makes this difficult, if not impossible, is the ability for the type of an
expression to be deduced from the type expected in the location in which it appears. This can result
in semantic analysis proceeding up and down the AST more than once as it attempts to deduce what the
type of the expression should be.

Additionally, the imperative nature of the type checker does not lend itself well to the diagrams
shown by Dr. Cockett. I have already spoken with him and I was given the OK to not draw plumbing
diagrams to avoid the complexity. To compensate, the code for the type-checker itself (i.e.
everything in `src/analysis.rs` which deals with expressions) has relatively thorough commenting.
