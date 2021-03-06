# CPSC 411 - Assignment 5

Author: Benjamin Thomas (10125343)

This is a fully fledged compiler for M++ written in Rust as specified
[here](http://pages.cpsc.ucalgary.ca/%7Erobin/class/411/M+/Mspec.pdf). This compiler includes a
lexer, parser, semantic analyzer, IL generator, IL optimizer, and amd64 code generator.

Unfortunately, this compiler does **not** support generating code for the AM stack machine. This is
a result of that stack machine having some weird behaviour that doesn't match at all with how the IL
represents computations.

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

IL generation can be performed by running `cargo run -- compile --target=il <file>`. This will
generate the IL that is used internally by the compiler and print it to stdout:

```
.local #0 $1 i32
block @0
read.i32 $0
copy $1 $0
copy $3 $1
copy $4 i32:1
add.i32 $2 $3 $4
print.i32 $2
j @end
```

Additionally, optimizations can be enabled by adding the `-O2` option. These optimizations are
generally experimental, so don't expect them to be 100% accurate.

Finally, x86-64 assembly code can be generated by running `cargo run -- compile <file>`, optionally
including the `-O2` option for optimizations to be enabled. This will yield a large blob of x86-64
assembly code in the format used by NASM:

```
[bits 64]
[global main]
[extern printf]
[extern putchar]
[extern getchar]
[extern scanf]
[extern stdin]
[extern fgets]
[extern strcmp]
[extern malloc]
[extern ceil]
[extern floor]
[extern abort]
main:
push rbp
mov rbp, rsp
sub rsp, 16
.L0:
; read.i32 $0
.L0_0_scanf:
lea rsi, [rbp - 4]
mov rdi, __mpp_read_i32
xor eax, eax
call scanf
mov ebx, eax
.L0_0_getchar:
call getchar
cmp eax, 10
jne .L0_0_getchar
cmp ebx, 1
je .L0_0_end
mov rdi, __mpp_invalid_input
xor eax, eax
call printf
jmp .L0_0_scanf
.L0_0_end:
; add.i32 $2 $0 i32:1
lea rdx, [rbp - 4]
mov eax, [rdx]
mov ecx, 1
add eax, ecx
lea rdx, [rbp - 8]
mov [rdx], eax
; print.i32 $2
lea rdx, [rbp - 8]
mov esi, [rdx]
mov rdi, __mpp_print_i32
xor eax, eax
call printf
.Lend:
mov rsp, rbp
pop rbp
ret
[section .rodata]
__mpp_print_i32: db `%d\n\0`
__mpp_print_f64: db `%lf\n\0`
__mpp_print_str: db `%s\0`
__mpp_true: db `true\n\0`
__mpp_false: db `false\n\0`
__mpp_read_i32: db `%d\0`
__mpp_read_f64: db `%lf\0`
__mpp_invalid_input: db `error: invalid input\n\0`
[section .data]
```

This assembly code can be compiled down to machine code by using the
[Netwide Assembler](https://nasm.us/). On x86-64 Linux this can be done by running
`nasm -f elf64 -F dwarf -g -o <file>.o <file>.asm`. This will turn the assembly code into an object
file that can then be linked using any standard ELF executable linker. For instance, to link this
object file with gcc, run `gcc <file>.o -no-pie -lm -g -o <file>`. Once the executable has been
linked, it can be executed as with any other executable: `./<file>`.

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
