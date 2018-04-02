#![feature(match_default_bindings)]
#![feature(plugin)]
#![feature(vec_remove_item)]
#![plugin(plex)]

extern crate clap;
extern crate itertools;
#[macro_use] extern crate lazy_static;

use clap::{App,AppSettings,Arg,ArgMatches,SubCommand};
use std::collections::HashSet;
use std::fs::{File};
use std::io::{BufRead,BufReader,Write};

pub mod ast;
pub mod codegen;
pub mod il;
#[macro_use] pub mod lex;
pub mod optimize;
pub mod parse;
pub mod symbol;
pub mod util;

use util::PrettyDisplay;

fn create_lexer<'a, T: BufRead>(read: &'a mut T) -> lex::TokenStream<'a> {
    lex::TokenStream::new(lex::Lexer::new(Box::new(move || -> Option<String> {
        let mut line = String::new();

        read.read_line(&mut line)
            .expect("Failed to read line");

        if line.len() > 0 {
            Some(line)
        } else {
            None
        }
    })))
}

fn print_errors(errors: &mut Vec<(String, lex::Span)>) {
    errors.sort_unstable_by_key(|e| e.1);

    for (err, span) in errors {
        println!("error at line {}, col {}: {}", span.lo.line, span.lo.col, err)
    }
}

macro_rules! optimizer_flag {
    ($name:expr, $desc:expr) => {
        [
            Arg::with_name(concat!("opt-", $name, "-on"))
                .long(concat!("opt-", $name))
                .conflicts_with(concat!("opt-", $name, "-off"))
                .help(concat!("Enables ", $desc)),
            Arg::with_name(concat!("opt-", $name, "-off"))
                .long(concat!("opt-no-", $name))
                .conflicts_with(concat!("opt-", $name, "-on"))
                .help(concat!("Disables ", $desc))
        ]
    };
}

static OPTIMIZER_LEVELS: [&[&'static str]; 3] = [
    &[],
    &["const", "dead-store", "dead-code"],
    &["const", "dead-store", "dead-code"]
];

fn get_optimizations<'a>(matches: &ArgMatches<'a>) -> HashSet<&'static str> {
    let mut optimizations: HashSet<&'static str> = HashSet::new();

    for opt in OPTIMIZER_LEVELS[matches.value_of("opt-level").unwrap().parse::<usize>().unwrap()] {
        optimizations.insert(opt);
    }

    for opt in ["const", "dead-store", "dead-code"].into_iter() {
        if matches.occurrences_of(&format!("opt-{}-on", opt)) != 0 {
            optimizations.insert(opt);
        } else if matches.occurrences_of(&format!("opt-{}-off", opt)) != 0 {
            optimizations.remove(opt);
        }
    };

    optimizations
}

fn parse_args<'a>() -> ArgMatches<'a> {
    App::new("M++ Compiler")
        .version("0.1")
        .author("Benjamin Thomas (Aviansie Ben) <ben@benthomas.ca>")
        .about("Compiles M++ code into basic stack machine code")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(SubCommand::with_name("lex")
            .about("Turns M++ code into tokens")
            .arg(Arg::with_name("input")
                .value_name("INPUT FILE")
                .help("Sets the input file to use (stdin is used if no input file is specified)")
                .index(1)
            )
        )
        .subcommand(SubCommand::with_name("parse")
            .about("Turns M++ code into an abstract syntax tree")
            .arg(Arg::with_name("input")
                .value_name("INPUT FILE")
                .help("Sets the input file to use (stdin is used if no input file is specified)")
                .index(1)
            )
        )
        .subcommand(SubCommand::with_name("analyze")
            .about("Analyzes M++ code to create an annotated AST")
            .arg(Arg::with_name("input")
                .value_name("INPUT FILE")
                .help("Sets the input file to use (stdin is used if no input file is specified)")
                .index(1)
            )
        )
        .subcommand(SubCommand::with_name("compile")
            .about("Compiles M++ code")
            .arg(Arg::with_name("input")
                .value_name("INPUT FILE")
                .help("Sets the input file to use (stdin is used if no input file is specified)")
                .index(1)
            )
            .arg(Arg::with_name("target")
                .long("target")
                .takes_value(true)
                .value_name("TARGET ARCH")
                .possible_values(&["stack", "il"])
                .default_value("stack")
                .help("Sets the architecture to generate code for")
            )
            .arg(Arg::with_name("verbose")
                .short("v")
                .help("Enables verbose logging")
            )
            .arg(Arg::with_name("opt-level")
                .short("O")
                .takes_value(true)
                .value_name("LEVEL")
                .possible_values(&["0", "1", "2"])
                .default_value("0")
                .help("Sets the level of optimization to perform")
            )
            .args(&optimizer_flag!("const", "constant folding and propagation"))
            .args(&optimizer_flag!("dead-store", "dead store removal"))
            .args(&optimizer_flag!("dead-code", "dead code removal"))
        )
        .get_matches()
}

fn lex_command<'a>(args: &ArgMatches<'a>) -> () {
    let stdin = std::io::stdin();
    let mut input: Box<BufRead> = if let Some(input) = args.value_of("input") {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(err) => {
                println!("error opening input file: {}", err);
                return;
            }
        }
    } else {
        Box::new(stdin.lock())
    };

    let mut tokens = create_lexer(&mut input);
    let mut errors: Vec<(String, lex::Span)> = Vec::new();

    for (t, _) in tokens.iter() {
        println!("{:?}", t);
    };

    tokens.finish(&mut errors);
    print_errors(&mut errors);
}

fn parse_command<'a>(args: &ArgMatches<'a>) {
    let stdin = std::io::stdin();
    let mut input: Box<BufRead> = if let Some(input) = args.value_of("input") {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(err) => {
                println!("error opening input file: {}", err);
                return;
            }
        }
    } else {
        Box::new(stdin.lock())
    };

    let mut tokens = create_lexer(&mut input);
    let mut errors: Vec<(String, lex::Span)> = Vec::new();
    let parse_result = parse::parse_program(&mut tokens);

    match parse_result {
        Result::Err(ref err) => {
            errors.push(err.clone());
            tokens.drain_tokens();
        },
        Result::Ok(_) => {}
    }

    tokens.finish(&mut errors);
    print_errors(&mut errors);

    if errors.is_empty() {
        println!("{}", parse_result.unwrap().pretty());
    }
}

fn analyze_command<'a>(args: &ArgMatches<'a>) {
    let stdin = std::io::stdin();
    let mut input: Box<BufRead> = if let Some(input) = args.value_of("input") {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(err) => {
                println!("error opening input file: {}", err);
                return;
            }
        }
    } else {
        Box::new(stdin.lock())
    };

    let mut tokens = create_lexer(&mut input);
    let mut errors: Vec<(String, lex::Span)> = Vec::new();
    let mut parse_result = parse::parse_program(&mut tokens);

    match parse_result {
        Result::Err(ref err) => {
            errors.push(err.clone());
            tokens.drain_tokens();
        },
        Result::Ok(ref mut program) => {
            symbol::populate_symbol_tables(program, &mut errors);
        }
    }

    tokens.finish(&mut errors);
    print_errors(&mut errors);

    if errors.is_empty() {
        println!("{}", parse_result.unwrap().pretty());
    }
}

fn compile_stack_command<'a>(args: &ArgMatches<'a>) {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let mut input: Box<BufRead> = if let Some(input) = args.value_of("input") {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(err) => {
                println!("error opening input file: {}", err);
                return;
            }
        }
    } else {
        Box::new(stdin.lock())
    };
    let mut debug_output: Box<Write> = if args.occurrences_of("verbose") != 0 {
        Box::new(stderr.lock())
    } else {
        Box::new(util::NullWriter::new())
    };

    let mut tokens = create_lexer(&mut input);
    let mut errors: Vec<(String, lex::Span)> = Vec::new();
    let parse_result = parse::parse_program(&mut tokens);

    match &parse_result {
        &Result::Err(ref err) => {
            errors.push(err.clone());
            tokens.drain_tokens();
        },
        &Result::Ok(_) => {}
    }

    tokens.finish(&mut errors);
    print_errors(&mut errors);

    if errors.is_empty() {
        let mut g = il::generate_il(parse_result.as_ref().unwrap(), debug_output.as_mut());

        optimize::optimize_il(&mut g, debug_output.as_mut(), &get_optimizations(args));

        // TODO Stack codegen
    }
}

fn compile_il_command<'a>(args: &ArgMatches<'a>) {
    let stdin = std::io::stdin();
    let stderr = std::io::stderr();
    let mut input: Box<BufRead> = if let Some(input) = args.value_of("input") {
        match File::open(input) {
            Ok(f) => Box::new(BufReader::new(f)),
            Err(err) => {
                println!("error opening input file: {}", err);
                return;
            }
        }
    } else {
        Box::new(stdin.lock())
    };
    let mut debug_output: Box<Write> = if args.occurrences_of("verbose") != 0 {
        Box::new(stderr.lock())
    } else {
        Box::new(util::NullWriter::new())
    };

    let mut tokens = create_lexer(&mut input);
    let mut errors: Vec<(String, lex::Span)> = Vec::new();
    let parse_result = parse::parse_program(&mut tokens);

    match &parse_result {
        &Result::Err(ref err) => {
            errors.push(err.clone());
            tokens.drain_tokens();
        },
        &Result::Ok(_) => {}
    }

    tokens.finish(&mut errors);
    print_errors(&mut errors);

    if errors.is_empty() {
        let mut g = il::generate_il(parse_result.as_ref().unwrap(), debug_output.as_mut());

        optimize::optimize_il(&mut g, debug_output.as_mut(), &get_optimizations(args));

        println!("{}", g);
    }
}

fn compile_command<'a>(args: &ArgMatches<'a>) {
    let target = args.value_of("target").unwrap();

    if target == "stack" {
        compile_stack_command(args);
    } else if target == "il" {
        compile_il_command(args);
    }
}

fn main() {
    let args = parse_args();

    if let Some(lex_args) = args.subcommand_matches("lex") {
        lex_command(lex_args);
    } else if let Some(parse_args) = args.subcommand_matches("parse") {
        parse_command(parse_args);
    } else if let Some(analyze_args) = args.subcommand_matches("analyze") {
        analyze_command(analyze_args);
    } else if let Some(compile_args) = args.subcommand_matches("compile") {
        compile_command(compile_args);
    }
}
