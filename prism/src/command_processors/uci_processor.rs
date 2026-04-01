/*
    This file is part of Prism.

    Copyright (C) 2026 Tomasz Jaworski

    Prism is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Prism is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Prism.  If not, see <https://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with the ONNX Runtime and/or NVIDIA TensorRT libraries
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective ONNX Runtime and NVIDIA license
    agreements, the licensors of this Program grant you additional
    permission to convey the resulting work.
*/

use std::sync::atomic::{AtomicBool, Ordering};

use prism_chess::{ChessBoard, ChessPosition, FEN};
use prism_engine::{Engine, EngineConfig};

use crate::input_wrapper::InputWrapper;

pub struct UciProcessor;
impl UciProcessor {
    pub fn execute<C: EngineConfig>(
        cmd: &str,
        shutdown_token: &AtomicBool,
        _input_wrapper: &mut InputWrapper,
        engine: &mut Engine<C>,
    ) -> bool {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();

        match tokens[0] {
            "uci" => Self::uci(engine),
            "isready" => println!("readyok"),
            "position" => Self::position(&tokens[1..], engine),
            "setoption" => Self::set_option(&tokens[1..], engine),
            "quit" | "q" => shutdown_token.store(true, Ordering::SeqCst),
            _ => return false,
        }

        true
    }

    fn uci<C: EngineConfig>(engine: &mut Engine<C>) {
        println!("id name {}", env!("ENGINE_NAME"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
        engine.params().print_options();
        println!("uciok");
    }

    fn position<C: EngineConfig>(args: &[&str], engine: &mut Engine<C>) {
        let mut idx = 0;

        let fen = if args[idx] == "startpos" {
            idx += 1;
            FEN::start_position().to_string()
        } else if args[idx] == "fen" {
            idx += 1;
            let mut fen_parts = Vec::with_capacity(6);
            while idx < args.len() && args[idx] != "moves" {
                fen_parts.push(args[idx]);
                idx += 1;
            }
            fen_parts.join(" ")
        } else {
            println!("info string Invalid position command");
            return;
        };

        if !FEN::validate_fen(&fen) {
            println!("info string Provided FEN is invalid");
            return;
        }

        let mut chess_position = ChessPosition::from(ChessBoard::from(&FEN::from(fen)));

        if idx < args.len() && args[idx] == "moves" {
            idx += 1;
            for &mv_str in &args[idx..] {
                let mut legal_move = false;
                chess_position.board().clone().map_legal_moves(|legal_mv| {
                    legal_move = mv_str == legal_mv.to_string(engine.params().general().ches960());
                    if legal_move {
                        chess_position.make_move_no_mask(legal_mv);
                    }
                });

                if !legal_move {
                    println!("info string Illegal move: {}", mv_str);
                    return;
                }
            }
        }

        engine.set_position(&chess_position);

        println!("info string Position set successfully");
    }

    fn set_option<C: EngineConfig>(args: &[&str], engine: &mut Engine<C>) {
        if args.len() < 2 || args[0] != "name" {
            println!("info string Error: setoption must start with 'name <id>'");
            return;
        }

        let mut name = Vec::new();
        let mut value = Vec::new();
        let mut is_value = false;

        for &token in &args[1..] {
            if token == "value" {
                is_value = true;
                continue;
            }

            if is_value {
                value.push(token);
            } else {
                name.push(token);
            }
        }

        let name_str = name.join(" ");
        let value_str = value.join(" ");

        if let Err(msg) = engine.set_option(&name_str, &value_str) {
            println!("info string {msg}");
            return;
        }

        if is_value {
            println!("info string Option {name_str} has been set to {value_str}");
        } else {
            println!("info string Option {name_str} has been triggered");
        }
    }
}
