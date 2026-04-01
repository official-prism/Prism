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

use prism_engine::{Engine, EngineConfig};

use crate::input_wrapper::InputWrapper;

pub struct UciProcessor;
impl UciProcessor {
    pub fn execute<C: EngineConfig>(
        cmd: &str,
        shutdown_token: &AtomicBool,
        _input_wrapper: &mut InputWrapper,
        engine: &mut Engine<C>,
    ) {
        match cmd {
            "uci" => {
                println!("id name {}", env!("ENGINE_NAME"));
                println!("id author {}", env!("CARGO_PKG_AUTHORS"));
                engine.params().print_options();
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "quit" => shutdown_token.store(true, Ordering::SeqCst),
            _ => {}
        }
    }
}
