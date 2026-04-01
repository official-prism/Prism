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

use std::process::Command;

use prism_chess::DEFAULT_PERFT_DEPTH;
use prism_engine::{Engine, EngineConfig};

use crate::number_to_string::{number_to_string, time_to_string};

pub struct MiscProcessor;
impl MiscProcessor {
    pub fn execute<C: EngineConfig>(cmd: &str, engine: &Engine<C>) -> bool {
        let tokens: Vec<&str> = cmd.split_whitespace().collect();

        match tokens[0] {
            "draw" | "d" => engine.position().board().draw_board(),
            "tunables" => engine.params().print_tunables(),
            "perft" => Self::perft::<_, true>(&tokens[1..], engine),
            "perft_no_bulk" => Self::perft::<_, false>(&tokens[1..], engine),
            "bench" => Self::bench(&tokens[1..], engine),
            "clear" | "cls" => Self::clear_terminal_screen(),
            _ => return false,
        }

        true
    }

    fn perft<C: EngineConfig, const BULK: bool>(args: &[&str], engine: &Engine<C>) {
        println!();

        engine.position().board().draw_board();

        let depth = if args.len() > 0 {
            args[0].parse::<u8>().ok()
        } else {
            None
        };

        println!("-----------------------------------------------------------");
        println!("  Running PERFT");
        println!("  Depth: {}", depth.unwrap_or(DEFAULT_PERFT_DEPTH));
        println!("  Bulk: {BULK}");
        println!("  PEXT: {}", cfg!(target_feature = "bmi2"));
        println!("-----------------------------------------------------------\n");

        let (result, duration) = if engine.params().general().ches960() {
            prism_chess::perft::<BULK, true, true>(engine.position().board(), depth)
        } else {
            prism_chess::perft::<BULK, true, false>(engine.position().board(), depth)
        };

        let miliseconds = duration.as_millis().max(1);

        println!("\n-----------------------------------------------------------");
        println!(
            "  Perft ended! {result} nodes, {}, {}n/s",
            time_to_string(miliseconds),
            number_to_string(((result * 1000) as f64 / miliseconds as f64) as u128)
        );
        println!("-----------------------------------------------------------\n");
    }

    fn bench<C: EngineConfig>(args: &[&str], engine: &Engine<C>) {}

    fn clear_terminal_screen() {
        if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(["/c", "cls"])
                .spawn()
                .expect("cls command failed to start")
                .wait()
                .expect("failed to wait");
        } else {
            Command::new("clear")
                .spawn()
                .expect("clear command failed to start")
                .wait()
                .expect("failed to wait");
        };
    }
}
