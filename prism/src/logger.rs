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

use prism_chess::Move;
use prism_engine::{EngineConfig, engine::Engine};

#[allow(unused)]
pub struct Logger;
impl prism_engine::Logger for Logger {
    #[allow(unused_variables)]
    fn print<C: EngineConfig>(msg: &str, _engine: &Engine<C>) {
        #[cfg(feature = "debug")]
        println!("info string {msg}")
    }

    fn search_report<C: EngineConfig>(_engine: &Engine<C>) {
        println!("info depth xyz seldepth xyz score cp xyz time xyz nodes xyz nps xyz hashfull xyz multipv x pv xyz xyz xyz xyz")
    }

    fn best_move<C: EngineConfig>(mv: Move, engine: &Engine<C>) {
        println!("bestmove {}", mv.to_string(engine.params().general().ches960()))
    }
}