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
use prism_engine::{EngineConfig, SearchStats, engine::Engine};

#[allow(unused)]
pub struct Logger;
impl prism_engine::Logger for Logger {
    #[allow(unused_variables)]
    fn print<C: EngineConfig>(msg: &str, _engine: &Engine<C>) {
        #[cfg(feature = "debug")]
        println!("info string {msg}")
    }

    fn search_report<C: EngineConfig>(
        time_passed: u64,
        search_stats: &SearchStats,
        engine: &Engine<C>,
    ) {
        let depth = search_stats.avg_depth();
        let max_depth = search_stats.max_depth();
        let score = 0;
        let nodes = if engine.params().general().iters_as_nodes() {
            search_stats.iterations()
        } else {
            search_stats.cumulative_depth()
        };
        let nps = nodes as u128 * 1000 / time_passed.max(1) as u128;
        let hashfull = 0;
        let pv_idx = 1;
        let pv: Vec<String> = Vec::new();

        println!(
            "info depth {depth} seldepth {max_depth} score cp {score} time {time_passed} nodes {nodes} nps {nps} hashfull {hashfull} multipv {pv_idx} pv {}",
            pv.join(" ")
        )
    }

    fn best_move<C: EngineConfig>(mv: Move, engine: &Engine<C>) {
        println!(
            "bestmove {}",
            mv.to_string(engine.params().general().ches960())
        )
    }
}
