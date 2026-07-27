/*
    This file is part of Valkyrie.

    Copyright (C) 2026 Tomasz Jaworski and Valkyrie Contributors

    Valkyrie is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Valkyrie is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Valkyrie.  If not, see <https://gnu.org>.

    Additional terms under GNU GPL version 3 section 7

    1. Linking Exception:
    If you modify this Program, or any covered work, by linking or
    combining it with the ONNX Runtime and/or NVIDIA TensorRT libraries
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective ONNX Runtime and NVIDIA license
    agreements, the licensors of this Program grant you additional
    permission to convey the resulting work.

   2. Attribution Requirement (Section 7b):
    Under Section 7(b) of the GNU General Public License, version 3,
    the licensors of this Program require that any modified or redistributed
    versions of this software must prominently preserve original author
    attributions and credit to the Valkyrie Contributors in the software
    documentation.
*/

use valkyrie_engine::prelude::*;

valkyrie_engine::define_strategy_params! {
    LoggerParams {
        Options {
            ["Minimal"] uci_minimal: bool => false;
        }
    }
}

#[derive(Debug)]
pub struct Logger;

impl Strategy for Logger {
    type Params = LoggerParams;
}

impl<C: EngineConfig> LoggerTrait<C> for Logger 
    where C::BestMove: BestMoveStrategy<C>
{
    #[allow(unused_variables)]
    fn print(msg: &str, _params: &Self::Params, _engine: &Engine<C>) {
        #[cfg(feature = "debug")]
        println!("info string {msg}")
    }

    fn search_report(
        time_passed: u64,
        search_stats: &SearchStats,
        params: &Self::Params,
        engine: &Engine<C>,
        end_of_search: bool,
    ) {
        if params.uci_minimal() && !end_of_search {
            return;
        }

        let (score, pv) = engine.tree().get_pv(0, engine);

        let depth = search_stats.avg_depth();
        let max_depth = search_stats.max_depth();
        let nodes = if engine.params().general().iters_as_nodes() {
            search_stats.iterations()
        } else {
            search_stats.cumulative_depth()
        };
        let nps = nodes as u128 * 1000 / time_passed.max(1) as u128;
        let hashfull = 0;
        let pv_idx = 1;
        let mut pv_str: Vec<String> = Vec::new();

        for mv in pv {
            pv_str.push(mv.to_string(engine.params().general().chess960()));
        }

        println!(
            "info depth {depth} seldepth {max_depth} score cp {score} time {time_passed} nodes {nodes} nps {nps} hashfull {hashfull} multipv {pv_idx} pv {}",
            pv_str.join(" ")
        )
    }

    fn best_move(mv: Move, _params: &Self::Params, engine: &Engine<C>) {
        println!(
            "bestmove {}",
            mv.to_string(engine.params().general().chess960())
        )
    }
}
