/*
    This file is part of Valkyrie.

    Copyright (C) 2026 Tomasz Jaworski

    Valkyrie is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Valkyrie is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Valkyrie.  If not, see <https://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with the ONNX Runtime and/or NVIDIA TensorRT libraries
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective ONNX Runtime and NVIDIA license
    agreements, the licensors of this Program grant you additional
    permission to convey the resulting work.
*/

use std::time::Instant;

use valkyrie_chess::Move;

use crate::{SearchLimits, SearchStats, prelude::*};

pub(crate) struct IterationStats {
    depth: u64,
}

impl IterationStats {
    pub fn new() -> Self {
        Self { depth: 0 }
    }

    pub fn depth(&self) -> u64 {
        self.depth
    }

    #[allow(dead_code)]
    pub fn add_depth(&mut self) {
        self.depth += 1
    }
}

#[derive(Debug)]
pub struct Classical;

crate::define_strategy_params! {
    ClassicalSearchParams {
    }
}

impl SearchStrategy for Classical {
    type Params = ClassicalSearchParams;

    fn execute<C: crate::EngineConfig>(limits: &SearchLimits, params: &Self::Params, engine: &crate::Engine<C>) -> SearchStats {
        engine.set_interruption_token(false);

        let search_stats = SearchStats::new();
        let search_time = Instant::now();

        Self::main_thread_search(limits, &search_stats, &search_time, params, engine);

        C::Logger::search_report(
            search_time.elapsed().as_millis() as u64,
            &search_stats,
            engine,
        );
        C::Logger::best_move(Move::NULL, engine);

        search_stats
    }
}

type Params = <Classical as SearchStrategy>::Params;
impl Classical {
    fn main_thread_search<C: crate::EngineConfig>(
        limits: &SearchLimits,
        stats: &SearchStats,
        search_time: &Instant,
        params: &Params,
        engine: &crate::Engine<C>
    ) {
        let mut last_raport_time = Instant::now();

        let time_manager = C::TimeManager::new(limits, engine.params().time_manager(), engine);
        let mut main_thread_iters = 0u64;

        while !engine.interruption_token() {
            let iteration_stats = Self::search_step(params, engine);

            let avg_depth = stats.avg_depth();
            let max_depth = stats.max_depth();

            stats.add_iteration(iteration_stats.depth());
            main_thread_iters += 1;

            if stats.avg_depth() > avg_depth
                || stats.max_depth() > max_depth
                || (main_thread_iters.is_multiple_of(128) //todo: maybe remove that
                    && last_raport_time.elapsed().as_millis() >= 1000)
            {
                let time_passed = search_time.elapsed().as_millis() as u64;
                C::Logger::search_report(time_passed, stats, &engine);
                last_raport_time = Instant::now();
            }

            if limits.check_limits(&stats, &engine) {
                engine.set_interruption_token(true);
                break;
            }

            let hash_size = engine.params().general().hash() as usize;
            if engine.tree().tree_full(hash_size) {
                engine.set_interruption_token(true);
                break;
            }

            if main_thread_iters.is_multiple_of(128)
                && time_manager.hard_limit(
                    search_time.elapsed().as_millis() as u64,
                    engine.params().time_manager(),
                    engine,
                )
            {
                engine.set_interruption_token(true);
                break;
            }

            if main_thread_iters.is_multiple_of(4096)
                && time_manager.soft_limit(
                    search_time.elapsed().as_millis() as u64,
                    engine.params().time_manager(),
                    engine,
                )
            {
                engine.set_interruption_token(true);
                break;
            }
        }
    }

    fn search_step<C: crate::EngineConfig>(
        _params: &Params,
        _engine: &crate::Engine<C>
    ) -> IterationStats {
        IterationStats::new()
    }
}