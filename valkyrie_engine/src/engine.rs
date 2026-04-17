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

pub mod builder;
mod logger;
mod params;
mod structures;
mod tree;

pub use logger::LoggerTrait;
pub use params::EngineParams;
pub use params::StrategyParams;
pub use structures::SearchLimits;
pub use structures::SearchStats;

use valkyrie_chess::ChessPosition;
use valkyrie_chess::Move;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::engine::builder::EngineConfig;
use crate::engine::builder::TimeManagerStrategy;
use crate::engine::builder::search_step_strategy::SearchStepStrategy;
use crate::engine::tree::Tree;

#[derive(Debug)]
pub struct Engine<C: EngineConfig> {
    params: EngineParams<C>,
    position: ChessPosition,
    interruption_token: AtomicBool,
    tree: Tree<C::NodePayload, C::EdgePayload>,
    _c: PhantomData<C>,
}

impl<C: EngineConfig> Engine<C> {
    #[inline]
    pub fn params(&self) -> &EngineParams<C> {
        &self.params
    }

    #[inline]
    pub fn params_mut(&mut self) -> &mut EngineParams<C> {
        &mut self.params
    }

    #[inline]
    pub fn position(&self) -> &ChessPosition {
        &self.position
    }

    #[inline]
    pub fn set_position(&mut self, position: &ChessPosition) {
        self.position = *position;
    }

    #[inline]
    pub fn tree(&self) -> &Tree<C::NodePayload, C::EdgePayload> {
        &self.tree
    }

    #[inline]
    pub fn tree_mut(&mut self) -> &mut Tree<C::NodePayload, C::EdgePayload> {
        &mut self.tree
    }

    #[inline]
    pub fn interruption_token(&self) -> bool {
        self.interruption_token.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn interrupt_search(&self) {
        self.interruption_token.store(true, Ordering::Relaxed)
    }

    #[inline]
    pub fn print(&self, msg: &str) {
        C::Logger::print(msg, self)
    }

    pub fn search(&self, limits: &SearchLimits) -> SearchStats {
        self.interruption_token.store(false, Ordering::Relaxed);

        let search_stats = SearchStats::new();
        let search_time = Instant::now();

        self.main_thread_search(limits, &search_stats, &search_time);

        C::Logger::search_report(
            search_time.elapsed().as_millis() as u64,
            &search_stats,
            &self,
        );
        C::Logger::best_move(Move::NULL, &self);

        search_stats
    }

    fn main_thread_search(
        &self,
        limits: &SearchLimits,
        stats: &SearchStats,
        search_time: &Instant,
    ) {
        let mut last_raport_time = Instant::now();

        let time_manager = C::TimeManager::new(limits, self.params().time_manager(), self);
        let mut main_thread_iters = 0u64;

        while !self.interruption_token() {
            let iteration_stats = C::SearchStep::excute(self.params().search(), &self);

            let avg_depth = stats.avg_depth();
            let max_depth = stats.max_depth();

            stats.add_iteration(&iteration_stats);
            main_thread_iters += 1;

            if stats.avg_depth() > avg_depth
                || stats.max_depth() > max_depth
                || (main_thread_iters.is_multiple_of(128) //todo: maybe remove that
                    && last_raport_time.elapsed().as_millis() >= 1000)
            {
                let time_passed = search_time.elapsed().as_millis() as u64;
                C::Logger::search_report(time_passed, stats, &self);
                last_raport_time = Instant::now();
            }

            if limits.check_limits(&stats, &self) {
                self.interrupt_search();
                break;
            }

            let hash_size = self.params().general().hash() as usize;
            if self.tree().tree_full(hash_size) {
                self.interrupt_search();
                break;
            }

            if main_thread_iters.is_multiple_of(128)
                && time_manager.hard_limit(
                    search_time.elapsed().as_millis() as u64,
                    self.params().time_manager(),
                    self,
                )
            {
                self.interrupt_search();
                break;
            }

            if main_thread_iters.is_multiple_of(4096)
                && time_manager.soft_limit(
                    search_time.elapsed().as_millis() as u64,
                    self.params().time_manager(),
                    self,
                )
            {
                self.interrupt_search();
                break;
            }
        }
    }
}
