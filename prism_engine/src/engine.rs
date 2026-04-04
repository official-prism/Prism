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

pub mod builder;
mod logger;
mod params;
mod structures;

pub use logger::Logger;
pub use params::EngineParams;
pub use params::StrategyParams;

use builder::{BestMoveStrategy, ExplorationStrategy};
use prism_chess::ChessPosition;
use prism_chess::Move;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::engine::structures::SearchLimits;
use crate::engine::structures::SearchStats;

pub trait EngineConfig {
    type BestMove: BestMoveStrategy;
    type Exploration: ExplorationStrategy;
    type Logger: Logger;
}

pub struct GenericConfig<BMS, ES, L> {
    _bms: PhantomData<BMS>,
    _es: PhantomData<ES>,
    _l: PhantomData<L>,
}

impl<BMS: BestMoveStrategy, ES: ExplorationStrategy, L: Logger> EngineConfig
    for GenericConfig<BMS, ES, L>
{
    type BestMove = BMS;
    type Exploration = ES;
    type Logger = L;
}

#[derive(Debug)]
pub struct Engine<C: EngineConfig> {
    params: EngineParams<C>,
    position: ChessPosition,
    interruption_token: AtomicBool,
    _c: PhantomData<C>,
}

impl<C: EngineConfig> Engine<C> {
    #[inline]
    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), String> {
        self.params.set_option(name, value)
    }

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
    pub fn interruption_token(&self) -> bool {
        self.interruption_token.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn interrupt_search(&self) {
        self.interruption_token.store(true, Ordering::Relaxed)
    }

    #[inline]
    #[allow(unused_variables)]
    pub fn print(&self, msg: &str) {
        #[cfg(feature = "debug")]
        C::Logger::print(msg)
    }

    pub fn search(&self, limits: &SearchLimits) -> SearchStats {
        self.interruption_token.store(false, Ordering::Relaxed);

        let search_stats = SearchStats::new();

        self.main_thread_search(limits, &search_stats);

        C::Logger::search_report(&self);
        C::Logger::best_move(Move::NULL, &self);

        search_stats
    }

    fn main_thread_search(&self, _limits: &SearchLimits, _stats: &SearchStats) {
        while !self.interruption_token() {
            //iteration step

            //add iteration
            //print report 

            //test limits

            //check for max tree size
        }
    }
}
