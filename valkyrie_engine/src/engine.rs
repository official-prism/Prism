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

mod macros;

pub mod builder;
pub mod tree;

mod logger;
mod search_limits;
mod search_stats;
mod strategy_params;

pub use builder::{EngineBuilder, EngineConfig, EngineParams, GeneralParams, SearchStrategy};
pub use logger::{LoggerTrait, NoLogger};
pub use search_limits::SearchLimits;
pub use search_stats::SearchStats;
pub use strategy_params::{EmptyParams, OptionError, StrategyParams, UciOptionType};

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use valkyrie_chess::ChessPosition;

use crate::engine::tree::Tree;

#[derive(Debug)]
pub struct Engine<C: EngineConfig> {
    params: EngineParams<C>,
    position: ChessPosition,
    interruption_token: AtomicBool,
    tree: Tree<C::Node>,
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
    pub fn tree(&self) -> &Tree<C::Node> {
        &self.tree
    }

    #[inline]
    pub fn tree_mut(&mut self) -> &mut Tree<C::Node> {
        &mut self.tree
    }

    #[inline]
    pub fn interruption_token(&self) -> bool {
        self.interruption_token.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_interruption_token(&self, value: bool) {
        self.interruption_token.store(value, Ordering::Relaxed)
    }
}

impl<C: EngineConfig> Engine<C>
where
    C::Search: SearchStrategy<C>,
    C::Logger: LoggerTrait<C>,
{
    #[inline]
    pub fn print(&self, msg: &str) {
        C::Logger::print(msg, self.params().logger(), self)
    }

    pub fn search(&self, limits: &SearchLimits) -> SearchStats {
        C::Search::execute(limits, self.params().search(), self)
    }
}
