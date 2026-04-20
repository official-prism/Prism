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
mod structures;
mod tree;

pub use builder::EngineParams;
pub use structures::StrategyParams;
pub use logger::LoggerTrait;
pub use structures::SearchLimits;
pub use structures::SearchStats;

use valkyrie_chess::ChessPosition;
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::prelude::*;
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
    pub fn set_interruption_token(&self, value: bool) {
        self.interruption_token.store(value, Ordering::Relaxed)
    }

    #[inline]
    pub fn print(&self, msg: &str) {
        C::Logger::print(msg, self)
    }

    pub fn search(&self, limits: &SearchLimits) -> SearchStats {
        C::Search::execute(limits, self.params().search(), self)
    }
}
