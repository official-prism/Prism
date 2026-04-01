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

pub mod macros;

use crate::engine::EngineConfig;
use crate::engine::builder::{BestMoveStrategy, ExplorationStrategy};

crate::define_strategy_params! {
    GeneralParams {
        Options {
            ["Hash"] hash: i32 => 1024, 1, 524288;
            ["UCI_Chess960"] ches960: bool => false;
        }
    }
}

pub trait StrategyParams: std::fmt::Debug + Clone {
    fn new() -> Self;
    fn set_option(&mut self, name: &str, value: &str) -> std::result::Result<(), String>;
    fn print_options(&self);
    fn print_tunables(&self);
}

#[derive(Debug)]
pub struct EngineParams<C: EngineConfig> {
    pub general: GeneralParams,
    pub best_move: <C::BestMove as BestMoveStrategy>::Params,
    pub exploration: <C::Exploration as ExplorationStrategy>::Params,
}

impl<C: EngineConfig> EngineParams<C> {
    pub fn new() -> Self {
        Self {
            general: GeneralParams::new(),
            best_move: <C::BestMove as BestMoveStrategy>::Params::new(),
            exploration: <C::Exploration as ExplorationStrategy>::Params::new(),
        }
    }

    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), String> {
        self.general
            .set_option(name, value)
            .or_else(|_| self.best_move.set_option(name, value))
            .or_else(|_| self.exploration.set_option(name, value))
    }

    pub fn print_options(&self) {
        self.general.print_options();
        self.best_move.print_options();
        self.exploration.print_options();
    }

    pub fn print_tunables(&self) {
        self.general.print_tunables();
        self.best_move.print_tunables();
        self.exploration.print_tunables();
    }
}

impl<C: EngineConfig> Default for EngineParams<C> {
    fn default() -> Self {
        Self::new()
    }
}
