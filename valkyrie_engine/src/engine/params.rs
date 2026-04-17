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

pub mod macros;

use crate::define_engine_params;
use crate::engine::EngineConfig;
use crate::engine::builder::{
    BestMoveStrategy, ExplorationStrategy, SearchStepStrategy, TimeManagerStrategy,
};

crate::define_strategy_params! {
    GeneralParams {
        Options {
            ["Hash"] hash: i32 => 1024, 1, 524288;
            ["UCI_Chess960"] ches960: bool => false;
            ["ItersAsNodes"] iters_as_nodes: bool => false;
        }
        Buttons {
            "Clear",
        }
    }
}

pub trait StrategyParams: std::fmt::Debug + Clone + Send + Sync {
    fn new() -> Self;
    fn set_option(&mut self, name: &str, value: &str) -> std::result::Result<(), String>;
    fn print_options(&self);
    fn print_tunables(&self);
}

define_engine_params!(
    pub struct EngineParams<C: EngineConfig> {
        general: GeneralParams,
        best_move: <C::BestMove as BestMoveStrategy>::Params,
        exploration: <C::Exploration as ExplorationStrategy>::Params,
        search: <C::SearchStep as SearchStepStrategy>::Params,
        time_manager: <C::TimeManager as TimeManagerStrategy>::Params,
    }
);
