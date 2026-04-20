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

#[macro_use]
mod macros;

pub mod best_move_strategy;
pub mod exploration_strategy;
pub mod expansion_strategy;
pub mod backpropagate_strategy;
pub mod node_strategy;
pub mod search_strategy;
pub mod time_manager_strategy;

pub use self::best_move_strategy::BestMoveStrategy;
pub use self::exploration_strategy::ExplorationStrategy;
pub use self::expansion_strategy::ExpansionStrategy;
pub use self::backpropagate_strategy::BackpropagateStrategy;
pub use self::node_strategy::NodeStrategy;
pub use self::search_strategy::SearchStrategy;
pub use self::time_manager_strategy::TimeManagerStrategy;

use crate::engine::logger::{LoggerTrait, NoLogger};

pub struct Unspecified;

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

define_engine_config! {
    general: GeneralParams,
    with_params {
        BestMove:        BestMoveStrategy      | best_move       | Unspecified,
        Exploration:     ExplorationStrategy   | exploration     | Unspecified,
        Expansion:       ExpansionStrategy     | expansion       | Unspecified,
        Backpropagation: BackpropagateStrategy | backpropagation | Unspecified,
        Search:          SearchStrategy        | search          | Unspecified,
        TimeManager:     TimeManagerStrategy   | time_manager    | Unspecified,
    }
    without_params {
        Node:   NodeStrategy | node   | Unspecified,
        Logger: LoggerTrait  | logger | NoLogger,
    }
}
