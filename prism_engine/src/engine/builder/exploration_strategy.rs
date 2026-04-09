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

use crate::{
    Engine,
    engine::{
        StrategyParams,
        builder::EngineConfig,
        structures::{IterationStats, SearchStats},
    },
};

register_strategy!(puct);
register_strategy!(test_compound);

pub trait ExplorationStrategy: std::fmt::Debug + Send + Sync {
    type Params: StrategyParams + Send + Sync;

    fn excute<C: EngineConfig>(
        params: &Self::Params,
        engine: &Engine<C>,
        search_stats: &SearchStats,
        iteration_stats: &IterationStats,
    );
}
