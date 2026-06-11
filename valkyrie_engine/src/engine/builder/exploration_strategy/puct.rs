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

use crate::{Engine, engine::structures::SearchStats, prelude::*};

#[derive(Debug)]
pub struct Puct;

crate::define_strategy_params! {
    PuctParams {
        Options {
            ["Puct_Log"] log: bool => true;
        }
        Tunables {
            cpuct: f64 => 1.41, 0.1, 5.0, 0.1, 0.1;
            puct_fPU: f64 => 0.5, 0.0, 1.0, 0.1, 0.1;
        }
    }
}

impl ExplorationStrategy for Puct {
    type Params = PuctParams;

    fn execute<C: EngineConfig>(_params: &Self::Params, _engine: &Engine<C>, _search_stats: &SearchStats)
    where
        C::EdgePayload: QScore,
    {

    }
}
