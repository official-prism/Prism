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

use crate::{Engine, engine::structures::SearchStats, prelude::*};

#[derive(Debug)]
pub struct ClassicBackpropagate;

crate::define_strategy_params! {
    ClassicBackpropagateParams {
        Options {
            ["MaxQ_Depth"] depth: i32 => 10, 1, 100;
            ["MaxQ_Verbose"] verbose: bool => false;
        }
        Tunables {
            lMR_Base: f64 => 0.75, 0.0, 2.0, 0.1, 0.1;
            lMR_Division: f64 => 2.25, 1.0, 10.0, 0.5, 0.1;
            fPU_Value: f64 => 0.5, 0.0, 1.0, 0.1, 0.1;
            aspiration_Window: i64 => 50, 1, 500, 10, 1;
            history_Threshold: i64 => 100, 0, 1000, 20, 1;
        }
    }
}

impl BackpropagateStrategy for ClassicBackpropagate {
    type Params = ClassicBackpropagateParams;

    fn execute<C: EngineConfig>(_params: &Self::Params, _engine: &Engine<C>, _search_stats: &SearchStats)
    where
        C::EdgePayload: QScore,
    {

    }
}