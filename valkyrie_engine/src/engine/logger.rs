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

use valkyrie_chess::Move;

use crate::{
    EmptyParams, Engine, StrategyParams, engine::{builder::EngineConfig, structures::SearchStats}
};

pub trait LoggerTrait: std::fmt::Debug + Send + Sync {
    type Params: StrategyParams + Send + Sync;

    fn print<C: EngineConfig>(msg: &str, params: &Self::Params, engine: &Engine<C>);
    fn search_report<C: EngineConfig>(
        time_passed: u64,
        search_stats: &SearchStats,
        params: &Self::Params,
        engine: &Engine<C>,
        end_of_search: bool
    );
    fn best_move<C: EngineConfig>(mv: Move, params: &Self::Params, engine: &Engine<C>);
}

#[derive(Debug)]
pub struct NoLogger;
impl LoggerTrait for NoLogger {
    type Params = EmptyParams;

    fn print<C: EngineConfig>(_msg: &str, _params: &Self::Params, _engine: &Engine<C>) {}
    fn search_report<C: EngineConfig>(
        _time_passed: u64,
        _search_stats: &SearchStats,
        _params: &Self::Params, 
        _engine: &Engine<C>,
        _end_of_search: bool
    ) {
    }
    fn best_move<C: EngineConfig>(_mv: Move, _params: &Self::Params, _engine: &Engine<C>) {}
}
