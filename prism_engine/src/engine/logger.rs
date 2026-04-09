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

use prism_chess::Move;

use crate::{
    Engine,
    engine::{builder::EngineConfig, structures::SearchStats},
};

pub trait LoggerTrait: Send + Sync {
    fn print<C: EngineConfig>(msg: &str, engine: &Engine<C>);
    fn search_report<C: EngineConfig>(
        time_passed: u64,
        search_stats: &SearchStats,
        engine: &Engine<C>,
    );
    fn best_move<C: EngineConfig>(mv: Move, engine: &Engine<C>);
}

pub struct NoLogger;
impl LoggerTrait for NoLogger {
    fn print<C: EngineConfig>(_msg: &str, _engine: &Engine<C>) {}
    fn search_report<C: EngineConfig>(
        _time_passed: u64,
        _search_stats: &SearchStats,
        _engine: &Engine<C>,
    ) {
    }
    fn best_move<C: EngineConfig>(_mv: Move, _engine: &Engine<C>) {}
}
