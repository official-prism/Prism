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

use crate::{SearchLimits, prelude::*};

#[derive(Debug, Default)]
pub struct SimpleTimeManager {
    hard_limit: Option<u64>
}

crate::define_strategy_params! {
    TimeManagerParams {
        Options {
            ["MoveOverhead"] move_overhead: i32 => 10, 0, 1000;
        }
    }
}

impl TimeManagerStrategy for SimpleTimeManager {
    type Params = TimeManagerParams;

    fn new<C: crate::EngineConfig>(limits: &SearchLimits, params: &Self::Params, _engine: &crate::Engine<C>) -> Self {
        let mut time_manager = Self {
            hard_limit: None
        };

        if let Some(move_time) = limits.move_time() {
            time_manager.hard_limit = Some(move_time);
            return time_manager;
        }

        let time = limits.time().unwrap_or_default();

        if time == 0 {
            return time_manager;
        }

        let increment = limits.increment().unwrap_or_default();
        let mtg = limits.moves_to_go().unwrap_or(20);

        let mut hard_limit = (time / mtg) + (increment / 2);
        hard_limit = hard_limit.saturating_sub(params.move_overhead() as u64);
        time_manager.hard_limit = Some(hard_limit);

        time_manager
    }

    fn soft_limit<C: crate::EngineConfig>(&self, _time_passed: u64, _params: &Self::Params, _engine: &crate::Engine<C>) -> bool {
        false
    }

    fn hard_limit<C: crate::EngineConfig>(&self, time_passed: u64, _params: &Self::Params, _engine: &crate::Engine<C>) -> bool {
        if self.hard_limit.is_none() {
            return false;
        }

        time_passed >= self.hard_limit.unwrap()
    }
}