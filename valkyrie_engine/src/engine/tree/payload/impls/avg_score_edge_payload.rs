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

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::engine::tree::payload::QScore;

#[derive(Debug, Default)]
pub struct AvgScoreEdgePayload {
    score: AtomicU64,
    draw_chance: AtomicU32,
}

impl QScore for AvgScoreEdgePayload {
    #[inline]
    fn total_score(&self) -> f64 {
        f64::from_bits(self.score.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_score(&self, value: f64) {
        self.score.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn add_score(&self, value: f64) {
        loop {
            let current = self.score.load(Ordering::Relaxed);
            let new = (f64::from_bits(current) + value).to_bits();
            if self
                .score
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    #[inline]
    fn draw_chance(&self) -> f32 {
        f32::from_bits(self.draw_chance.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_draw_chance(&self, value: f32) {
        self.draw_chance.store(value.to_bits(), Ordering::Relaxed);
    }
}
