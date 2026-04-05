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

use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::structures::IterationStats;

pub struct SearchStats {
    max_depth: AtomicU64,
    cumulative_depth: AtomicU64,
    iterations: AtomicU64,
}

impl SearchStats {
    pub fn new() -> Self {
        Self { max_depth: AtomicU64::new(0), cumulative_depth: AtomicU64::new(0), iterations: AtomicU64::new(0) }
    }

    pub fn max_depth(&self) -> u64 {
        self.max_depth.load(Ordering::Relaxed)
    }

    pub fn cumulative_depth(&self) -> u64 {
        self.cumulative_depth.load(Ordering::Relaxed)
    }

    pub fn iterations(&self) -> u64 {
        self.iterations.load(Ordering::Relaxed)
    }

    pub fn avg_depth(&self) -> u64 {
        self.cumulative_depth() / self.iterations()
    }

    pub fn add_iteration(&self, stats: IterationStats) {
        let depth = stats.depth();
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        self.cumulative_depth.fetch_add(depth, Ordering::Relaxed);
        self.iterations.fetch_add(1, Ordering::Relaxed);
    }
}