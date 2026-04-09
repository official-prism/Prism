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

use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering};

use prism_chess::Move;

use crate::engine::tree::node_index::AtomicNodeIndex;
use crate::engine::tree::node_index::NodeIndex;
use crate::engine::tree::payload::PayloadType;

#[derive(Debug)]
pub struct Edge<EP: PayloadType = ()> {
    child_node: AtomicNodeIndex,
    score: AtomicU64,
    visits: AtomicU64,
    policy: AtomicU32,
    draw_chance: AtomicU32,
    mv: AtomicU16,
    payload: EP,
}

impl<EP: PayloadType> Edge<EP> {
    pub fn new(mv: Move, policy: f32) -> Self {
        Self {
            score: AtomicU64::new(0f64.to_bits()),
            visits: AtomicU64::new(0),
            mv: AtomicU16::new(u16::from(mv)),
            policy: AtomicU32::new(policy.to_bits()),
            draw_chance: AtomicU32::new(0f32.to_bits()),
            child_node: AtomicNodeIndex::null(),
            payload: EP::default(),
        }
    }

    #[inline]
    pub fn payload(&self) -> &EP {
        &self.payload
    }

    #[inline]
    pub fn payload_mut(&mut self) -> &mut EP {
        &mut self.payload
    }

    #[inline]
    pub fn score(&self) -> f64 {
        f64::from_bits(self.score.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_score(&self, value: f64) {
        self.score.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    pub fn add_score(&self, delta: f64) {
        loop {
            let current = self.score.load(Ordering::Relaxed);
            let new = (f64::from_bits(current) + delta).to_bits();
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
    pub fn visits(&self) -> u64 {
        self.visits.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn add_visit(&self) {
        self.visits.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_visits(&self, count: u64) {
        self.visits.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn mv(&self) -> Move {
        Move::from(self.mv.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn policy(&self) -> f32 {
        f32::from_bits(self.policy.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_policy(&self, value: f32) {
        self.policy.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    pub fn draw_chance(&self) -> f32 {
        f32::from_bits(self.draw_chance.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn set_draw_chance(&self, value: f32) {
        self.draw_chance.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    pub fn child_node(&self) -> NodeIndex {
        self.child_node.load()
    }

    #[inline]
    pub fn set_child_node(&self, index: NodeIndex) {
        self.child_node.store(index);
    }

    #[inline]
    pub fn has_child(&self) -> bool {
        self.child_node.load() != NodeIndex::NULL
    }
}
