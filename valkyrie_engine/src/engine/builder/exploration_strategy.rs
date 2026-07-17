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

use crate::prelude::*;

crate::register_strategy!(puct);
crate::register_strategy!(lazy_batch);

pub trait ExplorationStrategy<C: EngineConfig>: Strategy {
    fn execute(node: &C::Node, budget: u64, params: &Self::Params, engine: &Engine<C>) -> VisitDistribution;
}

#[derive(Clone)]
pub struct VisitDistribution {
    values: [VisitDistributionEntry; 256],
    len: usize,
}

#[derive(Clone, Copy, Default)]
pub struct VisitDistributionEntry {
    edge_idx: usize,
    visits: u64,
}

impl VisitDistributionEntry {
    pub fn edge_index(&self) -> usize {
        self.edge_idx
    }

    pub fn vists(&self) -> u64 {
        self.visits
    }
}

impl VisitDistribution {
    fn new() -> Self {
        Self { 
            values: [VisitDistributionEntry::default(); 256], 
            len: 0 
        }
    }

    pub fn push(&mut self, edge_idx: usize, visits: u64) {
        self.values[self.len] = VisitDistributionEntry { edge_idx, visits };
        self.len = self.len + 1;
    }

    pub fn as_slice(&self) -> &[VisitDistributionEntry] {
        &self.values[..self.len]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, VisitDistributionEntry> {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a VisitDistribution {
    type Item = &'a VisitDistributionEntry;
    type IntoIter = std::slice::Iter<'a, VisitDistributionEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}