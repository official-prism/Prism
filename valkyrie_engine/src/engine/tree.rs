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

pub mod components;
pub mod edge_storage;
pub mod edges;
pub mod nodes;

pub(crate) mod node_index;

pub use edge_storage::EdgesStore;
pub use node_index::{AtomicNodeIndex, NodeIndex};

#[derive(Debug)]
pub struct Tree<N> {
    nodes: Vec<N>,
    capacity: usize,
}

impl<N> Tree<N> {
    pub fn new(size_in_mb: usize) -> Self {
        let capacity = Self::capacity_for(size_in_mb);
        Self {
            nodes: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn capacity_for(size_in_mb: usize) -> usize {
        let bytes = size_in_mb * 1024 * 1024;
        bytes / std::mem::size_of::<N>().max(1)
    }

    pub fn resize(&mut self, size_in_mb: usize) {
        *self = Self::new(size_in_mb);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    #[inline]
    pub fn get(&self, index: NodeIndex) -> Option<&N> {
        self.nodes.get(index.raw() as usize)
    }

    #[inline]
    pub fn push(&mut self, node: N) -> NodeIndex {
        let index = NodeIndex::new(self.nodes.len() as u64);
        self.nodes.push(node);
        index
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.nodes.len() >= self.capacity
    }
}
