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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NodeIndex(u64);

impl NodeIndex {
    pub const NULL: Self = Self(0);

    #[inline]
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl From<u64> for NodeIndex {
    #[inline]
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<NodeIndex> for u64 {
    #[inline]
    fn from(value: NodeIndex) -> Self {
        value.0
    }
}

#[derive(Debug)]
pub struct AtomicNodeIndex(AtomicU64);

impl AtomicNodeIndex {
    #[inline]
    pub fn new(index: NodeIndex) -> Self {
        Self(AtomicU64::new(index.raw()))
    }

    #[inline]
    pub fn null() -> Self {
        Self(AtomicU64::new(0))
    }

    #[inline]
    pub fn load(&self) -> NodeIndex {
        NodeIndex::new(self.0.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn store(&self, index: NodeIndex) {
        self.0.store(index.raw(), Ordering::Relaxed);
    }
}

impl Default for AtomicNodeIndex {
    fn default() -> Self {
        Self::null()
    }
}
