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

mod node;
pub(crate) mod node_index;

pub use node::Node;
pub use node_index::NodeIndex;

use std::{
    collections::HashMap,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};

#[derive(Debug)]
pub struct Tree {
    tree: RwLock<HashMap<NodeIndex, Node>>,
}

impl Tree {
    pub fn new(size_in_mb: usize) -> Self {
        let bytes = size_in_mb * 1024 * 1024;
        let size = bytes / Node::size();

        Self {
            tree: RwLock::new(HashMap::with_capacity(size)),
        }
    }

    pub fn resize(&mut self, size_in_mb: usize) {
        let bytes = size_in_mb * 1024 * 1024;
        let size = bytes / Node::size();

        self.tree = RwLock::new(HashMap::with_capacity(size));
    }

    pub fn tree_full(&self, size_in_mb: usize) -> bool {
        let bytes = size_in_mb * 1024 * 1024;
        let size = bytes / Node::size();

        self.tree.read().unwrap().len() >= size
    }

    #[inline]
    pub fn read(&self) -> RwLockReadGuard<'_, HashMap<NodeIndex, Node>> {
        self.tree.read().unwrap()
    }

    #[inline]
    pub fn write(&self) -> RwLockWriteGuard<'_, HashMap<NodeIndex, Node>> {
        self.tree.write().unwrap()
    }

    #[inline]
    pub fn contains_key(&self, key: NodeIndex) -> bool {
        self.tree.read().unwrap().contains_key(&key)
    }

    #[inline]
    pub fn remove(&self, key: NodeIndex) -> Option<Node> {
        self.tree.write().unwrap().remove(&key)
    }

    #[inline]
    pub fn clear(&self) {
        self.tree.write().unwrap().clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tree.read().unwrap().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tree.read().unwrap().is_empty()
    }
}
