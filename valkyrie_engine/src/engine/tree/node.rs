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

mod edge;
mod game_state;

pub use edge::Edge;
pub use game_state::{AtomicGameState, GameState};

use super::payload::PayloadType;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug)]
pub struct Node<NP: PayloadType = (), EP: PayloadType = ()> {
    state: AtomicGameState,
    edges: RwLock<Vec<Edge<EP>>>,
    payload: NP,
}

impl<NP: PayloadType, EP: PayloadType> Node<NP, EP> {
    #[inline]
    pub fn size() -> usize {
        std::mem::size_of::<Node<NP, EP>>()
    }

    pub fn new() -> Self {
        Self {
            state: AtomicGameState::new(GameState::Ongoing),
            edges: RwLock::new(Vec::new()),
            payload: NP::default(),
        }
    }

    #[inline]
    pub fn payload(&self) -> &NP {
        &self.payload
    }

    #[inline]
    pub fn payload_mut(&mut self) -> &mut NP {
        &mut self.payload
    }

    #[inline]
    pub fn game_state(&self) -> GameState {
        self.state.load()
    }

    #[inline]
    pub fn set_game_state(&self, state: GameState) {
        self.state.store(state);
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.state.load() != GameState::Ongoing
    }

    #[inline]
    pub fn edges(&self) -> RwLockReadGuard<'_, Vec<Edge<EP>>> {
        self.edges.read().unwrap()
    }

    #[inline]
    pub fn edges_mut(&self) -> RwLockWriteGuard<'_, Vec<Edge<EP>>> {
        self.edges.write().unwrap()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.read().unwrap().len()
    }

    pub fn total_visits(&self) -> u64 {
        let edges = self.edges.read().unwrap();
        edges.iter().map(|e| e.visits()).sum()
    }
}

impl<NP: PayloadType, EP: PayloadType> Default for Node<NP, EP> {
    fn default() -> Self {
        Self::new()
    }
}
