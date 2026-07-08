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

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    Ongoing,
    Won(u8),
    Lost(u8),
    Drew,
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Ongoing
    }
}

#[derive(Debug)]
pub struct AtomicGameState(AtomicU8);

impl AtomicGameState {
    #[inline]
    pub fn new(state: GameState) -> Self {
        Self(AtomicU8::new(match state {
            GameState::Ongoing => 0,
            GameState::Won(dtm) => 1 << 6 | (dtm & 0x3F),
            GameState::Lost(dtm) => 2 << 6 | (dtm & 0x3F),
            GameState::Drew => 3 << 6,
        }))
    }

    #[inline]
    pub fn load(&self) -> GameState {
        let value = self.0.load(Ordering::Relaxed);
        let tag = value >> 6;
        let payload = value & 0x3F;
        match tag {
            0 => GameState::Ongoing,
            1 => GameState::Won(payload),
            2 => GameState::Lost(payload),
            3 => GameState::Drew,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn store(&self, state: GameState) {
        self.0.store(
            match state {
                GameState::Ongoing => 0,
                GameState::Won(dtm) => 1 << 6 | (dtm & 0x3F),
                GameState::Lost(dtm) => 2 << 6 | (dtm & 0x3F),
                GameState::Drew => 3 << 6,
            },
            Ordering::Relaxed,
        );
    }
}

impl Default for AtomicGameState {
    fn default() -> Self {
        Self::new(GameState::Ongoing)
    }
}
