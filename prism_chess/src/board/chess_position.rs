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

use crate::{
    Move,
    board::{chess_board::ChessBoard, move_history::MoveHistory},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChessPosition {
    board: ChessBoard,
    history: MoveHistory,
}

impl ChessPosition {
    #[inline]
    pub fn board(&self) -> &ChessBoard {
        &self.board
    }

    #[inline]
    pub fn history(&self) -> &MoveHistory {
        &self.history
    }

    #[inline]
    pub fn reset_history(&mut self) {
        self.history.reset()
    }

    #[inline]
    pub fn make_move_no_mask(&mut self, mv: Move) {
        let mask = self.board.castle_rights().get_castle_mask();
        self.make_move(mv, &mask);
    }

    #[inline]
    pub fn make_move(&mut self, mv: Move, mask: &[u8; 64]) {
        self.board.make_move(mv, mask);

        if self.board.half_moves() == 0 {
            self.history.reset()
        }

        self.history.push(self.board.hash())
    }
}

impl From<ChessBoard> for ChessPosition {
    fn from(value: ChessBoard) -> Self {
        let mut position = Self {
            board: value,
            history: MoveHistory::default(),
        };

        position.history.push(value.hash());

        position
    }
}
