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

use crate::{FEN, Piece, Side, Square, board::ChessBoard};
use colored::Colorize;

impl ChessBoard {
    pub fn draw_board(&self) {
        let piece_icons: [[&str; 6]; 2] = [
            [" P", " N", " B", " R", " Q", " K"],
            [" p", " n", " b", " r", " q", " k"],
        ];

        let mut info = Vec::new();
        let fen = format!("FEN: {}", FEN::from(self));
        info.push(fen.as_str());
        let zobrist = format!("Zobrist Key: {}", self.hash());
        info.push(zobrist.as_str());

        let castle_rights = format!("Castle Rights: {}", self.castle_rights());
        info.push(castle_rights.as_str());
        let side_sign = format!("Side To Move: {}", self.side());
        info.push(side_sign.as_str());
        let en_passant = format!("En Passant: {}", self.en_passant_square());
        info.push(en_passant.as_str());
        let half_moves = format!("Half Moves: {}", self.half_moves());
        info.push(half_moves.as_str());
        let in_check = format!("In Check: {}", self.is_in_check());
        info.push(in_check.as_str());
        let phase = format!("Phase: {}", self.phase());
        info.push(phase.as_str());

        let mut result = "   -----------------\n".to_string();
        for rank in 0..8 {
            result += format!(
                "{} |",
                if self.side() == Side::WHITE {
                    7 - rank
                } else {
                    rank
                } + 1
            )
            .as_str();
            for file in 0..8 {
                let square = Square::from_coords(
                    if self.side() == Side::WHITE {
                        7 - rank
                    } else {
                        rank
                    },
                    if self.side() == Side::WHITE {
                        file
                    } else {
                        7 - file
                    },
                );

                if square == self.en_passant_square() {
                    result += " x";
                    continue;
                }

                let piece_type = self.piece_on_square(square);
                let piece_side = self.color_on_square(square);
                if piece_type == Piece::NONE {
                    result += " .";
                } else if piece_side == Side::BLACK {
                    result += piece_icons[usize::from(Side::BLACK)][usize::from(piece_type)]
                        .truecolor(15, 60, 220)
                        .to_string()
                        .as_str();
                } else {
                    result += piece_icons[usize::from(Side::WHITE)][usize::from(piece_type)]
                        .truecolor(225, 225, 160)
                        .to_string()
                        .as_str();
                }
            }
            result += format!(" | {}", info[rank as usize]).as_str();
            result += "\n";
        }
        result += "   -----------------\n";

        let mut files: Vec<&str> = vec!["A", "B", "C", "D", "E", "F", "G", "H"];
        if self.side() == Side::BLACK {
            files.reverse();
        };
        result += format!("    {}\n", files.join(" ")).as_str();

        println!("{}", result);
    }
}
