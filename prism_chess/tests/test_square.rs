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

use prism_chess::{Bitboard, Square};

#[test]
fn shift_left() {
    assert_eq!(Square::from(32) << 3, Square::from(35));
    assert_eq!(
        Square::from(32) << 3,
        Bitboard::from(Square::from(32)).shift_left(3).ls1b_square()
    );
}

#[test]
fn shift_right() {
    assert_eq!(Square::from(32) >> 3, Square::from(29));
    assert_eq!(
        Square::from(32) >> 3,
        Bitboard::from(Square::from(32))
            .shift_right(3)
            .ls1b_square()
    );
}

#[test]
fn from_string() {
    assert_eq!(Square::from("h8"), Square::H8);
    assert_eq!(Square::from("b2".to_string()), Square::B2);
}

#[test]
fn to_string() {
    assert_eq!(String::from(Square::H8), "h8");
    assert_eq!(String::from(Square::B2), "b2");
    assert_eq!(String::from(Square::NULL), "NULL");
}
