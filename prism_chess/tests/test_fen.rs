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

use prism_chess::FEN;

#[test]
fn haha() {
    let fen = FEN::from("rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 0 9");
    assert_eq!(fen.castle_rights, "HAha");

    let fen = FEN::from("rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w KQkq - 0 9");
    assert_eq!(fen.castle_rights, "HAha");

    let fen = FEN::from("rrrkrrrr/pp2p1pp/8/2pp1p2/2P4P/P7/BP1PPPP1/R1BRNKRR w KQkq - 0 9");
    assert_eq!(fen.castle_rights, "HAha");

    let fen = FEN::from("rrrkrrr1/pp2p1pp/8/2pp1p2/2P4P/P7/BP1PPPP1/R1BRNKRR w KQq - 0 9");
    assert_eq!(fen.castle_rights, "HAa");
}

#[test]
fn hegb() {
    let fen = FEN::from("brnr1krr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BRRKR w KQgq - 2 9");
    assert_eq!(fen.castle_rights, "HEgb");
}

#[test]
fn hfhb() {
    let fen = FEN::from("brnr1krr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BRRKR w KFkq - 2 9");
    assert_eq!(fen.castle_rights, "HFhb");
}

#[test]
fn hchb() {
    let fen = FEN::from("brkr2rr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQRBKR1R w KQkq - 2 9");
    assert_eq!(fen.castle_rights, "HChb");
}

#[test]
fn fbda() {
    let fen = FEN::from("rrkrrrrr/pp3ppp/3pp3/2p5/5P2/P2P4/1PP1P1PP/RRRRKRRR w FBdq - 2 9");
    assert_eq!(fen.castle_rights, "FBda");
}
