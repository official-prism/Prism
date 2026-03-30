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

fn main() {
    let board = prism_chess::ChessBoard::from(&FEN::start_position());
    const RUNS: u32 = 5;

    let mut bulk_total = 0u128;
    for _ in 0..RUNS {
        let (_, duration) = prism_chess::perft::<true, false, false>(&board, Some(7));
        bulk_total += duration.as_millis();
    }

    let mut nobulk_total = 0u128;
    for _ in 0..RUNS {
        let (_, duration) = prism_chess::perft::<false, false, false>(&board, Some(6));
        nobulk_total += duration.as_millis();
    }

    println!("Bulk   perft(7) avg: {} ms", bulk_total / RUNS as u128);
    println!("NoBulk perft(6) avg: {} ms", nobulk_total / RUNS as u128);
}
