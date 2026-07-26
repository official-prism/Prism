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

use crate::{
    engine::tree::components::{
        HasMove,
        HasQ,
        HasVisits,
    },
    prelude::*,
};

#[derive(Debug)]
pub struct MaxQ;

const EVAL_SCALE: f32 = 400.0;
const SCORE_EPSILON: f32 = 0.0001;

impl Strategy for MaxQ {
    type Params = EmptyParams;
}

impl<C: EngineConfig> BestMoveStrategy<C> for MaxQ
where
    C::Edge: HasQ + HasVisits,
{
    fn execute(line_idx: usize, _params: &Self::Params, engine: &Engine<C>) -> (Move, i32) {
        let edges = engine.tree().root_node().edges();

        let mut ranking: Vec<usize> = (0..edges.len()).collect();
        ranking.sort_unstable_by(|&lhs, &rhs| {
            let (lhs, rhs) = (&edges[lhs], &edges[rhs]);
            rhs.q()
                .total_cmp(&lhs.q())
                .then(rhs.visits().cmp(&lhs.visits()))
        });

        let Some(&edge_idx) = ranking.get(line_idx) else {
            return (Move::NULL, 0);
        };

        let edge = &edges[edge_idx];

        (edge.mv(), score_to_cp(edge.q()))
    }
}

fn score_to_cp(score: f32) -> i32 {
    let score = score.clamp(SCORE_EPSILON, 1.0 - SCORE_EPSILON);
    (-EVAL_SCALE * (1.0 / score - 1.0).ln()) as i32
}
