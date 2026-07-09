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

use valkyrie_chess::Move;

use crate::engine::tree::NodeIndex;
use crate::engine::tree::components::{
    ChildLink, HasChild, HasMove, HasPolicy, HasQScore, HasVisits, MoveField, PolicyPrior,
    ScoreSum, VisitCount,
};

#[derive(Debug, Default)]
pub struct AvgScoreEdge {
    mv: MoveField,
    child: ChildLink,
    visits: VisitCount,
    policy: PolicyPrior,
    q: ScoreSum,
}

crate::forward! {
    impl HasMove for AvgScoreEdge => self.mv {
        fn mv(&self) -> Move;
        fn set_mv(&self, mv: Move);
    }
}

crate::forward! {
    impl HasChild for AvgScoreEdge => self.child {
        fn child(&self) -> NodeIndex;
        fn set_child(&self, index: NodeIndex);
    }
}

crate::forward! {
    impl HasVisits for AvgScoreEdge => self.visits {
        fn visits(&self) -> u64;
        fn add_visits(&self, count: u64);
    }
}

crate::forward! {
    impl HasPolicy for AvgScoreEdge => self.policy {
        fn policy(&self) -> f32;
        fn set_policy(&self, value: f32);
    }
}

crate::forward! {
    impl HasQScore for AvgScoreEdge => self.q {
        fn total_score(&self) -> f64;
        fn set_score(&self, value: f64);
        fn add_score(&self, value: f64);
        fn draw_chance(&self) -> f32;
        fn set_draw_chance(&self, value: f32);
    }
}
