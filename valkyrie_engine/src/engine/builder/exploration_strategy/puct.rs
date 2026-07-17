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
    engine::{
        builder::exploration_strategy::VisitDistribution, 
        tree::components::{HasPolicy, HasQ, HasVisits}
    }, 
    prelude::*
};

#[derive(Debug)]
pub struct Puct;

crate::define_strategy_params! {
    PuctParams {
        Tunables {
            cpuct: f64 => 1.41, 0.1, 5.0, 0.1, 0.1;
        }
    }
}

impl Strategy for Puct {
    type Params = PuctParams;
}

impl<C: EngineConfig> ExplorationStrategy<C> for Puct
where
    C::Node: HasVisits,
    C::Edge: HasVisits + HasPolicy + HasQ,
{
    fn execute(node: &<C as EngineConfig>::Node, _budget: u64, params: &Self::Params, _engine: &Engine<C>) -> VisitDistribution {
        assert!(node.edge_count() > 0);
        
        let mut distribution = VisitDistribution::new();

        let cpuct = cpuct::<C>(node, params);
        let parent_visits = node.visits();

        let mut edge_idx = usize::MAX;
        let mut best_puct = f64::NEG_INFINITY;

        for (idx, edge) in node.edges().iter().enumerate() {
            let child_visits = edge.visits();
            let score = if child_visits > 0 {
                edge.q()
            } else {
                0.5
            };

            let expl_score = exploration_score(parent_visits, child_visits);

            let puct = score + cpuct * (edge.policy() as f64) * expl_score;

            if puct > best_puct {
                best_puct = puct;
                edge_idx = idx;
            } 
        }

        assert_ne!(edge_idx, usize::MAX);

        distribution.push(edge_idx, 1);

        distribution
    }
}

fn cpuct<C: EngineConfig>(_parent_node: &<C as EngineConfig>::Node, params: &PuctParams) -> f64
where
    C::Node: HasVisits,
    C::Edge: HasVisits + HasPolicy + HasQ, 
{
    params.cpuct()
}

fn exploration_score(parent_visits: u64, child_visits: u64) -> f64 {
    (parent_visits as f64).sqrt().max(1.0) / (child_visits as f64 + 1.0)
}