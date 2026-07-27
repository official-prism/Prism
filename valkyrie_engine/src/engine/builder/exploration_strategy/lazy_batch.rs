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

use std::marker::PhantomData;

use crate::{
    engine::{builder::exploration_strategy::VisitDistribution, tree::components::HasVisits}, prelude::*,
};

#[deprecated(note = "LazyBatch is development module to test single selection formulas a batch selectors. It assumes total virtual loss and was not made with multithreading in mind.")]
#[derive(Debug)]
pub struct LazyBatch<S>(PhantomData<S>);

#[allow(deprecated)]
impl<S: Strategy> Strategy for LazyBatch<S> {
    type Params = S::Params;
}

#[allow(deprecated)]
impl<C, S> ExplorationStrategy<C> for LazyBatch<S>
where
    C: EngineConfig,
    S: ExplorationStrategy<C>,
    C::Node: HasVisits,
    C::Edge: HasVisits
{
    fn execute(
        distribution: &mut VisitDistribution,
        node: &<C as EngineConfig>::Node,
        budget: u64,
        params: &Self::Params,
        engine: &Engine<C>,
    ) {
        assert!(node.edge_count() > 0);

        distribution.clear();

        let node_clone = node.clone();

        let mut counts = vec![0u64; node_clone.edge_count()];
        let mut step_distribution = VisitDistribution::new();

        for _ in 0..budget {
            S::execute(&mut step_distribution, &node_clone, 1, params, engine);

            let step = step_distribution.as_slice()[0];
            let edge_idx = step.edge_index();
            counts[edge_idx] += step.vists();
            node_clone.add_visit();
            node_clone.edges()[edge_idx].add_visit();
        }

        for (idx, &visits) in counts.iter().enumerate() {
            if visits > 0 {
                distribution.push(idx, visits);
            }
        }
    }
}