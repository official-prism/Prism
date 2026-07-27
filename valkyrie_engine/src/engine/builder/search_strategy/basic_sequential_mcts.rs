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

use std::time::Instant;

use valkyrie_chess::ChessPosition;

use crate::{
    engine::{
        builder::SimulationStrategy,
        policy_entry::PolicyEntry,
        tree::components::{
            GameState,
            HasChild,
            HasGameState,
            HasMove,
            HasVisits,
        },
    },
    prelude::*,
};

#[derive(Debug)]
pub struct BasicSequentialMCTS;

const SOFT_LIMIT_CHECK_THRESHOLD: u64 = 4096;
const HARD_LIMIT_CHECK_THRESHOLD: u64 = 128;
const PRINT_REPORT_INTERVAL: u128 = 1000;
const REPORT_INTERVAL_CHECK_THRESHOLD: u64 = 128;

crate::define_strategy_params! {
    ClassicalSearchParams { }
}

impl Strategy for BasicSequentialMCTS {
    type Params = ClassicalSearchParams;
}

impl<C: EngineConfig> SearchStrategy<C> for BasicSequentialMCTS
where
    C::TimeManager: TimeManagerStrategy<C>,
    C::Logger: LoggerTrait<C>,
    C::Exploration: ExplorationStrategy<C>,
    C::Expansion: ExpansionStrategy<C>,
    C::Simulation: SimulationStrategy<C>,
    C::Backpropagation: BackpropagationStrategy<C>,
    C::BestMove: BestMoveStrategy<C>,
    C::Node: HasVisits,
{
    fn execute(limits: &SearchLimits, params: &Self::Params, engine: &Engine<C>) -> SearchStats {
        let search_stats = SearchStats::new();
        let search_time = Instant::now();

        let root_node = &engine.tree()[engine.tree().root_index()];

        if root_node.edge_count() == 0 {
            root_node.set_game_state(node_state(engine.position(), engine));

            if !root_node.is_terminal() {
                Self::expand_node(root_node, engine.position(), engine);
            }
        }

        if root_node.edge_count() > 0 {
            Self::main_thread_search(limits, &search_stats, search_time, params, engine);
        }

        C::Logger::search_report(
            search_time.elapsed().as_millis() as u64,
            &search_stats,
            engine.params().logger(),
            engine,
            true,
        );

        let best_move = engine.tree().get_best_move(engine);
        C::Logger::best_move(best_move, engine.params().logger(), engine);

        search_stats
    }
}

impl BasicSequentialMCTS {
    fn main_thread_search<C: EngineConfig>(
        limits: &SearchLimits,
        stats: &SearchStats,
        search_time: Instant,
        params: &ClassicalSearchParams,
        engine: &Engine<C>,
    ) where
        C::TimeManager: TimeManagerStrategy<C>,
        C::Logger: LoggerTrait<C>,
        C::Exploration: ExplorationStrategy<C>,
        C::Expansion: ExpansionStrategy<C>,
        C::Simulation: SimulationStrategy<C>,
        C::Backpropagation: BackpropagationStrategy<C>,
        C::Node: HasVisits,
    {
        let mut last_report_time = Instant::now();

        let time_manager = C::TimeManager::new(limits, engine.params().time_manager(), engine);
        let mut main_thread_iters = 0u64;

        while !engine.interruption_token() {
            let position = engine.position().clone();
            let mut depth = 0;

            if Self::search_step::<_, true>(engine.tree().root_index(), position, params, stats, engine, &mut depth).is_none() {
                engine.set_interruption_token(true);
                break;
            }

            let prev_avg_depth = stats.avg_depth();
            let prev_max_depth = stats.max_depth();

            stats.add_iteration(depth);
            main_thread_iters += 1;

            if should_print_report(stats, main_thread_iters, prev_max_depth, prev_avg_depth, last_report_time) {
                let time_passed = search_time.elapsed().as_millis() as u64;
                C::Logger::search_report(time_passed, stats, engine.params().logger(), engine, false);
                last_report_time = Instant::now();
            }

            if should_stop(main_thread_iters, limits, stats, engine, search_time, &time_manager) {
                engine.set_interruption_token(true);
                break;
            }        
        }
    }

    fn search_step<C: EngineConfig, const ROOT: bool>(
        current_node_idx: NodeIndex,
        position: ChessPosition,
        params: &ClassicalSearchParams,
        stats: &SearchStats,
        engine: &Engine<C>,
        depth: &mut u64,
    ) -> Option<<C::Simulation as SimulationStrategy<C>>::Output>
    where
        C::Exploration: ExplorationStrategy<C>,
        C::Expansion: ExpansionStrategy<C>,
        C::Simulation: SimulationStrategy<C>,
        C::Backpropagation: BackpropagationStrategy<C>,
        C::Node: HasVisits,
    {
        let current_node = &engine.tree()[current_node_idx];

        let payload = if !ROOT &&
            (current_node.visits() == 0 || current_node.edge_count() == 0 || current_node.is_terminal()) {
            let payload = Self::simulate_node(&[], current_node, &position, engine);

            if current_node.visits() == 0 && !current_node.is_terminal() {
                Self::expand_node(current_node, &position, engine);
            }

            payload
        } else {
            let distrib = C::Exploration::execute(current_node, 1, engine.params().exploration(), engine);
            let edge_idx = distrib.as_slice()[0].edge_index();

            let (mv, child_idx) = {
                let edges_lock = current_node.edges();
                let edge = edges_lock.get(edge_idx).unwrap();

                let child_idx = if edge.has_child() {
                    edge.child()
                } else {
                    let new_node_idx = engine.tree().create_node()?;
                    edge.set_child(new_node_idx);
                    new_node_idx
                };

                (edge.mv(), child_idx)
            };

            let mut position_clone = position.clone();
            position_clone.make_move_no_mask(mv);

            *depth = depth.saturating_add(1);

            let payload_opt = Self::search_step::<_, false>(child_idx, position_clone, params, stats, engine, depth);

            let payload = payload_opt?;

            C::Backpropagation::execute(&payload, current_node, edge_idx, engine.params().backpropagation(), engine);

            payload
        };

        current_node.add_visit();

        Some(payload.flipped())
    }

    fn expand_node<C: EngineConfig>(node: &C::Node, position: &ChessPosition, engine: &Engine<C>)
    where
        C::Expansion: ExpansionStrategy<C>,
    {
        let mut policy_distribution = Vec::with_capacity(35);
        position.board().map_legal_moves(|mv| {
            policy_distribution.push(PolicyEntry::new(mv, 1.0));
        });

        C::Expansion::execute(&mut policy_distribution, node, engine.params().expansion(), engine);
    }

    fn simulate_node<C: EngineConfig>(raw_evals: &[f32], node: &C::Node, position: &ChessPosition, engine: &Engine<C>) -> <<C as EngineConfig>::Simulation as SimulationStrategy<C>>::Output
    where
        C::Simulation: SimulationStrategy<C>,
        C::Node: HasVisits,
    {
        if node.visits() == 0 {
            node.set_game_state(node_state(position, engine));
        }

        match node.game_state() {
            GameState::Won(_) => Payload::WIN,
            GameState::Lost(_) => Payload::LOSS,
            GameState::Drew => Payload::DRAW,
            GameState::Ongoing => C::Simulation::execute(raw_evals, position, engine.params().simulation(), engine),
        }
    }

}

fn node_state<C: EngineConfig>(position: &ChessPosition, engine: &Engine<C>) -> GameState {
    let mut possible_moves = 0;
    position.board().map_legal_moves(|_| possible_moves += 1);

    if possible_moves == 0 {
        if position.board().is_in_check() {
            GameState::Lost(0)
        } else {
            GameState::Drew
        }
    } else if is_draw(position, engine) {
        GameState::Drew
    } else {
        GameState::Ongoing
    }
}

fn is_draw<C: EngineConfig>(position: &ChessPosition, engine: &Engine<C>) -> bool {
    if position.board().half_moves() >= 100 || position.board().is_insufficient_material() {
        return true;
    }

    let key = position.board().hash();
    let history_repetitions = engine.position().history().get_repetitions(key);
    let search_repetitions = position.history().get_repetitions(key) - history_repetitions;

    history_repetitions >= 3 || search_repetitions >= 2 || history_repetitions + search_repetitions >= 3
}

fn should_stop<C: EngineConfig>(
    iterations: u64, 
    limits: &SearchLimits, 
    stats: &SearchStats, 
    engine: &Engine<C>, 
    search_time: Instant, 
    time_manager: &C::TimeManager
) -> bool
    where
        C::TimeManager: TimeManagerStrategy<C>
{
    if limits.check_limits(stats, engine) {
        return true;
    }

    if engine.tree().is_full() {
        return true;
    }

    if iterations.is_multiple_of(HARD_LIMIT_CHECK_THRESHOLD)
        && time_manager.hard_limit(
            search_time.elapsed().as_millis() as u64,
            engine.params().time_manager(),
            engine,
        )
    {
        return true;
    }

    if iterations.is_multiple_of(SOFT_LIMIT_CHECK_THRESHOLD)
        && time_manager.soft_limit(
            search_time.elapsed().as_millis() as u64,
            engine.params().time_manager(),
            engine,
        )
    {
        return true;
    }

    false
}

fn should_print_report(
    stats: &SearchStats, 
    iterations: u64, 
    prev_max: u64, 
    prev_avg: u64, 
    last_report_time: Instant
) -> bool {
    stats.avg_depth() > prev_avg
        || stats.max_depth() > prev_max
        || (iterations.is_multiple_of(REPORT_INTERVAL_CHECK_THRESHOLD)
        && last_report_time.elapsed().as_millis() >= PRINT_REPORT_INTERVAL)
}
