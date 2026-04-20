/*
    This file is part of Valkyrie.

    Copyright (C) 2026 Tomasz Jaworski

    Valkyrie is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Valkyrie is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Valkyrie.  If not, see <https://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with the ONNX Runtime and/or NVIDIA TensorRT libraries
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective ONNX Runtime and NVIDIA license
    agreements, the licensors of this Program grant you additional
    permission to convey the resulting work.
*/

mod command_processors;
mod input_wrapper;
mod logger;

use std::sync::atomic::{AtomicBool, Ordering};

use command_processors::misc_processor::MiscProcessor;
use command_processors::uci_processor::UciProcessor;
use input_wrapper::InputWrapper;
use valkyrie_engine::EngineBuilder;
use valkyrie_engine::engine::builder::node_strategy::AvgScoreNode;
use valkyrie_engine::engine::builder::search_strategy::Classical;
use valkyrie_engine::engine::builder::time_manager_strategy::SimpleTimeManager;
use valkyrie_engine::engine::builder::{best_move_strategy::MaxQ, exploration_strategy::Puct};

fn main() {
    let mut engine = EngineBuilder::new()
        .best_move::<MaxQ>()
        .exploration::<Puct>()
        .search::<Classical>()
        .node::<AvgScoreNode>()
        .time_manager::<SimpleTimeManager>()
        .logger::<crate::logger::Logger>()
        .build();

    let shutdown_token = AtomicBool::new(false);
    let mut input_wrapper = InputWrapper::new();

    while !shutdown_token.load(Ordering::SeqCst) {
        let cmd = match input_wrapper.get_input() {
            Some(cmd) => cmd,
            None => break,
        };

        let cmd = cmd.trim();

        if MiscProcessor::execute(cmd, &engine)
            || UciProcessor::execute(cmd, &shutdown_token, &mut input_wrapper, &mut engine)
        {
            continue;
        }

        println!("info string Unknown command");
    }
}
