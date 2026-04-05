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

use prism_chess::{ChessBoard, ChessPosition, FEN};

use crate::engine::builder::search_step_strategy::SearchStepStrategy;
use crate::engine::logger::{Logger, NoLogger};

use super::{Engine, EngineParams};
use std::marker::PhantomData;
use std::sync::atomic::AtomicBool;

macro_rules! register_strategy {
    ($($name:ident),*) => {
        $(
            pub mod $name;
            pub use self::$name::*;
        )*
    };
}

pub mod best_move_strategy;
pub mod exploration_strategy;
pub mod search_step_strategy;

pub use self::best_move_strategy::BestMoveStrategy;
pub use self::exploration_strategy::ExplorationStrategy;

pub struct Unspecified;

pub struct EngineBuilder<BMS = Unspecified, ES = Unspecified, SS = Unspecified, L = NoLogger> {
    _bms: PhantomData<BMS>,
    _es: PhantomData<ES>,
    _ss: PhantomData<SS>,
    _l: PhantomData<L>,
}

impl EngineBuilder<Unspecified, Unspecified, Unspecified, NoLogger> {
    pub fn new() -> Self {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
            _ss: PhantomData,
            _l: PhantomData,
        }
    }
}

impl Default for EngineBuilder<Unspecified, Unspecified> {
    fn default() -> Self {
        Self::new()
    }
}

impl<BMS, ES, SS, L> EngineBuilder<BMS, ES, SS, L> {
    pub fn exploration_strategy<S: ExplorationStrategy>(self) -> EngineBuilder<BMS, S, SS, L> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
            _ss: PhantomData,
            _l: PhantomData,
        }
    }

    pub fn best_move_strategy<S: BestMoveStrategy>(self) -> EngineBuilder<S, ES, SS, L> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
            _ss: PhantomData,
            _l: PhantomData,
        }
    }

    pub fn search_step_strategy<S: SearchStepStrategy>(self) -> EngineBuilder<BMS, ES, S, L> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
            _ss: PhantomData,
            _l: PhantomData,
        }
    }

    pub fn logger<NL: Logger>(self) -> EngineBuilder<BMS, ES, SS, NL> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
            _ss: PhantomData,
            _l: PhantomData,
        }
    }
}

pub trait EngineConfig {
    type BestMove: BestMoveStrategy;
    type Exploration: ExplorationStrategy;
    type SearchStep: SearchStepStrategy;
    type Logger: Logger;
}

pub struct GenericConfig<BMS, ES, SS, L> {
    _bms: PhantomData<BMS>,
    _es: PhantomData<ES>,
    _ss: PhantomData<SS>,
    _l: PhantomData<L>,
}

impl<BMS: BestMoveStrategy, ES: ExplorationStrategy, SS: SearchStepStrategy, L: Logger> EngineConfig
    for GenericConfig<BMS, ES, SS, L>
{
    type BestMove = BMS;
    type Exploration = ES;
    type SearchStep = SS;
    type Logger = L;
}

impl<BMS: BestMoveStrategy, ES: ExplorationStrategy, SS: SearchStepStrategy, L: Logger> EngineBuilder<BMS, ES, SS, L> {
    pub fn build(self) -> Engine<GenericConfig<BMS, ES, SS, L>> {
        Engine {
            params: EngineParams::new(),
            position: ChessPosition::from(ChessBoard::from(&FEN::start_position())),
            interruption_token: AtomicBool::new(false),
            _c: PhantomData,
        }
    }
}

