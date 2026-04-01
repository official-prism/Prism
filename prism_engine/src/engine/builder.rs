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

use super::{Engine, EngineParams, GenericConfig};
use std::marker::PhantomData;

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

pub use self::best_move_strategy::BestMoveStrategy;
pub use self::exploration_strategy::ExplorationStrategy;

pub struct Unspecified;

pub struct EngineBuilder<BMS = Unspecified, ES = Unspecified> {
    _bms: PhantomData<BMS>,
    _es: PhantomData<ES>,
}

impl EngineBuilder<Unspecified, Unspecified> {
    pub fn new() -> Self {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
        }
    }
}

impl Default for EngineBuilder<Unspecified, Unspecified> {
    fn default() -> Self {
        Self::new()
    }
}

impl<BMS, ES> EngineBuilder<BMS, ES> {
    pub fn exploration_strategy<S: ExplorationStrategy>(self) -> EngineBuilder<BMS, S> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
        }
    }

    pub fn best_move_strategy<S: BestMoveStrategy>(self) -> EngineBuilder<S, ES> {
        EngineBuilder {
            _bms: PhantomData,
            _es: PhantomData,
        }
    }
}

impl<BMS: BestMoveStrategy, ES: ExplorationStrategy> EngineBuilder<BMS, ES> {
    pub fn build(self) -> Engine<GenericConfig<BMS, ES>> {
        Engine {
            params: EngineParams::new(),
            position: ChessPosition::from(ChessBoard::from(&FEN::start_position())),
            _c: PhantomData,
        }
    }
}
