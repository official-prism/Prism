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

use crate::engine::logger::{LoggerTrait, NoLogger};
use crate::engine::tree::Tree;

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
pub mod time_manager_strategy;

pub use self::best_move_strategy::BestMoveStrategy;
pub use self::exploration_strategy::ExplorationStrategy;
pub use self::search_step_strategy::SearchStepStrategy;
pub use self::time_manager_strategy::TimeManagerStrategy;

pub struct Unspecified;

macro_rules! define_engine_config {
    ( $( $assoc:ident : $bound:path | $method:ident | $default:ty ),+ $(,)? ) => {
        pub trait EngineConfig: Send + Sync {
            $( type $assoc: $bound; )+
        }

        pub struct EngineBuilder< $( $assoc = $default ),+ >(
            $( PhantomData<$assoc>, )+
        );

        impl EngineBuilder< $( $default ),+ > {
            pub fn new() -> Self {
                EngineBuilder( $( <PhantomData<$default>>::default(), )+ )
            }
        }

        define_engine_config!(@setters
            []
            [ $( $assoc : $bound | $method | $default ),+ ]
        );

        pub struct GenericConfig< $( $assoc ),+ >(
            $( PhantomData<$assoc>, )+
        );

        impl< $( $assoc: $bound ),+ > EngineConfig for GenericConfig< $( $assoc ),+ > {
            $( type $assoc = $assoc; )+
        }

        impl< $( $assoc: $bound ),+ > EngineBuilder< $( $assoc ),+ > {
            pub fn build(self) -> Engine<GenericConfig< $( $assoc ),+ >> {
                let params = EngineParams::new();
                let hash_size = params.general().hash() as usize;
                Engine {
                    params,
                    position: ChessPosition::from(ChessBoard::from(&FEN::start_position())),
                    interruption_token: AtomicBool::new(false),
                    tree: Tree::new(hash_size),
                    _c: PhantomData,
                }
            }
        }
    };

    ( @setters
        [ $( $b_assoc:ident : $b_bound:path | $b_method:ident | $b_default:ty, )* ]
        [ $cur_assoc:ident : $cur_bound:path | $cur_method:ident | $cur_default:ty
          $( , $r_assoc:ident : $r_bound:path | $r_method:ident | $r_default:ty )* ]
    ) => {
        impl< $( $b_assoc, )* $cur_assoc $( , $r_assoc )* >
            EngineBuilder< $( $b_assoc, )* $cur_assoc $( , $r_assoc )* >
        {
            pub fn $cur_method<__S: $cur_bound>(self)
                -> EngineBuilder< $( $b_assoc, )* __S $( , $r_assoc )* >
            {
                EngineBuilder(
                    $( <PhantomData<$b_assoc>>::default(), )*
                    <PhantomData<__S>>::default(),
                    $( <PhantomData<$r_assoc>>::default(), )*
                )
            }
        }

        define_engine_config!(@setters
            [ $( $b_assoc : $b_bound | $b_method | $b_default, )* $cur_assoc : $cur_bound | $cur_method | $cur_default, ]
            [ $( $r_assoc : $r_bound | $r_method | $r_default ),* ]
        );
    };

    ( @setters [ $( $b_assoc:ident : $b_bound:path | $b_method:ident | $b_default:ty, )* ] [] ) => {};
}

define_engine_config! {
    BestMove:    BestMoveStrategy    | best_move    | Unspecified,
    Exploration: ExplorationStrategy | exploration  | Unspecified,
    SearchStep:  SearchStepStrategy  | search_step  | Unspecified,
    TimeManager: TimeManagerStrategy | time_manager | Unspecified,
    Logger:      LoggerTrait         | logger       | NoLogger,
}