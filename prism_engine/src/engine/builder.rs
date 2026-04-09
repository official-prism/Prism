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

use crate::engine::builder::node_strategy::NodeStrategy;
use crate::engine::logger::{LoggerTrait, NoLogger};
use crate::engine::tree::Tree;
use crate::engine::tree::payload::PayloadType;

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
pub mod node_strategy;

pub use self::best_move_strategy::BestMoveStrategy;
pub use self::exploration_strategy::ExplorationStrategy;
pub use self::search_step_strategy::SearchStepStrategy;
pub use self::time_manager_strategy::TimeManagerStrategy;

pub struct Unspecified;

macro_rules! define_engine_config {
    (
        strategies {
            $( $s_assoc:ident : $s_bound:path | $s_method:ident | $s_default:ty ),+ $(,)?
        }
        services {
            $( $v_assoc:ident : $v_bound:path | $v_method:ident | $v_default:ty ),+ $(,)?
        }
    ) => {
        paste::paste! {
            #[derive(Debug, Default)]
            pub struct CombinedNodePayload< $( [<$s_method:camel Np>] ),+ > {
                $( pub $s_method: [<$s_method:camel Np>], )+
            }

            #[derive(Debug, Default)]
            pub struct CombinedEdgePayload< $( [<$s_method:camel Ep>] ),+ > {
                $( pub $s_method: [<$s_method:camel Ep>], )+
            }
        }

        pub trait EngineConfig: Send + Sync {
            $( type $s_assoc: $s_bound; )+
            $( type $v_assoc: $v_bound; )+
            type NodePayload: PayloadType;
            type EdgePayload: PayloadType;
        }

        pub struct EngineBuilder< $( $s_assoc = $s_default, )+ $( $v_assoc = $v_default ),+ >(
            $( PhantomData<$s_assoc>, )+
            $( PhantomData<$v_assoc>, )+
        );

        impl EngineBuilder< $( $s_default, )+ $( $v_default ),+ > {
            pub fn new() -> Self {
                EngineBuilder(
                    $( <PhantomData<$s_default>>::default(), )+
                    $( <PhantomData<$v_default>>::default(), )+
                )
            }
        }

        define_engine_config!(@setters
            []
            [ $( $s_assoc : $s_bound | $s_method | $s_default ),+ , $( $v_assoc : $v_bound | $v_method | $v_default ),+ ]
        );

        pub struct GenericConfig< $( $s_assoc, )+ $( $v_assoc ),+ >(
            $( PhantomData<$s_assoc>, )+
            $( PhantomData<$v_assoc>, )+
        );

        impl< $( $s_assoc: $s_bound, )+ $( $v_assoc: $v_bound ),+ > EngineConfig for GenericConfig< $( $s_assoc, )+ $( $v_assoc ),+ > {
            $( type $s_assoc = $s_assoc; )+
            $( type $v_assoc = $v_assoc; )+
            type NodePayload = CombinedNodePayload< $( <$s_assoc as $s_bound>::NodePayload ),+ >;
            type EdgePayload = CombinedEdgePayload< $( <$s_assoc as $s_bound>::EdgePayload ),+ >;
        }

        impl< $( $s_assoc: $s_bound, )+ $( $v_assoc: $v_bound ),+ > EngineBuilder< $( $s_assoc, )+ $( $v_assoc ),+ > {
            pub fn build(self) -> Engine<GenericConfig< $( $s_assoc, )+ $( $v_assoc ),+ >> {
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
    strategies {
        BestMove:    BestMoveStrategy    | best_move    | Unspecified,
        Exploration: ExplorationStrategy | exploration  | Unspecified,
        SearchStep:  SearchStepStrategy  | search_step  | Unspecified,
        Node:        NodeStrategy        | node         | Unspecified,
    }
    services {
        TimeManager: TimeManagerStrategy | time_manager | Unspecified,
        Logger:      LoggerTrait         | logger       | NoLogger,
    }
}