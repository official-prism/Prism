// Temporary test file — will be removed after verifying compound params work.

use std::marker::PhantomData;
use crate::{Engine, EngineConfig, engine::{builder::ExplorationStrategy, structures::{IterationStats, SearchStats}}};

#[derive(Debug)]
pub struct TestCompound<Inner: ExplorationStrategy>(PhantomData<Inner>);

crate::define_compound_params! {
    TestCompoundParams<Inner: ExplorationStrategy> {
        Strategies {
            inner: Inner::Params;
        }
        Options {
            ["Compound_Mode"] mode: i32 => 0, 0, 1;
            ["Compound_Flag"] flag: bool => true;
        }
    }
}

impl<Inner: ExplorationStrategy> ExplorationStrategy for TestCompound<Inner> {
    type Params = TestCompoundParams<Inner>;

    fn excute<C: EngineConfig>(
        params: &Self::Params,
        engine: &Engine<C>,
        search_stats: &SearchStats,
        iteration_stats: &IterationStats,
    ) {
        if params.mode() == 0 {
            Inner::excute(params.inner(), engine, search_stats, iteration_stats);
        }
    }
}
