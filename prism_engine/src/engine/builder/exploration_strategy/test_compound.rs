use std::marker::PhantomData;
use crate::{Engine, EngineConfig, engine::{builder::ExplorationStrategy, structures::{IterationStats, SearchStats}}};

#[derive(Debug)]
pub struct DummyExploration;

crate::define_strategy_params! {
    DummyParams {
        Options {
            ["Dummy_Alpha"] alpha: i32 => 5, 1, 10;
            ["Dummy_Beta"]  beta:  bool => false;
        }
    }
}

impl ExplorationStrategy for DummyExploration {
    type Params = DummyParams;

    fn excute<C: EngineConfig>(
        _params: &Self::Params,
        _engine: &Engine<C>,
        _search_stats: &SearchStats,
        _iteration_stats: &IterationStats,
    ) {}
}

#[derive(Debug)]
pub struct TestCompound<InnerA: ExplorationStrategy, InnerB: ExplorationStrategy>(
    PhantomData<(InnerA, InnerB)>,
);

crate::define_compound_params! {
    TestCompoundParams<InnerA: ExplorationStrategy, InnerB: ExplorationStrategy> {
        Strategies {
            inner_a: InnerA::Params;
            inner_b: InnerB::Params;
        }
        Options {
            ["Compound_Mode"] mode: i32 => 0, 0, 1;
            ["Compound_Flag"] flag: bool => true;
        }
    }
}

impl<InnerA: ExplorationStrategy, InnerB: ExplorationStrategy> ExplorationStrategy
    for TestCompound<InnerA, InnerB>
{
    type Params = TestCompoundParams<InnerA, InnerB>;

    fn excute<C: EngineConfig>(
        params: &Self::Params,
        engine: &Engine<C>,
        search_stats: &SearchStats,
        iteration_stats: &IterationStats,
    ) {
        if params.mode() == 0 {
            InnerA::excute(params.inner_a(), engine, search_stats, iteration_stats);
        } else {
            InnerB::excute(params.inner_b(), engine, search_stats, iteration_stats);
        }
    }
}