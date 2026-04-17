# Modular Search Architecture Plan

## Overview of Final Strategy Slots

The engine will have 5 strategy slots in `EngineConfig`:

| Slot | Responsibility |
|---|---|
| `SearchStepStrategy` | Orchestrates one full MCTS iteration |
| `ExplorationStrategy` | Selects which child to descend into |
| `QStrategy` | Owns Q storage format and read semantics |
| `BackpropagationStrategy` | Delivers leaf eval up the path |
| `BestMoveStrategy` | Selects the final move after search completes |

---

## Step 1: Custom Node/Edge Payloads

Each strategy can define strategy-specific data that lives on every node or edge at compile time — no `Box<dyn Any>`, no runtime cost.

Each strategy trait gets two associated types:
- `type NodePayload: Default + Send + Sync`
- `type EdgePayload: Default + Send + Sync`

`EngineConfig` collects all of them into a single combined payload type (likely a tuple or flat struct). `Node` and `Edge` become generic over `C: EngineConfig` and store `payload: C::NodePayload` / `payload: C::EdgePayload`.

Strategies access their slice of the payload through their own associated type — the `EngineConfig` mapping is what connects them. The builder macro generates the combined type automatically.

This is the foundation everything else depends on. Do this first.

---

## Step 2: `QStrategy` Trait

Owns everything about what the Q field means — how it is written, how it is read, and where virtual loss is applied.

Methods:
- `write(edge_payload, leaf_eval)` — called by backprop at each node on the path; decides whether to accumulate a raw sum, update a running average, blend via soft minimax, etc.
- `read_for_exploration(edge_payload, visits, virtual_loss_count) -> f64` — called by exploration for each child; applies virtual loss correction here, not in the search step
- `read_for_backprop(edge_payload, visits) -> f64` — called by backprop to get a node's Q value to propagate to its parent; may differ from exploration read (e.g., no virtual loss needed here)

Virtual loss lives as an atomic counter on `Node` (not in the payload system — it's universal). `QStrategy::read_for_exploration` is the only place that consumes it. The search step applies virtual loss by incrementing the counter on descent and decrementing on backprop, with no strategy-specific logic.

`QStrategy` likely needs `type EdgePayload` for strategies like Thompson Sampling that store distribution parameters per edge.

---

## Step 3: Refine `ExplorationStrategy`

The current signature is wrong — it operates at the engine level. It needs to operate at the node level during tree descent.

New interface:
- `select_child(node, children_payloads, parent_payload, q_strategy, params) -> usize` — single child selection
- Optionally: `select_batch(node, ..., batch_size) -> Vec<usize>` for multigather

The strategy reads children's Q values exclusively through `QStrategy::read_for_exploration`. It does not touch raw storage directly.

`NodePayload` on this strategy can hold a per-node cache (e.g., precomputed `sqrt(parent_N)`) that survives across the batch if multigather is used.

---

## Step 4: Add `BackpropagationStrategy` Trait

Owns which nodes get updated and in what order after a leaf is evaluated.

Interface:
- `backpropagate(path, leaf_eval, tree, q_strategy, params)` — walks the path (root → leaf or leaf → root depending on impl) and calls `q_strategy.write()` at each node

For standard MCTS this is a simple loop up the path. For MCGS, the strategy has access to the full tree and can update non-path nodes. The trait signature passes `&Tree` specifically to allow this — the strategy is not limited to the path.

Draw probability and other NN output heads are passed alongside `leaf_eval` so strategies can handle them (averaging draw separately, ignoring it, folding it in, etc.).

---

## Step 5: Refine `BestMoveStrategy`

Post-search, after the search loop exits.

Interface:
- `select(root_node, tree, q_strategy, params) -> Move`

Receives the root node and full tree. Reads children's Q, N, and policy through the same `q_strategy.read_for_exploration` interface as exploration, so it sees the same Q values. No virtual loss correction needed here — the counter should be zero after search completes.

Wire this into `engine.rs` where `best_move()` is currently hardcoded.

---

## Step 6: `SearchStepStrategy` as Orchestrator

The classical implementation calls the other strategies in order:

1. Descend via `ExplorationStrategy::select_child` until a leaf
2. Apply virtual loss on the descended path
3. Evaluate the leaf (NN call)
4. Call `BackpropagationStrategy::backpropagate`
5. Remove virtual loss from the path

The trait's `execute()` signature stays at the engine level — it takes `&Engine<C>` and returns `IterationStats`. The implementation is what changes. The strategy has access to all sub-strategies through `C: EngineConfig`.

For MCGS or beam search, a completely different `SearchStepStrategy` is written that ignores the classical descent pattern entirely and orchestrates the sub-strategies differently — or doesn't use them at all.

---

## Step 7: Wire Everything Into `EngineConfig` and `EngineBuilder`

`EngineConfig` gains two new associated types: `QStrategy` and `BackpropagationStrategy`.

`EngineBuilder` gains two new builder methods, both required before `.build()` compiles (phantom type enforcement as currently done).

`EngineParams` gains the corresponding params fields for both new strategies.

The macro system (`define_strategy_params!`, `register_strategy!`) handles param registration automatically.

---

## Implementation Order

```
1. Custom node/edge payload system
   └── Node<C> and Edge<C> generic over EngineConfig
   └── EngineConfig::NodePayload / EdgePayload combined types
   └── Builder macro generates combined payload struct

2. QStrategy trait
   └── Default impl: raw sum storage, Q/N on read, virtual loss on exploration read
   └── Wire into Edge (replaces raw score field)
   └── Virtual loss counter on Node (atomic, universal)

3. BackpropagationStrategy trait
   └── Default impl: walk path leaf→root, call q_strategy.write() at each step
   └── Wire into EngineConfig, EngineBuilder, EngineParams

4. Fix ExplorationStrategy signature
   └── Node-level select_child using q_strategy.read_for_exploration

5. Fix BestMoveStrategy signature
   └── Returns Move, receives tree + root index

6. Wire SearchStepStrategy classical impl
   └── Calls exploration → leaf eval → backprop in order
   └── Manages virtual loss counter

7. Wire BestMoveStrategy into engine post-search loop
```

After step 7, all four of Naphthalin's concerns are covered with proper strategy slots, the payload system enables sampling strategies and distribution-based approaches, and virtual loss is handled universally without polluting any individual strategy.

---

## Mapping to Naphthalin's Four Concerns

| Naphthalin's Concern | Valkyrie Slot | Status after plan |
|---|---|---|
| 1. Selection of move to play | `BestMoveStrategy` | Covered (step 5 + 7) |
| 2. Selection of move to explore | `ExplorationStrategy` | Covered (step 3) |
| 3. Result propagation | `BackpropagationStrategy` | Covered (step 4) |
| 4. Node value recalculation | `QStrategy` | Covered (step 2) |