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

// Capability traits and concrete payloads are kept apart, one item per file:
//   `traits/` — capability traits a strategy can require (e.g. `QScore`)
//   `impls/`  — concrete payload structs, named after the node strategy that
//               selects them; each implements whatever traits its set needs.
mod impls;
mod traits;

pub use impls::AvgScoreEdgePayload;
pub use traits::QScore;

/// Marker bound every node/edge payload must satisfy. `()` is a valid payload.
pub trait PayloadType: Default + std::fmt::Debug + Send + Sync + 'static {}
impl<T: Default + std::fmt::Debug + Send + Sync + 'static> PayloadType for T {}