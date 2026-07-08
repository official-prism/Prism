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

pub trait StrategyParams: std::fmt::Debug + Clone + Send + Sync {
    fn new() -> Self;
    fn set_option(&mut self, name: &str, value: &str) -> std::result::Result<(), String>;
    fn print_options(&self);
    fn print_tunables(&self);
}

/// Shared parameter type for strategies that expose no options, tunables or
/// variables. Use `type Params = EmptyParams;` instead of declaring a fresh
/// empty struct per strategy.
#[derive(Debug, Clone)]
pub struct EmptyParams;

impl StrategyParams for EmptyParams {
    fn new() -> Self {
        Self
    }

    fn set_option(&mut self, name: &str, _value: &str) -> std::result::Result<(), String> {
        Err(format!("Unknown option '{}'", name))
    }

    fn print_options(&self) {}

    fn print_tunables(&self) {}
}
