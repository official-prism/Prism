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

use std::fmt::{Display, Formatter, Result};

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Side(u8);
impl Side {
    pub const WHITE: Self = Self(0);
    pub const BLACK: Self = Self(1);

    #[inline]
    pub const fn get_value(&self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn flipped(&self) -> Self {
        Self(1 - self.0)
    }

    #[inline]
    pub const fn flip(&mut self) {
        self.0 = 1 - self.0;
    }
}

impl From<bool> for Side {
    fn from(value: bool) -> Self {
        Self(u8::from(value))
    }
}

impl From<u8> for Side {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<Side> for u8 {
    fn from(value: Side) -> Self {
        value.0
    }
}

impl From<Side> for usize {
    fn from(value: Side) -> Self {
        value.0 as usize
    }
}

impl From<Side> for String {
    fn from(value: Side) -> Self {
        if value == Side::WHITE {
            "White"
        } else {
            "Black"
        }
        .to_string()
    }
}

impl Display for Side {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result {
        write!(formatter, "{}", String::from(*self))
    }
}
