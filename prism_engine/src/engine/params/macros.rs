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

#[macro_export]
macro_rules! define_strategy_params {
    (
        $name:ident {
            $(Options {
                $(
                    [$option_key:literal] $option:ident : $option_ty:ty =>
                        $option_default:expr $(, $option_min:expr, $option_max:expr)?;
                )*
            })?
            $(Buttons {
                $(
                    $button_key:literal,
                )*
            })?
            $(Tunables {
                $(
                    $(#[$tune_meta:meta])*
                    $tunable:ident : $tunable_ty:ty =>
                        $tunable_default:expr,
                        $tunable_min:expr,
                        $tunable_max:expr,
                        $tunable_c:expr,
                        $tunable_r:expr;
                )*
            })?
            $(Variables {
                $(
                    $(#[$var_meta:meta])*
                    $variable:ident : $variable_ty:ty =
                        $variable_default:expr;
                )*
            })?
        }
    ) => {
        #[derive(Debug, Clone)]
        #[allow(non_snake_case)]
        pub struct $name {
            $(
                $($option: $option_ty,)*
            )?
            $(
                $(
                #[cfg(feature = "tunable")]
                $tunable: $tunable_ty,
                )*
            )?
            $(
                $($variable: $variable_ty,)*
            )?
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        #[allow(non_snake_case)]
        impl $name {
            pub fn new() -> Self {
                Self {
                    $(
                        $( $option: $option_default,)*
                    )?
                    $(
                        $(
                        #[cfg(feature = "tunable")]
                        $tunable: $tunable_default,
                        )*
                    )?
                    $(
                        $( $variable: $variable_default,)*
                    )?
                }
            }

            $(
                $(
                pub fn $option(&self) -> $option_ty {
                    self.$option.clone()
                }
                )*
            )?

            $(
                $(
                $(#[$tune_meta])*
                #[cfg(feature = "tunable")]
                pub const fn $tunable(&self) -> $tunable_ty {
                    self.$tunable
                }

                $(#[$tune_meta])*
                #[cfg(not(feature = "tunable"))]
                #[inline(always)]
                pub const fn $tunable(&self) -> $tunable_ty {
                    $tunable_default
                }
                )*
            )?

            $(
                $(
                $(#[$var_meta])*
                pub const fn $variable(&self) -> $variable_ty {
                    self.$variable
                }

                $crate::paste::paste! {
                    pub fn [< set_ $variable >] (&mut self, value: $variable_ty) {
                        self.$variable = value;
                    }
                }
                )*
            )?
        }

        impl $crate::StrategyParams for $name {
            fn new() -> Self {
                Self::new()
            }

            fn set_option(&mut self, _name: &str, _value: &str) -> std::result::Result<(), String> {
                $(
                    $(
                    if _name.eq_ignore_ascii_case($option_key) {
                        match _value.parse::<$option_ty>() {
                            Ok(new_value) => {
                                $(
                                    if !($option_min..=$option_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                )?
                                if self.$option == new_value {
                                    return Err(format!("Value of {} is already {}", _name, new_value));
                                }
                                self.$option = new_value;
                                return Ok(());
                            }
                            Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                        }
                    }
                    )*
                )?

                $(
                    $(
                    if _name.eq_ignore_ascii_case($button_key) {
                        return Ok(());
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        if _name.eq_ignore_ascii_case(stringify!($tunable)) {
                            match _value.parse::<$tunable_ty>() {
                                Ok(new_value) => {
                                    if !($tunable_min..=$tunable_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                    if self.$tunable == new_value {
                                        return Err(format!("Value of {} is already {}", _name, new_value));
                                    }
                                    self.$tunable = new_value;
                                    return Ok(());
                                }
                                Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                            }
                        }
                        )*
                    )?
                }

                $(
                    $(
                    if _name.eq_ignore_ascii_case(stringify!($variable)) {
                        match _value.parse::<$variable_ty>() {
                            Ok(new_value) => {
                                if self.$variable == new_value {
                                    return Err(format!("Value of {} is already {}", _name, new_value));
                                }
                                self.$variable = new_value;
                                return Ok(());
                            }
                            Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                        }
                    }
                    )*
                )?

                Err(format!("Unknown option '{}'", _name))
            }

            fn print_options(&self) {
                $(
                    $(
                    {
                        let uci_type = match stringify!($option_ty) {
                            "bool" => "check",
                            "i64"  => "spin",
                            "i32"  => "spin",
                            _      => "string",
                        };
                        let mut default_str = self.$option.to_string();
                        if default_str.is_empty() {
                            default_str = "<empty>".to_string();
                        }
                        print!("option name {} type {} default {}", $option_key, uci_type, default_str);
                        $( print!(" min {} max {}", $option_min, $option_max); )?
                        println!();
                    }
                    )*
                )?

                $(
                    $(
                    {
                        println!("option name {} type button", $button_key);
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let uci_type = match stringify!($tunable_ty) {
                                "bool" => "check",
                                "i64"  => "spin",
                                _      => "string",
                            };
                            print!("option name {} type {} default {}", stringify!($tunable), uci_type, self.$tunable);
                            print!(" min {} max {}", $tunable_min, $tunable_max);
                            println!();
                        }
                        )*
                    )?
                }
            }

            fn print_tunables(&self) {
                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, self.$tunable, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }
                #[cfg(not(feature = "tunable"))]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, $tunable_default, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_engine_params {
    (
        $vis:vis struct $name:ident < $gen:ident : $bound:ident > {
            $( $field_vis:vis $field:ident : $ftype:ty ),* $(,)?
        }
    ) => {
        #[derive(Debug)]
        $vis struct $name<$gen: $bound> {
            $( $field_vis $field: $ftype, )*
        }

        impl<$gen: $bound> $name<$gen> {
            pub fn new() -> Self {
                Self {
                    $( $field: <$ftype>::new(), )*
                }
            }

            $(
                pub fn $field(&self) -> &$ftype {
                    &self.$field
                }
            )*

            pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), String> {
                let unknown_msg = format!("Unknown option '{}'", name);
                $(
                    match self.$field.set_option(name, value) {
                        Err(e) if e == unknown_msg => {},
                        res => return res,
                    }
                )*
                Err(unknown_msg)
            }

            pub fn print_options(&self) {
                $( self.$field.print_options(); )*
            }

            pub fn print_tunables(&self) {
                $( self.$field.print_tunables(); )*
            }
        }

        impl<$gen: $bound> Default for $name<$gen> {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

#[macro_export]
macro_rules! define_compound_params {
    (
        $name:ident {
            Strategies {
                $( $inner_field:ident : $inner_type:ty; )+
            }
            $(Options {
                $(
                    [$option_key:literal] $option:ident : $option_ty:ty =>
                        $option_default:expr $(, $option_min:expr, $option_max:expr)?;
                )*
            })?
            $(Tunables {
                $(
                    $(#[$tune_meta:meta])*
                    $tunable:ident : $tunable_ty:ty =>
                        $tunable_default:expr,
                        $tunable_min:expr,
                        $tunable_max:expr,
                        $tunable_c:expr,
                        $tunable_r:expr;
                )*
            })?
        }
    ) => {
        #[derive(Debug, Clone)]
        #[allow(non_snake_case)]
        pub struct $name {
            $( $inner_field: $inner_type, )+
            $(
                $($option: $option_ty,)*
            )?
            $(
                $(
                #[cfg(feature = "tunable")]
                $tunable: $tunable_ty,
                )*
            )?
        }

        #[allow(non_snake_case)]
        impl $name {
            $( pub fn $inner_field(&self) -> &$inner_type { &self.$inner_field } )+

            $(
                $(
                pub fn $option(&self) -> $option_ty {
                    self.$option.clone()
                }
                )*
            )?

            $(
                $(
                $(#[$tune_meta])*
                #[cfg(feature = "tunable")]
                pub const fn $tunable(&self) -> $tunable_ty {
                    self.$tunable
                }

                $(#[$tune_meta])*
                #[cfg(not(feature = "tunable"))]
                #[inline(always)]
                pub const fn $tunable(&self) -> $tunable_ty {
                    $tunable_default
                }
                )*
            )?
        }

        impl $crate::StrategyParams for $name {
            fn new() -> Self {
                Self {
                    $( $inner_field: <$inner_type as $crate::StrategyParams>::new(), )+
                    $(
                        $($option: $option_default,)*
                    )?
                    $(
                        $(
                        #[cfg(feature = "tunable")]
                        $tunable: $tunable_default,
                        )*
                    )?
                }
            }

            fn set_option(&mut self, _name: &str, _value: &str) -> std::result::Result<(), String> {
                $(
                    $(
                    if _name.eq_ignore_ascii_case($option_key) {
                        match _value.parse::<$option_ty>() {
                            Ok(new_value) => {
                                $(
                                    if !($option_min..=$option_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                )?
                                if self.$option == new_value {
                                    return Err(format!("Value of {} is already {}", _name, new_value));
                                }
                                self.$option = new_value;
                                return Ok(());
                            }
                            Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                        }
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        if _name.eq_ignore_ascii_case(stringify!($tunable)) {
                            match _value.parse::<$tunable_ty>() {
                                Ok(new_value) => {
                                    if !($tunable_min..=$tunable_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                    if self.$tunable == new_value {
                                        return Err(format!("Value of {} is already {}", _name, new_value));
                                    }
                                    self.$tunable = new_value;
                                    return Ok(());
                                }
                                Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                            }
                        }
                        )*
                    )?
                }

                let unknown_msg = format!("Unknown option '{}'", _name);
                $(
                    match self.$inner_field.set_option(_name, _value) {
                        Err(e) if e == unknown_msg => {},
                        res => return res,
                    }
                )+

                Err(unknown_msg)
            }

            fn print_options(&self) {
                $(
                    $(
                    {
                        let uci_type = match stringify!($option_ty) {
                            "bool" => "check",
                            "i64"  => "spin",
                            "i32"  => "spin",
                            _      => "string",
                        };
                        let mut default_str = self.$option.to_string();
                        if default_str.is_empty() {
                            default_str = "<empty>".to_string();
                        }
                        print!("option name {} type {} default {}", $option_key, uci_type, default_str);
                        $( print!(" min {} max {}", $option_min, $option_max); )?
                        println!();
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let uci_type = match stringify!($tunable_ty) {
                                "bool" => "check",
                                "i64"  => "spin",
                                _      => "string",
                            };
                            print!("option name {} type {} default {}", stringify!($tunable), uci_type, self.$tunable);
                            print!(" min {} max {}", $tunable_min, $tunable_max);
                            println!();
                        }
                        )*
                    )?
                }

                $( self.$inner_field.print_options(); )+
            }

            fn print_tunables(&self) {
                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, self.$tunable, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }
                #[cfg(not(feature = "tunable"))]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, $tunable_default, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }

                $( self.$inner_field.print_tunables(); )+
            }
        }
    };

    (
        $name:ident < $( $gen:ident : $bound:path ),+ > {
            Strategies {
                $( $inner_field:ident : $inner_type:ty; )+
            }
            $(Options {
                $(
                    [$option_key:literal] $option:ident : $option_ty:ty =>
                        $option_default:expr $(, $option_min:expr, $option_max:expr)?;
                )*
            })?
            $(Tunables {
                $(
                    $(#[$tune_meta:meta])*
                    $tunable:ident : $tunable_ty:ty =>
                        $tunable_default:expr,
                        $tunable_min:expr,
                        $tunable_max:expr,
                        $tunable_c:expr,
                        $tunable_r:expr;
                )*
            })?
        }
    ) => {
        #[derive(Debug)]
        #[allow(non_snake_case)]
        pub struct $name<$($gen: $bound),+> {
            $( $inner_field: $inner_type, )+
            $(
                $($option: $option_ty,)*
            )?
            $(
                $(
                #[cfg(feature = "tunable")]
                $tunable: $tunable_ty,
                )*
            )?
        }

        impl<$($gen: $bound),+> Clone for $name<$($gen),+> {
            fn clone(&self) -> Self {
                Self {
                    $( $inner_field: self.$inner_field.clone(), )+
                    $(
                        $($option: self.$option.clone(),)*
                    )?
                    $(
                        $(
                        #[cfg(feature = "tunable")]
                        $tunable: self.$tunable.clone(),
                        )*
                    )?
                }
            }
        }

        #[allow(non_snake_case)]
        impl<$($gen: $bound),+> $name<$($gen),+> {
            $( pub fn $inner_field(&self) -> &$inner_type { &self.$inner_field } )+

            $(
                $(
                pub fn $option(&self) -> $option_ty {
                    self.$option.clone()
                }
                )*
            )?

            $(
                $(
                $(#[$tune_meta])*
                #[cfg(feature = "tunable")]
                pub const fn $tunable(&self) -> $tunable_ty {
                    self.$tunable
                }

                $(#[$tune_meta])*
                #[cfg(not(feature = "tunable"))]
                #[inline(always)]
                pub const fn $tunable(&self) -> $tunable_ty {
                    $tunable_default
                }
                )*
            )?
        }

        impl<$($gen: $bound),+> $crate::StrategyParams for $name<$($gen),+> {
            fn new() -> Self {
                Self {
                    $( $inner_field: <$inner_type as $crate::StrategyParams>::new(), )+
                    $(
                        $($option: $option_default,)*
                    )?
                    $(
                        $(
                        #[cfg(feature = "tunable")]
                        $tunable: $tunable_default,
                        )*
                    )?
                }
            }

            fn set_option(&mut self, _name: &str, _value: &str) -> std::result::Result<(), String> {
                $(
                    $(
                    if _name.eq_ignore_ascii_case($option_key) {
                        match _value.parse::<$option_ty>() {
                            Ok(new_value) => {
                                $(
                                    if !($option_min..=$option_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                )?
                                if self.$option == new_value {
                                    return Err(format!("Value of {} is already {}", _name, new_value));
                                }
                                self.$option = new_value;
                                return Ok(());
                            }
                            Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                        }
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        if _name.eq_ignore_ascii_case(stringify!($tunable)) {
                            match _value.parse::<$tunable_ty>() {
                                Ok(new_value) => {
                                    if !($tunable_min..=$tunable_max).contains(&new_value) {
                                        return Err(format!("Value out of range for {}", _name));
                                    }
                                    if self.$tunable == new_value {
                                        return Err(format!("Value of {} is already {}", _name, new_value));
                                    }
                                    self.$tunable = new_value;
                                    return Ok(());
                                }
                                Err(_) => return Err(format!("Incorrect param type for {}", _name)),
                            }
                        }
                        )*
                    )?
                }

                let unknown_msg = format!("Unknown option '{}'", _name);
                $(
                    match self.$inner_field.set_option(_name, _value) {
                        Err(e) if e == unknown_msg => {},
                        res => return res,
                    }
                )+

                Err(unknown_msg)
            }

            fn print_options(&self) {
                $(
                    $(
                    {
                        let uci_type = match stringify!($option_ty) {
                            "bool" => "check",
                            "i64"  => "spin",
                            "i32"  => "spin",
                            _      => "string",
                        };
                        let mut default_str = self.$option.to_string();
                        if default_str.is_empty() {
                            default_str = "<empty>".to_string();
                        }
                        print!("option name {} type {} default {}", $option_key, uci_type, default_str);
                        $( print!(" min {} max {}", $option_min, $option_max); )?
                        println!();
                    }
                    )*
                )?

                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let uci_type = match stringify!($tunable_ty) {
                                "bool" => "check",
                                "i64"  => "spin",
                                _      => "string",
                            };
                            print!("option name {} type {} default {}", stringify!($tunable), uci_type, self.$tunable);
                            print!(" min {} max {}", $tunable_min, $tunable_max);
                            println!();
                        }
                        )*
                    )?
                }

                $( self.$inner_field.print_options(); )+
            }

            fn print_tunables(&self) {
                #[cfg(feature = "tunable")]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, self.$tunable, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }
                #[cfg(not(feature = "tunable"))]
                {
                    $(
                        $(
                        {
                            let kind = if stringify!($tunable_ty) == "i64" { "int" } else { "float" };
                            println!("{}, {}, {}, {}, {}, {}, {}", stringify!($tunable), kind, $tunable_default, $tunable_min, $tunable_max, $tunable_c, $tunable_r);
                        }
                        )*
                    )?
                }

                $( self.$inner_field.print_tunables(); )+
            }
        }
    };
}
