/*
    This file is part of Valkyrie.

    Copyright (C) 2026 Tomasz Jaworski

    Valkyrie is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Valkyrie is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Valkyrie.  If not, see <https://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with the ONNX Runtime and/or NVIDIA TensorRT libraries
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective ONNX Runtime and NVIDIA license
    agreements, the licensors of this Program grant you additional
    permission to convey the resulting work.
*/

macro_rules! register_strategy {
    ($($name:ident),*) => {
        $(
            pub mod $name;
            pub use self::$name::*;
        )*
    };
}

macro_rules! define_engine_config {
    (
        general: $general_ty:ty,
        with_params {
            $( $s_assoc:ident : $s_bound:path | $s_method:ident | $s_default:ty ),+ $(,)?
        }
        without_params {
            $( $a_assoc:ident : $a_bound:path | $a_method:ident | $a_default:ty ),+ $(,)?
        }
    ) => {
        pub trait EngineConfig: Send + Sync {
            $( type $s_assoc: $s_bound; )+
            $( type $a_assoc: $a_bound; )+
            type NodePayload: $crate::engine::tree::payload::PayloadType;
            type EdgePayload: $crate::engine::tree::payload::PayloadType;
        }

        pub struct EngineBuilder<
            $( $s_assoc = $s_default, )+
            $( $a_assoc = $a_default, )+
        >(
            $( ::std::marker::PhantomData<$s_assoc>, )+
            $( ::std::marker::PhantomData<$a_assoc>, )+
        );

        impl EngineBuilder<
            $( $s_default, )+
            $( $a_default, )+
        > {
            pub fn new() -> Self {
                EngineBuilder(
                    $( <::std::marker::PhantomData<$s_default>>::default(), )+
                    $( <::std::marker::PhantomData<$a_default>>::default(), )+
                )
            }
        }

        define_engine_config!(@setters
            []
            [
                $( $s_assoc : $s_bound | $s_method | $s_default ),+ ,
                $( $a_assoc : $a_bound | $a_method | $a_default ),+
            ]
        );

        pub struct GenericConfig<
            $( $s_assoc, )+
            $( $a_assoc, )+
        >(
            $( ::std::marker::PhantomData<$s_assoc>, )+
            $( ::std::marker::PhantomData<$a_assoc>, )+
        );

        impl<
            $( $s_assoc: $s_bound, )+
            $( $a_assoc: $a_bound, )+
        > EngineConfig for GenericConfig<
            $( $s_assoc, )+
            $( $a_assoc, )+
        > {
            $( type $s_assoc = $s_assoc; )+
            $( type $a_assoc = $a_assoc; )+
            type NodePayload = Node::NodePayload;
            type EdgePayload = Node::EdgePayload;
        }

        impl<
            $( $s_assoc: $s_bound, )+
            $( $a_assoc: $a_bound, )+
        > EngineBuilder<
            $( $s_assoc, )+
            $( $a_assoc, )+
        > {
            pub fn build(self) -> $crate::Engine<GenericConfig<
                $( $s_assoc, )+
                $( $a_assoc, )+
            >> {
                let params = EngineParams::<GenericConfig<
                    $( $s_assoc, )+
                    $( $a_assoc, )+
                >>::new();
                let hash_size = params.general().hash() as usize;
                $crate::Engine {
                    params,
                    position: ::valkyrie_chess::ChessPosition::from(
                        ::valkyrie_chess::ChessBoard::from(
                            &::valkyrie_chess::FEN::start_position()
                        )
                    ),
                    interruption_token: ::std::sync::atomic::AtomicBool::new(false),
                    tree: $crate::engine::tree::Tree::new(hash_size),
                    _c: ::std::marker::PhantomData,
                }
            }
        }

        #[derive(Debug)]
        pub struct EngineParams<C: EngineConfig> {
            general: $general_ty,
            $( $s_method: <C::$s_assoc as $s_bound>::Params, )+
        }

        impl<C: EngineConfig> EngineParams<C> {
            pub fn new() -> Self {
                Self {
                    general: <$general_ty as $crate::StrategyParams>::new(),
                    $( $s_method: <<C::$s_assoc as $s_bound>::Params as $crate::StrategyParams>::new(), )+
                }
            }

            pub fn general(&self) -> &$general_ty {
                &self.general
            }

            $(
                pub fn $s_method(&self) -> &<C::$s_assoc as $s_bound>::Params {
                    &self.$s_method
                }
            )+

            pub fn set_option(&mut self, name: &str, value: &str) -> ::std::result::Result<(), String> {
                let unknown_msg = format!("Unknown option '{}'", name);
                match $crate::StrategyParams::set_option(&mut self.general, name, value) {
                    Err(e) if e == unknown_msg => {},
                    res => return res,
                }
                $(
                    match $crate::StrategyParams::set_option(&mut self.$s_method, name, value) {
                        Err(e) if e == unknown_msg => {},
                        res => return res,
                    }
                )+
                Err(unknown_msg)
            }

            pub fn print_options(&self) {
                $crate::StrategyParams::print_options(&self.general);
                $( $crate::StrategyParams::print_options(&self.$s_method); )+
            }

            pub fn print_tunables(&self) {
                $crate::StrategyParams::print_tunables(&self.general);
                $( $crate::StrategyParams::print_tunables(&self.$s_method); )+
            }
        }

        impl<C: EngineConfig> Default for EngineParams<C> {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    ( @setters
        [ $( $b_assoc:ident : $b_bound:path | $b_method:ident | $b_default:ty, )* ]
        [ $cur_assoc:ident : $cur_bound:path | $cur_method:ident | $cur_default:ty
          $( , $r_assoc:ident : $r_bound:path | $r_method:ident | $r_default:ty )* $(,)? ]
    ) => {
        impl< $( $b_assoc, )* $cur_assoc $( , $r_assoc )* >
            EngineBuilder< $( $b_assoc, )* $cur_assoc $( , $r_assoc )* >
        {
            pub fn $cur_method<__S: $cur_bound>(self)
                -> EngineBuilder< $( $b_assoc, )* __S $( , $r_assoc )* >
            {
                EngineBuilder(
                    $( <::std::marker::PhantomData<$b_assoc>>::default(), )*
                    <::std::marker::PhantomData<__S>>::default(),
                    $( <::std::marker::PhantomData<$r_assoc>>::default(), )*
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
