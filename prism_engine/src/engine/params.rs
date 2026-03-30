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

use crate::create_options;

mod macros;

create_options! {
    EngineParams {
        Options {
            //====== General ======
            ["Hash"]         hash:     i64   =>  1024,  1,  524288;
            ["UCI_Chess960"] chess960: bool  =>  false;

            //======= Debug =======
            ["MinimalPrint"] minimal_print:  bool  =>  false;
            ["ItersAsNodes"] iters_as_nodes: bool  =>  false;
        }
        Buttons {
            "Clear",
        }
        Tunables {
            root_pst: f64  =>  3.515,  0.5,  5.0,  0.30,  0.002;
        }
        Variables {
            kld_min: f64  =  0.0025;
        }
    }
}
