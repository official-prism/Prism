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

#[allow(unused)]
pub fn seconds_to_string(seconds: u128) -> String {
    let hh = seconds / 3600;
    let mm = (seconds - (hh * 3600)) / 60;
    let ss = seconds - (hh * 3600) - (mm * 60);

    let mut result = String::new();

    if hh > 0 {
        result.push_str(format!("{}h ", hh).as_str());
    }

    if hh > 0 || mm > 0 {
        result.push_str(format!("{}m ", mm).as_str());
    }

    result.push_str(format!("{}s", ss).as_str());

    result.trim().to_string()
}

#[allow(unused)]
pub fn time_to_string(miliseconds: u128) -> String {
    let hh = miliseconds / 3600000;
    let mm = (miliseconds - (hh * 3600000)) / 60000;
    let ss = (miliseconds - (mm * 60000) - (hh * 3600000)) as f32 / 1000.0;

    let mut result = String::new();

    if hh > 0 {
        result.push_str(format!("{}h ", hh).as_str());
    }

    if mm > 0 {
        result.push_str(format!("{}m ", mm).as_str());
    }

    if ss >= 1.0 || mm > 0 {
        result.push_str(format!("{:.2}s", ss).as_str());
    } else {
        result.push_str(format!("{:.0}ms", ss * 1000.0).as_str());
    }

    result.trim().to_string()
}

#[allow(unused)]
pub fn number_to_string(number: u128) -> String {
    if number < 1000 {
        return format!("{number}");
    }

    let number = round(number as f64 / 1000.0);

    if number < 1000.0 {
        return format!("{:.2}K", number);
    }

    let number = round(number / 1000.0);

    if number < 1000.0 {
        return format!("{:.2}M", number);
    }

    format!("{:.2}B", round(number / 1000.0))
}

#[allow(unused)]
pub fn bytes_to_string(number: u128) -> String {
    if number < 1024 {
        return format!("{number}");
    }

    let number = round(number as f64 / 1024.0);

    if number < 1024.0 {
        return format!("{:.2}K", number);
    }

    let number = round(number / 1024.0);

    if number < 1024.0 {
        return format!("{:.2}M", number);
    }

    format!("{:.2}G", round(number / 1024.0))
}

fn round(number: f64) -> f64 {
    (number * 100.0).round() / 100.0
}
