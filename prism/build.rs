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

#[cfg(not(feature = "release"))]
fn main() {
    use std::process::Command;

    //Monty yoink
    let git_commit_hash = Command::new("git")
        .args(["rev-parse", "--short=8", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "nogit000".to_owned());

    let current_date = chrono::Utc::now().format("%Y%m%d").to_string();

    let formatted_name = format!("Prism-dev-{current_date}-{git_commit_hash}");

    println!("cargo:rustc-env=ENGINE_NAME={formatted_name}");

    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "release")]
fn main() {
    let current_date = chrono::Utc::now().format("%Y%m%d").to_string();

    let formatted_name = format!("Prism v{}", env!("ENGINE_VERSION"));

    println!("cargo:rustc-env=ENGINE_NAME={formatted_name}");

    println!("cargo:rerun-if-changed=build.rs");
}