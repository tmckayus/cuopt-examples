# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions: The above copyright notice and this
# permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/python

import argparse
from pathlib import Path
import json
import cuopt_mps_parser

def _mps_parse(LP_problem_data, tolerances, time_limit, iteration_limit, no_var_names):

    if isinstance(LP_problem_data, cuopt_mps_parser.parser_wrapper.DataModel):
        model = LP_problem_data
    else:
        model = cuopt_mps_parser.ParseMps(LP_problem_data)

    problem_data = cuopt_mps_parser.toDict(model, json=True)
    if no_var_names:
        problem_data.pop("variable_names")

    problem_data["solver_config"] = {}
    if tolerances is not None:
        problem_data["solver_config"]["tolerances"] = tolerances
    if time_limit is not None:
        problem_data["solver_config"]["time_limit"] = time_limit
    if iteration_limit is not None:
        problem_data["solver_config"]["iteration_limit"] = iteration_limit
    return problem_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Read an MPS file and write an equivalent cuOpt JSON file."
    )
    parser.add_argument(
        "file",
        type=str,
        help="Filename"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output file name. If the file ends with '.mps', "
        "the default is to replace '.mps' with '.json' to get the output file name, "
        "otherwise append '.json' to the file name."
    )
    parser.add_argument(
        "-tol",
        "--tolerances",
        type=str,
        default=None,
        help="Filename or JSON string containing "
        "tolerances for LP problem type",
    )
    parser.add_argument(
        "-tl",
        "--time-limit",
        default=None,
        type=int,
        help="LP timit in milliseconds"
    )
    parser.add_argument(
        "-il",
        "--iteration-limit",
        default=None,
        type=int,
        help="LP iteration limit"
    )
    parser.add_argument(
        "-nv",
        "--no-var-names",
        action="store_true",
        help="If set, leave the variable names out of the cuopt JSON dataset. Default is False."
    )

    args = parser.parse_args()
    data = _mps_parse(args.file, args.tolerances, args.time_limit, args.iteration_limit, args.no_var_names)
    if args.output:
        out = args.output
    elif args.file.endswith(".mps"):
        out = Path(args.file).with_suffix(".json")
    else:
        out = args.file + ".json"
    print(f"Writing {out}")
    with open(out, "w") as f:
        json.dump(data, f)
