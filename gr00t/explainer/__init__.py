# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace capture for the GR00T N1.7 interactive explainer.

The explainer website renders a single forward pass of the model (one observation ->
one action chunk) as a step-by-step visualization. This package produces the data it
reads:

- :mod:`gr00t.explainer.schema` defines the ``groot-trace/1`` file format (a JSON
  manifest plus little-endian binary parts) and the list of tensors a trace contains.
- :mod:`gr00t.explainer.synthetic` generates a shape-correct trace with random data
  using only numpy, so the website renders before any real trace exists.
- :mod:`gr00t.explainer.capture` runs the real model under hooks and records the same
  tensors. It requires torch, a checkpoint, and (practically) a GPU.

Nothing in this package's top level imports torch; the synthetic path stays importable
in environments without it.
"""
