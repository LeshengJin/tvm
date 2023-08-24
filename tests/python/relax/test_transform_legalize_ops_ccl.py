# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest

import tvm
import tvm.testing
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_allreduce():
    # fmt: off
    @tvm.script.ir_module
    class AllReduce:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 10), "float32"):
            gv: R.Tensor((10, 10), "float32") = R.ccl.allreduce(x, "sum")
            return gv
        
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            gv: R.Tensor((10, 10), dtype="float32") = R.call_pure_packed("runtime.disco.allreduce", x, R.shape([0]), sinfo_args=(R.Tensor((10, 10), dtype="float32"),))
            return gv
    # fmt: on

    mod = LegalizeOps()(AllReduce)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
