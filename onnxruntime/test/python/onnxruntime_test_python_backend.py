# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest

import numpy as np
from helper import get_name
from numpy.testing import assert_allclose

import onnxruntime as onnxrt
import onnxruntime.backend as backend


class TestBackend(unittest.TestCase):
    def test_run_model(self):
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_allocation_plan_works_with_only_execute_path_to_fetches_option(self):
        """
               (inp0)  (inp1)
                  |  \\/  |
                  |  /\\  |
                 Add    Sub
                  |      |
              (tsor0)  (tsor1)
                  |      |
                 Neg    Neg
                  |      |
              (outp0)  (outp1)

        In this model, tsor0 and tsor1 has the same size. Allocation plan sets tsor1 to re-uses tsor0's memory.
        With run_options.only_execute_path_to_fetches == True and only to fetch outp1, the Add op is not executed.
        As a result tsor0 is not allocated through computation. It would fail to allocate tsor1 via re-use tsor0.
        This case is handled specifically in ExecutionFrame::AllocateAsPerAllocationPlan().
        This test is to ensure that the case is covered.
        """
        providers = onnxrt.get_available_providers()
        has_qnn_ep = "QNNExecutionProvider" in providers
        name = get_name("alloc_tensor_reuse.onnx")
        sess = onnxrt.InferenceSession(name, providers=providers)

        run_options = onnxrt.RunOptions()
        run_options.only_execute_path_to_fetches = True
        inp0, inp1 = np.ones((10,), dtype=np.float32), np.ones((10,), dtype=np.float32)

        session_run_results = sess.run(["outp0"], {"inp0": inp0, "inp1": inp1}, run_options)
        if has_qnn_ep:
            # QNN EP runs fp32 with fp16 precision, so relax tolerance.
            assert_allclose(session_run_results[0], -(inp0 + inp1), rtol=1e-6, atol=1e-6)
        else:
            assert_allclose(session_run_results[0], -(inp0 + inp1))

        session_run_results = sess.run(["outp1"], {"inp0": inp0, "inp1": inp1}, run_options)
        if has_qnn_ep:
            # QNN EP runs fp32 with fp16 precision, so relax tolerance.
            assert_allclose(session_run_results[0], -(inp0 - inp1), rtol=1e-6, atol=1e-6)
        else:
            assert_allclose(session_run_results[0], -(inp0 - inp1))


class TestBackendKwargsAllowlist(unittest.TestCase):
    """Tests that the SessionOptions/RunOptions kwargs allowlist correctly blocks
    dangerous attributes and allows safe ones, preventing arbitrary file writes
    through user-controlled kwargs."""

    def test_blocked_session_option_optimized_model_filepath_raises(self):
        """optimized_model_filepath is a known SessionOptions attr but is not in the allowlist.
        It must raise RuntimeError to prevent arbitrary file overwrites."""
        name = get_name("mul_1.onnx")
        with self.assertRaises(RuntimeError):
            backend.prepare(name, optimized_model_filepath="/tmp/should_not_exist.bin")

    def test_blocked_session_option_profile_file_prefix_raises(self):
        """profile_file_prefix is a known SessionOptions attr but is not in the allowlist.
        It must raise RuntimeError to prevent arbitrary file writes via profiling output."""
        name = get_name("mul_1.onnx")
        with self.assertRaises(RuntimeError):
            backend.prepare(name, profile_file_prefix="/tmp/should_not_exist_profile")

    def test_blocked_session_option_enable_profiling_raises(self):
        """enable_profiling is excluded from the allowlist because it causes uncontrolled
        file writes (profiling JSON) to the current working directory."""
        name = get_name("mul_1.onnx")
        with self.assertRaises(RuntimeError):
            backend.prepare(name, enable_profiling=True)

    def test_unknown_kwarg_is_silently_ignored(self):
        """A kwarg that is not a SessionOptions attribute at all must be silently ignored.
        This preserves backward compatibility for callers who pass extra kwargs."""
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name, totally_unknown_kwarg="foo")
        self.assertIsNotNone(rep)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_safe_session_option_graph_optimization_level_is_accepted(self):
        """graph_optimization_level is in the allowlist and must be accepted without error."""
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name, graph_optimization_level=onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL)
        self.assertIsNotNone(rep)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_safe_session_option_intra_op_num_threads_is_accepted(self):
        """intra_op_num_threads is in the allowlist and must be accepted without error."""
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name, intra_op_num_threads=1)
        self.assertIsNotNone(rep)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)

    def test_blocked_run_option_terminate_is_silently_ignored(self):
        """terminate is a known RunOptions attr but is not in _ALLOWED_RUN_OPTIONS.
        run_model() forwards the same kwargs to both prepare() and rep.run(), so blocked
        RunOptions attrs must be silently ignored — not raise — to avoid breaking run_model()."""
        name = get_name("mul_1.onnx")
        rep = backend.prepare(name)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        res = rep.run(x, terminate=True)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(res[0], output_expected, rtol=1e-05, atol=1e-08)


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
