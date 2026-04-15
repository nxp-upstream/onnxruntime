# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Implements ONNX's backend API.
"""

from onnx.backend.base import BackendRep

from onnxruntime import RunOptions

# Allowlist of RunOptions attributes that are safe to set via the backend API.
# Other attributes (e.g. terminate, training_mode) are intentionally excluded.
# SessionOptions keys forwarded from run_model() are silently ignored here.
_ALLOWED_RUN_OPTIONS = frozenset({
    "log_severity_level",
    "log_verbosity_level",
    "logid",
    "only_execute_path_to_fetches",
})


class OnnxRuntimeBackendRep(BackendRep):
    """
    Computes the prediction for a pipeline converted into
    an :class:`onnxruntime.InferenceSession` node.
    """

    def __init__(self, session):
        """
        :param session: :class:`onnxruntime.InferenceSession`
        """
        self._session = session

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        """
        Computes the prediction.
        See :meth:`onnxruntime.InferenceSession.run`.
        """

        options = RunOptions()
        for k, v in kwargs.items():
            if k in _ALLOWED_RUN_OPTIONS:
                setattr(options, k, v)
            # Unknown keys are silently ignored: run_model() forwards the same kwargs
            # used for SessionOptions, so those keys will arrive here and must not raise.

        if isinstance(inputs, list):
            inps = {}
            for i, inp in enumerate(self._session.get_inputs()):
                inps[inp.name] = inputs[i]
            outs = self._session.run(None, inps, options)
            if isinstance(outs, list):
                return outs
            else:
                output_names = [o.name for o in self._session.get_outputs()]
                return [outs[name] for name in output_names]
        else:
            inp = self._session.get_inputs()
            if len(inp) != 1:
                raise RuntimeError(f"Model expect {len(inp)} inputs")
            inps = {inp[0].name: inputs}
            return self._session.run(None, inps, options)
