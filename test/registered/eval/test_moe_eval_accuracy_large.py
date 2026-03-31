"""
Usage:
python -m unittest test_moe_eval_accuracy_large.TestMoEEvalAccuracyLarge.test_mmlu
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=500, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=500, suite="stage-b-test-2-gpu-large-amd")


class TestMoEEvalAccuracyLarge(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.93
    gsm8k_chat_template_kwargs = {"enable_thinking": False}

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--log-level-http",
                "warning",
                "--tp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
