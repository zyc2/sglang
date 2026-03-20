from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import is_in_amd_ci, is_in_ci, write_github_step_summary


class GSM8KMixin:
    """Mixin for few-shot GSM8K evaluation.

    Required attributes on the test class:
        base_url: str
        gsm8k_accuracy_thres: float
    """

    gsm8k_accuracy_thres: float
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_questions: int = 200
    gsm8k_parallel: int = 128

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=self.gsm8k_parallel,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_accuracy_thres)

        if self.gsm8k_accept_length_thres is not None:
            server_info = requests.get(self.base_url + "/server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.gsm8k_accept_length_thres)


class MMLUMixin:
    """Mixin for MMLU evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        mmlu_score_threshold: float
    """

    mmlu_score_threshold: float
    mmlu_num_examples: int = 5000
    mmlu_num_threads: int = 1024

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=self.mmlu_num_examples,
            num_threads=self.mmlu_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_mmlu\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.mmlu_score_threshold)


class HumanEvalMixin:
    """Mixin for HumanEval evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        humaneval_score_threshold: float
    """

    humaneval_score_threshold: float
    humaneval_score_threshold_amd: Optional[float] = None
    humaneval_num_threads: int = 1024

    def test_human_eval(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="humaneval",
            num_examples=None,
            num_threads=self.humaneval_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_human_eval\n{metrics['score']=:.4f}\n")

        threshold = self.humaneval_score_threshold
        if is_in_amd_ci() and self.humaneval_score_threshold_amd is not None:
            threshold = self.humaneval_score_threshold_amd

        self.assertGreaterEqual(metrics["score"], threshold)


class MGSMEnMixin:
    """Mixin for MGSM English evaluation.

    Required attributes on the test class:
        base_url: str
        model: str
        mgsm_en_score_threshold: float
    """

    mgsm_en_score_threshold: float
    mgsm_en_num_examples: Optional[int] = None
    mgsm_en_num_threads: int = 1024

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=self.mgsm_en_num_examples,
            num_threads=self.mgsm_en_num_threads,
        )

        metrics = run_eval(args)

        if is_in_ci():
            write_github_step_summary(f"### test_mgsm_en\n{metrics['score']=:.4f}\n")

        self.assertGreaterEqual(metrics["score"], self.mgsm_en_score_threshold)
