import argparse
import os
import shutil
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import torch

from sim.utils.helpers import (
    class_to_dict,
    export_policy_as_jit,
    export_policy_as_onnx,
    get_load_path,
    parse_sim_params,
    set_seed,
    update_cfg_from_args,
    update_class_from_dict,
)


class TestHelpers(unittest.TestCase):

    def test_class_to_dict(self) -> None:
        class TestClass:
            def __init__(self) -> None:
                self.a = 1
                self.b = [2, 3]
                self._c = 4  # should be ignored

        obj = TestClass()
        result = class_to_dict(obj)
        self.assertEqual(result, {"a": 1, "b": [2, 3]})

    def test_update_class_from_dict(self) -> None:
        class TestClass:
            def __init__(self) -> None:
                self.a = 1
                self.b = 2

        obj = TestClass()
        update_class_from_dict(obj, {"a": 10, "b": 20})
        self.assertEqual(obj.a, 10)
        self.assertEqual(obj.b, 20)

    def test_set_seed(self) -> None:
        set_seed(42)
        self.assertEqual(np.random.randint(0, 10000), 7270)
        self.assertEqual(torch.randint(0, 10000, (1,)).item(), 5944)

    def test_parse_sim_params(self) -> None:
        args = argparse.Namespace(
            physics_engine="SIM_PHYSX", use_gpu=True, subscenes=2, use_gpu_pipeline=True, num_threads=4
        )
        cfg = {}
        sim_params = parse_sim_params(args, cfg)
        self.assertTrue(sim_params.physx.use_gpu)
        self.assertEqual(sim_params.physx.num_subscenes, 2)

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.listdir")
    def test_get_load_path(self, mock_listdir, mock_open, mock_exists, mock_makedirs) -> None:
        root = "test_runs"
        mock_exists.return_value = True
        mock_listdir.return_value = ["2023_Jan01_00-00-00", "2023_Mar15_12-30-45", "2023_Dec31_23-59-59"]

        # Mock the open function to simulate file existence
        mock_open.side_effect = [
            mock_open(read_data="test").return_value,
            mock_open(read_data="test").return_value,
            mock_open(read_data="test").return_value,
            mock_open(read_data="test").return_value,
        ]

        # Test default behavior (latest run, latest model)
        path = get_load_path(root)
        self.assertTrue(path.endswith(os.path.join("2023_Dec31_23-59-59", "model_1.pt")))

        # Test with specific run
        path = get_load_path(root, load_run="2023_Mar15_12-30-45")
        self.assertTrue(path.endswith(os.path.join("2023_Mar15_12-30-45", "model_1.pt")))

        # Test with specific checkpoint
        path = get_load_path(root, load_run="2023_Jan01_00-00-00", checkpoint=2)
        self.assertTrue(path.endswith(os.path.join("2023_Jan01_00-00-00", "model_2.pt")))

        # Test with non-existent run
        with self.assertRaises(ValueError):
            get_load_path(root, load_run="non_existent_run")

    def test_update_cfg_from_args(self) -> None:
        env_cfg = type("obj", (object,), {"env": type("obj", (object,), {"num_envs": 1})()})
        cfg_train = type("obj", (object,), {"seed": 0, "runner": type("obj", (object,), {"max_iterations": 100})()})
        args = argparse.Namespace(
            num_envs=10,
            seed=42,
            max_iterations=200,
            resume=True,
            experiment_name="test",
            run_name="run1",
            load_run="run1",
            checkpoint=1,
        )
        env_cfg, cfg_train = update_cfg_from_args(env_cfg, cfg_train, args)
        self.assertEqual(env_cfg.env.num_envs, 10)
        self.assertEqual(cfg_train.seed, 42)
        self.assertEqual(cfg_train.runner.max_iterations, 200)

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("torch.jit.save")
    def test_export_policy_as_jit(self, mock_jit_save, mock_exists, mock_makedirs) -> None:
        class MockActorCritic:
            def __init__(self) -> None:
                self.actor = torch.nn.Linear(10, 2)

        actor_critic = MockActorCritic()
        path = "test_jit"
        mock_exists.return_value = True
        export_policy_as_jit(actor_critic, path)
        mock_jit_save.assert_called_once()
        mock_makedirs.assert_called_once_with(path, exist_ok=True)

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("torch.onnx.export")
    def test_export_policy_as_onnx(self, mock_onnx_export, mock_exists, mock_makedirs) -> None:
        class MockActorCritic:
            def __init__(self) -> None:
                self.actor = torch.nn.Linear(10, 2)

        actor_critic = MockActorCritic()
        path = "test_onnx"
        mock_exists.return_value = True
        export_policy_as_onnx(actor_critic, path)
        mock_onnx_export.assert_called_once()
        mock_makedirs.assert_called_once_with(path, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
