from types import SimpleNamespace

import numpy as np
import torch

import da3_streaming.eval_spedtest_da3 as eval_spedtest_da3


def test_split_reference_and_query_uses_dataset_sizes():
    descriptors = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    refs, queries = eval_spedtest_da3.split_reference_and_query(descriptors, num_references=6)

    assert refs.shape == (6, 2)
    assert queries.shape == (4, 2)
    assert torch.equal(refs, descriptors[:6])
    assert torch.equal(queries, descriptors[6:])


def test_build_sped_image_paths_preserves_dataset_order(tmp_path):
    image_names = ["ref/a.jpg", "query/z.jpg"]

    paths = eval_spedtest_da3.build_sped_image_paths(tmp_path, image_names)

    assert paths == [str(tmp_path / "ref/a.jpg"), str(tmp_path / "query/z.jpg")]


class _StubModel:
    def __init__(self):
        self.eval_called = False
        self.moved_to = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.moved_to = torch.device(device)
        return self


def test_move_model_to_available_device_uses_available_runtime(monkeypatch):
    stub = _StubModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model = eval_spedtest_da3.move_model_to_available_device(stub)

    assert model is stub
    assert stub.eval_called is True
    assert stub.moved_to == torch.device("cpu")


def test_real_salad_validation_recalls_on_tiny_synthetic_data():
    refs = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.1], [9.9, 9.9]], dtype=np.float32)
    ground_truth = [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)]

    recalls = eval_spedtest_da3.get_validation_recalls(
        r_list=refs,
        q_list=queries,
        k_values=[1, 2],
        gt=ground_truth,
        print_results=False,
        faiss_gpu=False,
        dataset_name="SPED",
    )

    assert recalls == {1: 1.0, 2: 1.0}


def test_evaluate_spedtest_uses_salad_validator_with_expected_arguments(monkeypatch, tmp_path):
    descriptors = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.1]],
        dtype=torch.float32,
    )
    dataset_root = tmp_path / "sped"
    captured = {}
    detector_state = {}

    class _FakeDataset:
        def __init__(self, input_transform=None):
            self.input_transform = input_transform
            self.images = ["ref/a.jpg", "ref/b.jpg", "query/c.jpg"]
            self.num_references = 2
            self.num_queries = 1
            self.ground_truth = [np.array([0], dtype=np.int64)]

    class _FakeDetector:
        def __init__(self, image_dir, config, da3_model):
            self.image_dir = image_dir
            self.config = config
            self.da3_model = da3_model
            self.image_paths = []
            self.extract_time_s = 1.5
            self.extract_images_per_sec = 2.0
            self.extract_ms_per_image = 3.0
            detector_state["instance"] = self

        def extract_descriptors(self):
            return descriptors

    def fake_load_sped_components():
        return _FakeDataset, dataset_root

    def fake_load_da3_model(model_name_or_path):
        assert model_name_or_path == "mock-model"
        return object()

    def fake_validation_recalls(**kwargs):
        captured.update(kwargs)
        return {1: 1.0, 5: 0.5}

    monkeypatch.setattr(eval_spedtest_da3, "load_sped_components", fake_load_sped_components)
    monkeypatch.setattr(eval_spedtest_da3, "load_da3_model", fake_load_da3_model)
    monkeypatch.setattr(eval_spedtest_da3, "DA3LoopDetector", _FakeDetector)
    monkeypatch.setattr(eval_spedtest_da3, "get_validation_recalls", fake_validation_recalls)

    args = SimpleNamespace(
        model_name_or_path="mock-model",
        batch_size=4,
        process_res=512,
        process_res_method="upper_bound_resize",
        ref_view_strategy="saddle_balanced",
        pooling="mean",
        gem_p=3.0,
        k_values=[1, 5],
    )

    result = eval_spedtest_da3.evaluate_spedtest(args)

    assert result["recalls"] == {1: 1.0, 5: 0.5}
    assert result["descriptor_dim"] == 2
    assert detector_state["instance"].image_dir == str(dataset_root)
    assert detector_state["instance"].image_paths == [
        dataset_root / "ref/a.jpg",
        dataset_root / "ref/b.jpg",
        dataset_root / "query/c.jpg",
    ]
    np.testing.assert_array_equal(captured["r_list"], descriptors[:2].numpy())
    np.testing.assert_array_equal(captured["q_list"], descriptors[2:].numpy())
    assert captured["k_values"] == [1, 5]
    assert len(captured["gt"]) == 1
    np.testing.assert_array_equal(captured["gt"][0], np.array([0], dtype=np.int64))
    assert captured["dataset_name"] == "SPED"
    assert captured["print_results"] is True
    assert set(captured) == {"r_list", "q_list", "k_values", "gt", "print_results", "dataset_name"}
