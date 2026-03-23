import numpy as np
import torch

from da3_streaming.eval_spedtest_da3 import (
    build_sped_image_paths,
    compute_validation_recalls,
    move_model_to_available_device,
    split_reference_and_query,
)


def test_split_reference_and_query_uses_dataset_sizes():
    descriptors = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    refs, queries = split_reference_and_query(descriptors, num_references=6)

    assert refs.shape == (6, 2)
    assert queries.shape == (4, 2)
    assert torch.equal(refs, descriptors[:6])
    assert torch.equal(queries, descriptors[6:])


def test_build_sped_image_paths_preserves_dataset_order(tmp_path):
    image_names = ["ref/a.jpg", "query/z.jpg"]

    paths = build_sped_image_paths(tmp_path, image_names)

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

    model = move_model_to_available_device(stub)

    assert model is stub
    assert stub.eval_called is True
    assert stub.moved_to == torch.device("cpu")


def test_compute_validation_recalls_falls_back_without_faiss():
    refs = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    queries = np.array([[0.1, 0.1], [9.9, 9.9]], dtype=np.float32)
    ground_truth = [np.array([0]), np.array([1])]

    def _missing_faiss(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'faiss'")

    recalls = compute_validation_recalls(
        validator=_missing_faiss,
        r_list=refs,
        q_list=queries,
        k_values=[1, 2],
        gt=ground_truth,
        dataset_name="SPED",
        print_results=False,
    )

    assert recalls == {1: 1.0, 2: 1.0}
