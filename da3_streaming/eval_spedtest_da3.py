import argparse
import numpy as np
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if __package__ in {None, ""}:
    from loop_utils.da3_loop_detector import DA3LoopDetector
else:
    from da3_streaming.loop_utils.da3_loop_detector import DA3LoopDetector

SALAD_ROOT = CURRENT_DIR / "loop_utils" / "salad"
DEFAULT_K_VALUES = [1, 5, 10, 15, 20, 25]


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_sped_components():
    if str(SALAD_ROOT) not in sys.path:
        sys.path.insert(0, str(SALAD_ROOT))

    with _pushd(SALAD_ROOT):
        import importlib

        sped_module = importlib.import_module("dataloaders.val.SPEDDataset")
        validation_module = importlib.import_module("utils.validation")

    dataset_root = (SALAD_ROOT / sped_module.DATASET_ROOT).resolve()
    gt_root = (SALAD_ROOT / sped_module.GT_ROOT).resolve()
    sped_module.DATASET_ROOT = f"{dataset_root}{os.sep}"
    sped_module.GT_ROOT = f"{gt_root}{os.sep}"
    return sped_module.SPEDDataset, validation_module.get_validation_recalls, dataset_root


def build_sped_image_paths(dataset_root, image_names):
    dataset_root = Path(dataset_root)
    return [str(dataset_root / image_name) for image_name in image_names]



def split_reference_and_query(descriptors: torch.Tensor, num_references: int):
    return descriptors[:num_references], descriptors[num_references:]



def move_model_to_available_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device)


def _compute_exact_validation_recalls(r_list, q_list, k_values, gt, print_results, dataset_name):
    distances = np.sum((q_list[:, None, :] - r_list[None, :, :]) ** 2, axis=-1)
    predictions = np.argsort(distances, axis=1)[:, : max(k_values)]

    correct_at_k = np.zeros(len(k_values), dtype=np.float32)
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    recalls = {k: float(v) for k, v in zip(k_values, correct_at_k)}

    if print_results:
        print()
        print(f"Performances on {dataset_name}")
        print("Recall@K", " ".join(f"{k}:{100 * recalls[k]:.2f}" for k in k_values))

    return recalls


def compute_validation_recalls(validator, r_list, q_list, k_values, gt, dataset_name, print_results):
    try:
        return validator(
            r_list=r_list,
            q_list=q_list,
            k_values=k_values,
            gt=gt,
            print_results=print_results,
            dataset_name=dataset_name,
            faiss_gpu=False,
        )
    except ModuleNotFoundError as exc:
        missing_module = exc.name or str(exc)
        if "faiss" not in missing_module:
            raise
        return _compute_exact_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=k_values,
            gt=gt,
            print_results=print_results,
            dataset_name=dataset_name,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DA3 loop descriptors on SPEDTEST")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="depth-anything/DA3-LARGE-1.1",
        help="Hugging Face model id or local path for Depth Anything 3",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for descriptor extraction")
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Input processing resolution passed to the DA3 input processor",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="DA3 input-processor resize mode",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="Reference-view strategy for DA3 backbone inference",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "gem"],
        help="Descriptor pooling mode for patch tokens",
    )
    parser.add_argument(
        "--gem-p",
        type=float,
        default=3.0,
        help="Exponent used when --pooling gem",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help="Recall@K values to report",
    )
    return parser.parse_args()



def load_da3_model(model_name_or_path: str):
    from depth_anything_3.api import DepthAnything3

    print(f"Loading DA3 model from: {model_name_or_path}")
    model = DepthAnything3.from_pretrained(model_name_or_path)
    return move_model_to_available_device(model)



def build_detector_config(args):
    return {
        "Loop": {
            "DA3": {
                "batch_size": args.batch_size,
                "process_res": args.process_res,
                "process_res_method": args.process_res_method,
                "pooling": args.pooling,
                "gem_p": args.gem_p,
            }
        },
        "Model": {"ref_view_strategy_loop": args.ref_view_strategy},
    }



def evaluate_spedtest(args):
    SPEDDataset, get_validation_recalls, dataset_root = load_sped_components()
    dataset = SPEDDataset(input_transform=None)

    da3_model = load_da3_model(args.model_name_or_path)
    detector = DA3LoopDetector(
        image_dir=str(dataset_root),
        config=build_detector_config(args),
        da3_model=da3_model,
    )
    detector.image_paths = [Path(path) for path in build_sped_image_paths(dataset_root, dataset.images)]

    print("Evaluating on SPED")
    descriptors = detector.extract_descriptors()

    refs, queries = split_reference_and_query(descriptors, dataset.num_references)

    print(f"Descriptor dimension {descriptors.shape[1]}")
    print("total_size", descriptors.shape[0], dataset.num_queries + dataset.num_references)
    print(f"total wall time {detector.extract_time_s:.3f}s")
    print(f"images_per_sec {detector.extract_images_per_sec:.2f}")
    print(f"ms_per_image {detector.extract_ms_per_image:.2f}")

    recalls = compute_validation_recalls(
        validator=get_validation_recalls,
        r_list=refs.detach().cpu().float().numpy(),
        q_list=queries.detach().cpu().float().numpy(),
        k_values=args.k_values,
        gt=dataset.ground_truth,
        dataset_name="SPED",
        print_results=True,
    )
    return {
        "recalls": recalls,
        "descriptor_dim": int(descriptors.shape[1]),
        "num_references": int(dataset.num_references),
        "num_queries": int(dataset.num_queries),
        "extract_time_s": float(detector.extract_time_s),
        "images_per_sec": float(detector.extract_images_per_sec),
        "ms_per_image": float(detector.extract_ms_per_image),
    }


if __name__ == "__main__":
    evaluate_spedtest(parse_args())
