import time
from pathlib import Path

try:
    import faiss
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    faiss = None

import torch

from depth_anything_3.model.loop_descriptor import build_loop_descriptor


class DA3LoopDetector:
    """Loop detector that mirrors the SALAD interface but uses DA3 descriptors."""

    def __init__(self, image_dir, output="loop_closures.txt", config=None, da3_model=None):
        self.image_dir = image_dir
        self.output = output
        self.config = config or {}
        self.da3_model = da3_model

        da3_cfg = self.config.get("Loop", {}).get("DA3", {})
        model_cfg = self.config.get("Model", {})
        self.batch_size = da3_cfg.get("batch_size", 8)
        self.process_res = da3_cfg.get("process_res", 504)
        self.process_res_method = da3_cfg.get("process_res_method", "upper_bound_resize")
        self.ref_view_strategy = model_cfg.get(
            "ref_view_strategy_loop",
            model_cfg.get("ref_view_strategy", "saddle_balanced"),
        )
        self.similarity_threshold = da3_cfg.get("similarity_threshold", 0.85)
        self.top_k = da3_cfg.get("top_k", 5)
        self.use_nms = da3_cfg.get("use_nms", True)
        self.nms_threshold = da3_cfg.get("nms_threshold", 25)
        self.loop_temporal_exclusion = model_cfg.get("loop_temporal_exclusion", 10)
        self.pooling = da3_cfg.get("pooling", "mean")
        self.gem_p = da3_cfg.get("gem_p", 3.0)

        self.device = None
        self.image_paths = None
        self.descriptors = None
        self.loop_closures = None

        self.extract_time_s = 0.0
        self.extract_images_per_sec = 0.0
        self.extract_ms_per_image = 0.0
        self.total_time_s = 0.0
        self.images_per_sec = 0.0
        self.ms_per_image = 0.0

    def load_model(self):
        if self.da3_model is None:
            raise ValueError(
                "DA3LoopDetector.load_model() requires an injected da3_model for v0"
            )

        if hasattr(self.da3_model, "_get_model_device"):
            self.device = self.da3_model._get_model_device()
        elif getattr(self.da3_model, "device", None) is not None:
            self.device = self.da3_model.device
        else:
            self.device = next(self.da3_model.parameters()).device
        return self.da3_model, self.device

    def get_image_paths(self):
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))

        self.image_paths = sorted(image_paths)
        return self.image_paths

    def extract_descriptors(self):
        if self.da3_model is None or self.device is None:
            self.load_model()

        if self.image_paths is None:
            self.get_image_paths()

        if not self.image_paths:
            self.descriptors = torch.empty((0, 0), dtype=torch.float32)
            return self.descriptors

        if not hasattr(self.da3_model, "input_processor"):
            raise ValueError("Injected da3_model must expose an input_processor")

        start_time = time.perf_counter()
        descriptors = []
        for start_idx in range(0, len(self.image_paths), self.batch_size):
            batch_paths = [
                str(path) for path in self.image_paths[start_idx : start_idx + self.batch_size]
            ]
            batch_tensor, _, _ = self.da3_model.input_processor(
                batch_paths,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
            )
            batch_tensor = batch_tensor.to(self.device).unsqueeze(1)

            need_sync = self.device.type == "cuda"
            if need_sync:
                torch.cuda.synchronize(self.device)
            with torch.no_grad():
                feats, _ = self.da3_model.model.backbone(
                    batch_tensor,
                    ref_view_strategy=self.ref_view_strategy,
                )
            if need_sync:
                torch.cuda.synchronize(self.device)

            camera_tokens, all_tokens = feats[-1]
            patch_tokens = all_tokens[:, :, 1:, :]
            batch_descriptors = build_loop_descriptor(
                camera_tokens,
                patch_tokens,
                pooling=self.pooling,
                gem_p=self.gem_p,
            )
            if batch_descriptors.ndim == 3 and batch_descriptors.shape[1] == 1:
                batch_descriptors = batch_descriptors[:, 0, :]
            descriptors.append(batch_descriptors.detach().cpu())

        self.descriptors = torch.cat(descriptors, dim=0)
        self.extract_time_s = time.perf_counter() - start_time
        num_images = len(self.image_paths)
        self.extract_images_per_sec = num_images / max(self.extract_time_s, 1e-6)
        self.extract_ms_per_image = 1000.0 * self.extract_time_s / max(num_images, 1)
        return self.descriptors

    def _apply_nms_filter(self, loop_closures, nms_threshold):
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()

        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)

        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue

            filtered_loops.append((idx1, idx2, sim))

            suppress_range = set()
            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2)
            suppress_range.update(range(start1, end1))

            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))
            suppressed.update(suppress_range)

        return filtered_loops

    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]

    def _search_descriptors(self, descriptors, k):
        if faiss is not None:
            index = faiss.IndexFlatIP(descriptors.shape[1])
            index.add(descriptors)
            return index.search(descriptors, k)

        desc_tensor = torch.from_numpy(descriptors)
        similarities = desc_tensor @ desc_tensor.T
        topk = torch.topk(similarities, k=k, dim=1)
        return topk.values.numpy(), topk.indices.numpy()

    def find_loop_closures(self):
        if self.descriptors is None:
            self.extract_descriptors()

        if self.descriptors.numel() == 0:
            self.loop_closures = []
            return self.loop_closures

        descriptors = self.descriptors.detach().cpu().float().numpy()
        search_k = min(self.top_k + 1, len(descriptors))
        similarities, indices = self._search_descriptors(descriptors, search_k)

        loop_closures = []
        for i in range(len(descriptors)):
            for j in range(1, search_k):
                neighbor_idx = int(indices[i, j])
                similarity = float(similarities[i, j])

                if neighbor_idx < 0:
                    continue
                if similarity <= self.similarity_threshold:
                    continue
                if abs(i - neighbor_idx) <= self.loop_temporal_exclusion:
                    continue

                if i < neighbor_idx:
                    loop_closures.append((i, neighbor_idx, similarity))
                else:
                    loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)
        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)

        self.loop_closures = self._ensure_decending_order(loop_closures)
        return self.loop_closures

    def save_results(self):
        if self.loop_closures is None:
            self.find_loop_closures()
        if self.image_paths is None:
            self.get_image_paths()

        with open(self.output, "w", encoding="utf-8") as handle:
            handle.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                handle.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            handle.write("\n# Loop pairs:\n")
            for idx1, idx2, sim in self.loop_closures:
                handle.write(f"{idx1}, {idx2}, {sim:.4f}\n")
            handle.write("\n# Image path list:\n")
            for idx, path in enumerate(self.image_paths):
                handle.write(f"# {idx}: {path}\n")

    def get_loop_list(self):
        if self.loop_closures is None:
            self.find_loop_closures()
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        start_time = time.perf_counter()

        if self.da3_model is None or self.device is None:
            self.load_model()

        self.get_image_paths()
        if not self.image_paths:
            self.loop_closures = []
            self.total_time_s = time.perf_counter() - start_time
            self.images_per_sec = 0.0
            self.ms_per_image = 0.0
            return self.loop_closures

        self.extract_descriptors()
        self.find_loop_closures()
        self.save_results()

        self.total_time_s = time.perf_counter() - start_time
        num_images = len(self.image_paths)
        self.images_per_sec = num_images / max(self.total_time_s, 1e-6)
        self.ms_per_image = 1000.0 * self.total_time_s / max(num_images, 1)
        return self.loop_closures
