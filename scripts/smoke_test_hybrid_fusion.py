#!/usr/bin/env python3
"""Verify Hybrid Fusion zero perturbation, control alignment, and gradients."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from dronevehicle import initialize_dual_stream, initialize_hybrid_stream
from ultralytics.nn.modules import ShallowCrossModalInteraction
from ultralytics.nn.tasks import OBBModel, load_checkpoint

DEFAULT_DUAL_MODEL = Path("configs/models/yolo11n-obb-rgbir-dual.yaml")
DEFAULT_HYBRID_MODEL = Path("configs/models/yolo11n-obb-rgbir-hybrid.yaml")
DEFAULT_WEIGHTS = Path("weights/yolo11n-obb.pt")


def parse_args() -> argparse.Namespace:
    """Parse model, checkpoint, and device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-model", type=Path, default=DEFAULT_DUAL_MODEL)
    parser.add_argument("--hybrid-model", type=Path, default=DEFAULT_HYBRID_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=64)
    return parser.parse_args()


def check_interaction_module(device: str) -> None:
    """Check identity initialization and two-step gradient flow through shared mixing."""
    torch.manual_seed(0)
    module = ShallowCrossModalInteraction(8).to(device)
    rgb = torch.randn(2, 8, 16, 16, device=device)
    infrared = torch.randn(2, 8, 16, 16, device=device)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    enhanced_rgb, enhanced_ir = module([rgb, infrared])
    assert torch.equal(enhanced_rgb, rgb)
    assert torch.equal(enhanced_ir, infrared)
    (enhanced_rgb.square().mean() + enhanced_ir.square().mean()).backward()
    for projection in (module.project_rgb, module.project_ir):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) == projection.weight.numel()
        assert torch.isfinite(projection.weight.grad).all()
    optimizer.step()
    assert (
        torch.count_nonzero(module.project_rgb.weight)
        == module.project_rgb.weight.numel()
    )
    assert (
        torch.count_nonzero(module.project_ir.weight)
        == module.project_ir.weight.numel()
    )

    optimizer.zero_grad(set_to_none=True)
    enhanced_rgb, enhanced_ir = module([rgb, infrared])
    (enhanced_rgb.square().mean() + enhanced_ir.square().mean()).backward()
    assert module.shared.weight.grad is not None
    assert (
        torch.count_nonzero(module.shared.weight.grad) == module.shared.weight.numel()
    )
    assert torch.isfinite(module.shared.weight.grad).all()
    print(f"zero-perturbation interaction and staged gradients on {device}: OK")


def dual_to_hybrid_index(index: int) -> int | None:
    """Map state-bearing pure-dual layers to their Hybrid counterparts."""
    if 2 <= index <= 4:
        return index
    if 5 <= index <= 12:
        return index + 6
    if 14 <= index <= 16:
        return index - 8
    if 17 <= index <= 40:
        return index + 3
    return None


def compare_control_states(dual: OBBModel, hybrid: OBBModel) -> None:
    """Require every pure-dual state tensor to have an identical Hybrid counterpart."""
    dual_state = dual.state_dict()
    hybrid_state = hybrid.state_dict()
    compared = set()
    missing = []
    mismatched = []
    for dual_key, dual_tensor in dual_state.items():
        match = re.match(r"^model\.(\d+)\.(.+)$", dual_key)
        if match is None:
            continue
        hybrid_index = dual_to_hybrid_index(int(match.group(1)))
        if hybrid_index is None:
            continue
        hybrid_key = f"model.{hybrid_index}.{match.group(2)}"
        if hybrid_key not in hybrid_state:
            missing.append((dual_key, hybrid_key))
            continue
        compared.add(hybrid_key)
        if dual_tensor.shape != hybrid_state[hybrid_key].shape or not torch.equal(
            dual_tensor, hybrid_state[hybrid_key]
        ):
            mismatched.append((dual_key, hybrid_key))
    extra = sorted(set(hybrid_state) - compared)
    assert not missing, missing
    assert not mismatched, mismatched
    assert extra and all(key.startswith("model.9.") for key in extra), extra
    assert len(extra) == 6
    print(f"control alignment: {len(compared)} pure-dual state tensors are identical")
    print("Hybrid-only state tensors: 6 under model.9")


def build_models(args: argparse.Namespace) -> tuple[OBBModel, OBBModel]:
    """Build deterministic dual/Hybrid models and apply explicit pretrained mappings."""
    source, _ = load_checkpoint(args.weights, device="cpu")
    torch.manual_seed(0)
    dual = OBBModel(str(args.dual_model), ch=4, nc=5, verbose=False)
    dual_count = initialize_dual_stream(dual, source)
    torch.manual_seed(0)
    hybrid = OBBModel(str(args.hybrid_model), ch=4, nc=5, verbose=False)
    hybrid_count = initialize_hybrid_stream(hybrid, source)
    assert dual_count == hybrid_count == 775
    assert len(dual.state_dict()) == 787
    assert len(hybrid.state_dict()) == 793
    compare_control_states(dual, hybrid)
    return dual, hybrid


def check_model_equivalence(
    dual: OBBModel, hybrid: OBBModel, device: str, imgsz: int
) -> None:
    """Require zero-perturbation Hybrid features and predictions to equal pure dual."""
    dual = dual.eval().to(device)
    hybrid = hybrid.eval().to(device)
    captured: dict[str, object] = {}

    def capture_interaction(
        _module: torch.nn.Module,
        inputs: tuple[list[torch.Tensor], ...],
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        rgb, infrared = inputs[0]
        captured["input_shapes"] = (tuple(rgb.shape), tuple(infrared.shape))
        captured["identity"] = torch.equal(outputs[0], rgb) and torch.equal(
            outputs[1], infrared
        )

    hook = hybrid.model[9].register_forward_hook(capture_interaction)
    torch.manual_seed(1)
    image = torch.randn(1, 4, imgsz, imgsz, device=device)
    try:
        with torch.inference_mode():
            dual_output = dual(image)[0]
            hybrid_output = hybrid(image)[0]
    finally:
        hook.remove()
    expected_shape = (1, 64, imgsz // 4, imgsz // 4)
    assert captured["input_shapes"] == (expected_shape, expected_shape)
    assert captured["identity"] is True
    assert torch.equal(hybrid_output, dual_output)
    assert hybrid_output.shape[0] == 1 and hybrid_output.shape[1] == 10
    print(f"Hybrid insertion feature: {expected_shape}, stride=4, P2/4 before P3")
    print(f"zero-perturbation prediction equivalence: {tuple(hybrid_output.shape)}")


def main() -> None:
    """Run all focused Hybrid Fusion checks."""
    args = parse_args()
    for path in (args.dual_model, args.hybrid_model, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    check_interaction_module(args.device)
    dual, hybrid = build_models(args)
    check_model_equivalence(dual, hybrid, args.device, args.imgsz)
    hybrid.info(verbose=True, imgsz=640)
    print("Hybrid Fusion smoke test: OK")


if __name__ == "__main__":
    main()
