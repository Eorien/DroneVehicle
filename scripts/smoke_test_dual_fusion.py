#!/usr/bin/env python3
"""Verify RGB-IR dual-stream isolation, adaptive fusion, and pretrained mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dronevehicle import initialize_dual_stream
from ultralytics.nn.modules import AdaptiveFeatureFusion, RGBIRSplit
from ultralytics.nn.tasks import OBBModel, load_checkpoint

DEFAULT_MODEL = Path("configs/models/yolo11n-obb-rgbir-dual.yaml")
DEFAULT_WEIGHTS = Path("weights/yolo11n-obb.pt")


def parse_args() -> argparse.Namespace:
    """Parse model, checkpoint, and device options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def check_modules(device: str) -> None:
    """Check exact channel splitting, equal initial fusion, and trainable gate gradients."""
    split = RGBIRSplit().to(device)
    image = torch.randn(2, 4, 16, 16, device=device)
    rgb, infrared = split(image)
    assert torch.equal(rgb, image[:, :3])
    assert torch.equal(infrared, image[:, 3:4])

    torch.manual_seed(0)
    fusion = AdaptiveFeatureFusion(8).to(device)
    rgb_feature = torch.randn(2, 8, 8, 8, device=device, requires_grad=True)
    ir_feature = torch.randn(2, 8, 8, 8, device=device, requires_grad=True)
    output = fusion([rgb_feature, ir_feature])
    expected = (rgb_feature + ir_feature) * 0.5
    assert torch.equal(output, expected)
    output.square().mean().backward()
    assert (
        fusion.gate.weight.grad is not None
        and torch.isfinite(fusion.gate.weight.grad).all()
    )
    assert torch.count_nonzero(fusion.gate.weight.grad) == fusion.gate.weight.numel()
    assert (
        fusion.gate.bias.grad is not None
        and torch.isfinite(fusion.gate.bias.grad).all()
    )
    assert torch.count_nonzero(fusion.gate.bias.grad) == fusion.gate.bias.numel()
    optimizer = torch.optim.Adam(fusion.parameters(), lr=1e-3)
    optimizer.step()
    assert torch.count_nonzero(fusion.gate.weight) == fusion.gate.weight.numel()
    assert torch.count_nonzero(fusion.gate.bias) == fusion.gate.bias.numel()
    print(f"RGBIRSplit and equal-weight adaptive fusion on {device}: OK")


def check_pretrained_mapping(model_path: Path, weights_path: Path) -> OBBModel:
    """Check exact RGB/IR backbone transfer and explicit expected omissions."""
    source, _ = load_checkpoint(weights_path, device="cpu")
    source_state = source.float().state_dict()
    model = OBBModel(str(model_path), ch=4, nc=5, verbose=False)
    transferred = initialize_dual_stream(model, source)
    state = model.state_dict()

    assert transferred == 775
    assert len(state) == 787
    assert torch.equal(
        state["model.2.conv.weight"], source_state["model.0.conv.weight"]
    )
    assert torch.equal(
        state["model.14.conv.weight"],
        source_state["model.0.conv.weight"].mean(dim=1, keepdim=True),
    )
    for fusion_index in (25, 26, 27):
        assert torch.count_nonzero(state[f"model.{fusion_index}.gate.weight"]) == 0
        assert torch.count_nonzero(state[f"model.{fusion_index}.gate.bias"]) == 0
    print("pretrained mapping: 775/787 target states loaded")
    print("expected new states: 6 fusion tensors + 6 five-class OBB outputs")
    return model


def capture_features(
    model: OBBModel, image: torch.Tensor
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run one forward pass and capture modality-specific and fused pyramid tensors."""
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for index in (6, 8, 12, 18, 20, 24, 25, 26, 27):

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            i: int = index,
        ) -> None:
            captured[i] = output.detach().cpu()

        hooks.append(model.model[index].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            output = model(image)
    finally:
        for hook in hooks:
            hook.remove()
    return captured, output[0].detach().cpu()


def check_stream_isolation(model: OBBModel, device: str) -> None:
    """Ensure changing one modality cannot alter the other branch before fusion."""
    model = model.eval().to(device)
    torch.manual_seed(1)
    base = torch.randn(1, 4, 64, 64, device=device)
    changed_ir = base.clone()
    changed_ir[:, 3:4].add_(1.0)
    changed_rgb = base.clone()
    changed_rgb[:, :3].add_(1.0)

    base_features, base_output = capture_features(model, base)
    ir_features, _ = capture_features(model, changed_ir)
    rgb_features, _ = capture_features(model, changed_rgb)

    for rgb_index in (6, 8, 12):
        assert torch.equal(base_features[rgb_index], ir_features[rgb_index])
        assert not torch.equal(base_features[rgb_index], rgb_features[rgb_index])
    for ir_index in (18, 20, 24):
        assert torch.equal(base_features[ir_index], rgb_features[ir_index])
        assert not torch.equal(base_features[ir_index], ir_features[ir_index])
    for rgb_index, ir_index, fused_index in ((6, 18, 25), (8, 20, 26), (12, 24, 27)):
        expected = (base_features[rgb_index] + base_features[ir_index]) * 0.5
        assert torch.equal(base_features[fused_index], expected)
    assert base_output.shape == (1, 10, 84) and torch.isfinite(base_output).all()
    print("RGB and IR branch isolation: OK")
    print("P3/P4/P5 initial equal-weight fusion and OBB output: OK")


def main() -> None:
    """Run all focused dual-stream checks."""
    args = parse_args()
    for path in (args.model, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    check_modules(args.device)
    model = check_pretrained_mapping(args.model, args.weights)
    check_stream_isolation(model, args.device)
    model.info(verbose=True)
    print("dual-stream smoke test: OK")


if __name__ == "__main__":
    main()
