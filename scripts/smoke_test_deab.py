#!/usr/bin/env python3
"""Verify DEAB kernels, gradients, model shape, and SPD control-variable alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dronevehicle import initialize_ir_channel
from ultralytics.nn.modules import DEAB, DetailEnhancedConv, SPDConv, SPDConvDEAB
from ultralytics.nn.tasks import OBBModel, load_checkpoint

DEFAULT_SPD_MODEL = Path("configs/models/yolo11n-obb-rgbir-spd.yaml")
DEFAULT_DEAB_MODEL = Path("configs/models/yolo11n-obb-rgbir-spd-deab.yaml")
DEFAULT_WEIGHTS = Path("weights/yolo11n-obb.pt")


def parse_args() -> argparse.Namespace:
    """Parse smoke-test paths and device selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spd-model", type=Path, default=DEFAULT_SPD_MODEL)
    parser.add_argument("--deab-model", type=Path, default=DEFAULT_DEAB_MODEL)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def check_difference_kernels() -> None:
    """Check zero-sum difference kernels and equivalent-convolution output."""
    torch.manual_seed(0)
    module = DetailEnhancedConv(4)
    central = module._central_kernel(module.central.weight)
    angular = module._angular_kernel(module.angular.weight)
    horizontal = module._horizontal_kernel(module.horizontal.weight)
    vertical = module._vertical_kernel(module.vertical.weight)
    for name, kernel in (
        ("central", central),
        ("angular", angular),
        ("horizontal", horizontal),
        ("vertical", vertical),
    ):
        sums = kernel.sum(dim=(-2, -1))
        assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-6), (
            f"{name} kernel is not zero-sum"
        )

    x = torch.randn(2, 4, 16, 16)
    kernel, bias = module.get_equivalent_kernel_bias()
    expected = F.conv2d(x, kernel, bias, padding=1)
    actual = module(x)
    assert torch.equal(actual, expected)
    print("DEConv directional kernels and equivalent convolution: OK")


def check_deab_backward(device: str) -> None:
    """Check shape preservation and finite gradients through a standalone DEAB."""
    torch.manual_seed(0)
    block = DEAB(16).to(device)
    x = torch.randn(2, 16, 20, 20, device=device, requires_grad=True)
    output = block(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in block.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in block.parameters())
    print(f"standalone DEAB forward/backward on {device}: OK")


def build_model(path: Path, seed: int) -> OBBModel:
    """Build a deterministic four-channel, five-class OBB model."""
    torch.manual_seed(seed)
    return OBBModel(str(path), ch=4, nc=5, verbose=False)


def check_model_alignment(args: argparse.Namespace) -> tuple[OBBModel, OBBModel]:
    """Ensure SPD+DEAB only adds DEAB state and preserves every SPD tensor."""
    source, _ = load_checkpoint(args.weights, device="cpu")
    spd = build_model(args.spd_model, seed=0)
    deab = build_model(args.deab_model, seed=0)
    spd.load(source, verbose=False)
    deab.load(source, verbose=False)
    assert initialize_ir_channel(spd, source)
    assert initialize_ir_channel(deab, source)

    assert len(spd.model) == len(deab.model)
    assert isinstance(spd.model[3], SPDConv)
    assert not isinstance(spd.model[3], SPDConvDEAB)
    assert isinstance(deab.model[3], SPDConvDEAB)

    spd_state = spd.state_dict()
    deab_state = deab.state_dict()
    missing = [key for key in spd_state if key not in deab_state]
    mismatched = [
        key
        for key, value in spd_state.items()
        if key in deab_state
        and (
            value.shape != deab_state[key].shape
            or not torch.equal(value, deab_state[key])
        )
    ]
    extra = sorted(set(deab_state) - set(spd_state))
    assert not missing, f"SPD state missing from SPD+DEAB: {missing}"
    assert not mismatched, f"Shared SPD tensors changed: {mismatched}"
    assert extra and all(key.startswith("model.3.deab.") for key in extra), extra
    print(f"control-variable alignment: all {len(spd_state)} SPD tensors identical")
    print(f"DEAB-only state tensors: {len(extra)}")
    return spd, deab


def check_model_forward(spd: OBBModel, deab: OBBModel, device: str) -> None:
    """Check unchanged layer count, P3/8 shape, and OBB output contract."""
    captures: dict[str, tuple[int, ...]] = {}

    def capture_p3(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captures["p3"] = tuple(output.shape)

    hook = deab.model[3].register_forward_hook(capture_p3)
    try:
        images = torch.randn(1, 4, 64, 64, device=device)
        spd = spd.eval().to(device)
        deab = deab.eval().to(device)
        with torch.inference_mode():
            spd_output = spd(images)
            deab_output = deab(images)
    finally:
        hook.remove()
    assert captures["p3"] == (1, 64, 8, 8)
    assert isinstance(spd_output, tuple) and isinstance(deab_output, tuple)
    assert spd_output[0].shape == deab_output[0].shape
    assert torch.isfinite(deab_output[0]).all()
    print(f"SPD+DEAB P3/8 feature: {captures['p3']}")
    print(f"OBB output contract: {tuple(deab_output[0].shape)}")


def main() -> None:
    """Run all focused DEAB checks."""
    args = parse_args()
    for path in (args.spd_model, args.deab_model, args.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    check_difference_kernels()
    check_deab_backward(args.device)
    spd, deab = check_model_alignment(args)
    check_model_forward(spd, deab, args.device)
    print("DEAB smoke test: OK")


if __name__ == "__main__":
    main()
