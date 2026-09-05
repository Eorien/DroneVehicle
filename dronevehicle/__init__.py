"""DroneVehicle RGB-IR OBB project extensions."""

from .rgbir_dataset import RGBIRDataset
from .rgbir_trainer import RGBIROBBTrainer, initialize_ir_channel

__all__ = ("RGBIRDataset", "RGBIROBBTrainer", "initialize_ir_channel")
