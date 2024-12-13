"""PyTorch model export utilities."""

import inspect
import json
import logging
import sys
from dataclasses import fields, is_dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Extract model information including input parameters and their types.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary containing model information
    """
    # Get model's forward method signature
    signature = inspect.signature(model.forward)

    # Extract parameter information
    params_info = {}
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        params_info[name] = {
            "default": None if param.default is param.empty else str(param.default),
        }

    return {
        "input_params": params_info,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def add_metadata_to_onnx(
    model_proto: onnx.ModelProto, 
    metadata: Dict[str, Any], 
    config: Optional[object] = None
) -> onnx.ModelProto:
    """Add metadata to ONNX model.

    Args:
        model_proto: ONNX model prototype
        metadata: Dictionary of metadata to add
        config: Optional configuration dataclass to add to metadata

    Returns:
        ONNX model with added metadata
    """
    # Build metadata dictionary
    metadata_dict = metadata.copy()

    # Add configuration if provided
    if config is not None:
        if is_dataclass(config):
            for field in fields(config):
                metadata_dict[field.name] = getattr(config, field.name)
        elif not isinstance(config, dict):
            raise ValueError("config must be a dataclass or dict. Got: " + str(type(config)))

    # Add metadata as JSON string
    meta = model_proto.metadata_props.add()
    meta.key = "kinfer_metadata"
    meta.value = json.dumps(metadata_dict)

    return model_proto


def infer_input_shapes(model: nn.Module) -> Union[torch.Size, List[torch.Size]]:
    """Infer input shapes from model architecture.

    Args:
        model: PyTorch model to analyze

    Returns:
        Single input shape or list of input shapes
    """
    # Check if model is Sequential or has Sequential as first child
    if isinstance(model, nn.Sequential):
        first_layer = model[0]
    else:
        # Get the first immediate child
        children = list(model.children())
        first_layer = children[0] if children else None

        # Unwrap if the first child is Sequential
        if isinstance(first_layer, nn.Sequential):
            first_layer = first_layer[0]
    # Check if first layer is a type we can infer from
    if not isinstance(first_layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise ValueError("First layer must be Linear or Conv layer to infer input shape")

    # Get input dimensions
    if isinstance(first_layer, nn.Linear):
        return torch.Size([1, first_layer.in_features])
    elif isinstance(first_layer, nn.Conv1d):
        raise ValueError("Cannot infer sequence length for Conv1d layer. Please provide input_tensors explicitly.")
    elif isinstance(first_layer, nn.Conv2d):
        raise ValueError("Cannot infer image dimensions for Conv2d layer. Please provide input_tensors explicitly.")
    elif isinstance(first_layer, nn.Conv3d):
        raise ValueError("Cannot infer volume dimensions for Conv3d layer. Please provide input_tensors explicitly.")

    raise ValueError("Could not infer input shape from model architecture")


def create_example_inputs(model: nn.Module) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Create example input tensors based on model's forward signature and architecture.

    Args:
        model: PyTorch model to analyze

    Returns:
        Single tensor or tuple of tensors matching the model's expected input
    """
    signature = inspect.signature(model.forward)
    params = [p for p in signature.parameters.items() if p[0] != "self"]

    # If single parameter (besides self), try to infer shape
    if len(params) == 1:
        shape = infer_input_shapes(model)
        return torch.randn(*shape) if isinstance(shape, torch.Size) else torch.randn(*shape[0])

    # For multiple parameters, try to infer from parameter annotations
    input_tensors = []
    for name, param in params:
        # Try to get shape from annotation
        if hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is torch.Tensor:
            # If annotation includes size information (e.g., Tensor[batch_size, channels, height, width])
            if hasattr(param.annotation, "__args__"):
                shape = param.annotation.__args__
                input_tensors.append(torch.randn(*shape) if isinstance(shape, torch.Size) else torch.randn(*shape[0]))
            else:
                # Default to a vector if no size info
                input_tensors.append(torch.randn(1, 32))
        else:
            # Default fallback
            input_tensors.append(torch.randn(1, 32))

    return tuple(input_tensors)


def export_to_onnx(
    model: nn.Module,
    input_tensors: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    config: Optional[object] = None,
    save_path: Optional[str] = None,
) -> ort.InferenceSession:
    """Export PyTorch model to ONNX format with metadata.

    Args:
        model: PyTorch model to export
        input_tensors: Optional example input tensors for model tracing. If None, will attempt to infer.
        config: Optional configuration dataclass to add to metadata
        save_path: Optional path to save the ONNX model

    Returns:
        ONNX inference session
    """
    # Get model information
    model_info = get_model_info(model)

    # Create example inputs if not provided
    if input_tensors is None:
        logging.warning(
            "No input_tensors provided. Attempting to automatically infer input shapes. "
            "Note: Input shape inference is *highly* experimental and may not work correctly for all models."
        )
        try:
            input_tensors = create_example_inputs(model)
            model_info["inferred_input_shapes"] = str(
                input_tensors.shape if isinstance(input_tensors, torch.Tensor) else [t.shape for t in input_tensors]
            )

        except ValueError as e:
            raise ValueError(
                f"Could not automatically infer input shapes. Please provide input_tensors. Error: {str(e)}"
            )

    # Convert model to JIT if not already
    if not isinstance(model, torch.jit.ScriptModule):
        model = torch.jit.script(model)

    # Export model to buffer
    buffer = BytesIO()
    torch.onnx.export(
        model,
        (input_tensors,) if isinstance(input_tensors, torch.Tensor) else input_tensors,
        buffer,
    )
    buffer.seek(0)

    # Load as ONNX model
    model_proto = onnx.load_model(buffer)

    # Add config dict to model info if provided
    if isinstance(config, dict):
        model_info.update(config)

    # Add metadata
    model_proto = add_metadata_to_onnx(model_proto, model_info, config)

    # Save if path provided
    if save_path:
        onnx.save_model(model_proto, save_path)

    # Convert to inference session
    buffer = BytesIO()
    onnx.save_model(model_proto, buffer)
    buffer.seek(0)

    return ort.InferenceSession(buffer.read()), model_proto
