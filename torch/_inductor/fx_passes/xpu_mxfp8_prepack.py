from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.fx import GraphModule, Node
from torch._subclasses.fake_tensor import is_fake

from torch._inductor.freezing_utils import (
    enter_freezing,
    maybe_set_is_frozen_param,
    record_has_frozen_params,
)


aten = torch.ops.aten
prims = torch.ops.prims


_STATIC_VIEW_OPS = {
    aten._unsafe_view.default,
    aten.alias.default,
    aten.detach.default,
    aten.permute.default,
    aten.reshape.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    prims.view_of.default,
}


def _tensor_meta(node: Any) -> torch.Tensor | None:
    if not isinstance(node, Node):
        return None
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _static_shape(tensor: torch.Tensor) -> tuple[int, ...] | None:
    try:
        return tuple(int(dim) for dim in tensor.shape)
    except (TypeError, ValueError, RuntimeError):
        return None


def _placeholder_index(
    node: Node, placeholder_indices: dict[Node, int]
) -> int | None:
    current = node
    while current.op == "call_function" and current.target in _STATIC_VIEW_OPS:
        if not current.args or not isinstance(current.args[0], Node):
            return None
        current = current.args[0]
    return placeholder_indices.get(current)


def _transpose_source(node: Node) -> Node | None:
    if node.op != "call_function" or not node.args:
        return None
    if node.target == aten.t.default:
        source = node.args[0]
    elif node.target == aten.transpose.int:
        if len(node.args) < 3:
            return None
        try:
            dimensions = {int(node.args[1]), int(node.args[2])}
        except (TypeError, ValueError):
            return None
        if dimensions not in ({0, 1}, {-2, -1}):
            return None
        source = node.args[0]
    elif node.target == aten.permute.default:
        if len(node.args) < 2 or tuple(node.args[1]) != (1, 0):
            return None
        source = node.args[0]
    else:
        return None
    return source if isinstance(source, Node) else None


def _match_scale_pack(node: Node) -> tuple[Node, Node] | None:
    if node.op != "call_function" or node.target != aten.clone.default:
        return None
    if node.kwargs.get("memory_format") != torch.contiguous_format:
        return None
    if not node.args or not isinstance(node.args[0], Node):
        return None
    transpose = node.args[0]
    source = _transpose_source(transpose)
    return (transpose, source) if source is not None else None


def _valid_mxfp8_scaled_mm(node: Node, scale_source: Node) -> bool:
    if len(node.args) < 4:
        return False
    mat_a = _tensor_meta(node.args[0])
    mat_b = _tensor_meta(node.args[1])
    scale_a = _tensor_meta(node.args[2])
    scale_b = _tensor_meta(node.args[3])
    source = _tensor_meta(scale_source)
    if any(value is None for value in (mat_a, mat_b, scale_a, scale_b, source)):
        return False
    assert mat_a is not None
    assert mat_b is not None
    assert scale_a is not None
    assert scale_b is not None
    assert source is not None
    if (
        mat_a.device.type != "xpu"
        or mat_b.device.type != "xpu"
        or scale_b.device.type != "xpu"
        or mat_a.dtype != torch.float8_e4m3fn
        or mat_b.dtype != torch.float8_e4m3fn
        or scale_a.dtype != torch.float8_e8m0fnu
        or scale_b.dtype != torch.float8_e8m0fnu
        or source.dtype != torch.float8_e8m0fnu
    ):
        return False

    a_shape = _static_shape(mat_a)
    b_shape = _static_shape(mat_b)
    packed_shape = _static_shape(scale_b)
    source_shape = _static_shape(source)
    if (
        a_shape is None
        or b_shape is None
        or packed_shape is None
        or source_shape is None
        or len(a_shape) != 2
        or len(b_shape) != 2
        or len(packed_shape) != 2
        or len(source_shape) != 2
    ):
        return False
    k, n = b_shape
    return (
        a_shape[1] == k
        and k % 32 == 0
        and source_shape == (n, k // 32)
        and packed_shape == (k // 32, n)
    )


def _prepack_value(
    source: torch.Tensor, transpose: Node, clone: Node
) -> torch.Tensor:
    with torch.utils._python_dispatch._disable_current_modes():
        transposed = transpose.target(
            source, *transpose.args[1:], **transpose.kwargs
        )
        return clone.target(transposed, *clone.args[1:], **clone.kwargs)


def prepack_xpu_mxfp8_weight_scales(
    gm: GraphModule,
    real_inputs: Sequence[object],
    static_input_indices: Sequence[int],
) -> int:
    graph = gm.graph
    placeholders = graph.find_nodes(op="placeholder")
    placeholder_indices = {node: index for index, node in enumerate(placeholders)}
    static_inputs = set(static_input_indices)
    packed_nodes: dict[Node, Node] = {}
    synchronized_devices: set[torch.device] = set()
    packed_count = 0

    for scaled_mm in list(
        graph.find_nodes(op="call_function", target=aten._scaled_mm.default)
    ):
        if len(scaled_mm.args) < 4 or not isinstance(scaled_mm.args[3], Node):
            continue
        clone = scaled_mm.args[3]
        match = _match_scale_pack(clone)
        if match is None:
            continue
        transpose, scale_source = match
        if not _valid_mxfp8_scaled_mm(scaled_mm, scale_source):
            continue

        scale_index = placeholder_indices.get(scale_source)
        mat_b = scaled_mm.args[1]
        if not isinstance(mat_b, Node):
            continue
        weight_index = _placeholder_index(mat_b, placeholder_indices)
        if (
            scale_index is None
            or weight_index is None
            or scale_index not in static_inputs
            or weight_index not in static_inputs
            or scale_index >= len(real_inputs)
            or weight_index >= len(real_inputs)
        ):
            continue
        source = real_inputs[scale_index]
        weight = real_inputs[weight_index]
        source_meta = _tensor_meta(scale_source)
        weight_meta = _tensor_meta(placeholders[weight_index])
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(weight, torch.Tensor)
            or is_fake(source)
            or is_fake(weight)
            or source.device.type != "xpu"
            or weight.device.type != "xpu"
            or source.dtype != torch.float8_e8m0fnu
            or weight.dtype != torch.float8_e4m3fn
            or source_meta is None
            or weight_meta is None
            or _static_shape(source) != _static_shape(source_meta)
            or _static_shape(weight) != _static_shape(weight_meta)
        ):
            continue

        packed_node = packed_nodes.get(scale_source)
        if packed_node is None:
            packed = _prepack_value(source, transpose, clone)
            name_index = packed_count
            name = f"_mxfp8_packed_scale_{name_index}"
            while hasattr(gm, name):
                name_index += 1
                name = f"_mxfp8_packed_scale_{name_index}"
            gm.register_buffer(name, packed)
            with enter_freezing():
                maybe_set_is_frozen_param(packed)
            with graph.inserting_before(clone):
                packed_node = graph.get_attr(name)
            packed_node.meta.update(clone.meta)
            packed_nodes[scale_source] = packed_node
            synchronized_devices.add(packed.device)
            packed_count += 1

        clone.replace_all_uses_with(packed_node)
        graph.erase_node(clone)
        if not transpose.users:
            graph.erase_node(transpose)

    if not packed_count:
        return 0

    record_has_frozen_params(gm)
    graph.lint()
    gm.recompile()
    for device in synchronized_devices:
        torch.xpu.synchronize(device)
    return packed_count
