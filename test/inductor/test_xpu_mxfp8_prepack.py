# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch.fx import Graph, GraphModule
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from torch._inductor import config
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.fx_passes.xpu_mxfp8_prepack import (
    prepack_xpu_mxfp8_weight_scales,
)
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import FakeTensorMode


aten = torch.ops.aten


def fake_xpu_tensor(shape, dtype, stride=None):
    if stride is None:
        meta = torch.empty(shape, device="meta", dtype=dtype)
    else:
        meta = torch.empty_strided(shape, stride, device="meta", dtype=dtype)
    mode = FakeTensorMode()
    return mode.fake_tensor_converter.from_meta_and_device(
        mode, meta, torch.device("xpu")
    )


def make_mxfp8_graph(shared_scale=False):
    graph = Graph()
    weight = graph.placeholder("weight")
    scale = graph.placeholder("scale")
    activation = graph.placeholder("activation")
    activation_scale = graph.placeholder("activation_scale")

    weight_t = graph.call_function(aten.permute.default, (weight, [1, 0]))
    weight_t.meta["val"] = fake_xpu_tensor(
        (32, 64), torch.float8_e4m3fn, (1, 32)
    )

    outputs = []
    clone_nodes = []
    for _ in range(2 if shared_scale else 1):
        scale_t = graph.call_function(aten.permute.default, (scale, [1, 0]))
        scale_t.meta["val"] = fake_xpu_tensor(
            (1, 64), torch.float8_e8m0fnu, (1, 1)
        )
        packed = graph.call_function(
            aten.clone.default,
            (scale_t,),
            {"memory_format": torch.contiguous_format},
        )
        packed.meta["val"] = fake_xpu_tensor(
            (1, 64), torch.float8_e8m0fnu
        )
        scaled_mm = graph.call_function(
            aten._scaled_mm.default,
            (activation, weight_t, activation_scale, packed),
        )
        scaled_mm.meta["val"] = fake_xpu_tensor((1, 64), torch.bfloat16)
        outputs.append(scaled_mm)
        clone_nodes.append(packed)
    graph.output(tuple(outputs))

    weight.meta["val"] = fake_xpu_tensor((64, 32), torch.float8_e4m3fn)
    scale.meta["val"] = fake_xpu_tensor((64, 1), torch.float8_e8m0fnu)
    activation.meta["val"] = fake_xpu_tensor((1, 32), torch.float8_e4m3fn)
    activation_scale.meta["val"] = fake_xpu_tensor(
        (1, 1), torch.float8_e8m0fnu
    )
    return GraphModule(torch.nn.Module(), graph), clone_nodes


class Mxfp8Linear(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        weight = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
        scale = torch.full(
            (64, 2), 127, device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        self.weight = torch.nn.Parameter(
            weight.to(torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = torch.nn.Parameter(scale, requires_grad=False)

    def forward(self, activation, activation_scale):
        return torch._scaled_mm(
            activation,
            self.weight.t(),
            activation_scale,
            self.scale.t().contiguous(),
            out_dtype=torch.bfloat16,
        )


def make_mxfp8_inputs(device):
    activation = torch.randn(1, 64, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    activation_scale = torch.full(
        (1, 2), 127, device=device, dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    return activation, activation_scale


class TestXpuMxfp8PrepackPass(TestCase):
    def test_prepack_and_deduplicate(self):
        gm, clone_nodes = make_mxfp8_graph(shared_scale=True)
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]

        with (
            mock.patch("torch.xpu.synchronize") as synchronize,
            mock.patch(
                "torch._inductor.fx_passes.xpu_mxfp8_prepack.is_fake",
                return_value=False,
            ),
        ):
            count = prepack_xpu_mxfp8_weight_scales(gm, inputs, [0, 1])

        self.assertEqual(count, 1)
        self.assertTrue(has_frozen_params(gm))
        self.assertTrue(hasattr(gm, "_mxfp8_packed_scale_0"))
        self.assertTrue(is_frozen_param(gm._mxfp8_packed_scale_0))
        self.assertEqual(
            len(gm.graph.find_nodes(op="call_function", target=aten.clone.default)),
            0,
        )
        self.assertEqual(clone_nodes[0]._erased, True)
        self.assertEqual(clone_nodes[1]._erased, True)
        synchronize.assert_called_once()

    def test_dynamic_weight_is_not_prepacked(self):
        gm, _ = make_mxfp8_graph()
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]

        with mock.patch("torch.xpu.synchronize") as synchronize:
            count = prepack_xpu_mxfp8_weight_scales(gm, inputs, [0])

        self.assertEqual(count, 0)
        self.assertFalse(has_frozen_params(gm))
        self.assertEqual(
            len(gm.graph.find_nodes(op="call_function", target=aten.clone.default)),
            1,
        )
        synchronize.assert_not_called()

    def test_wrong_scale_dtype_is_not_prepacked(self):
        gm, _ = make_mxfp8_graph()
        scale = list(gm.graph.find_nodes(op="placeholder"))[1]
        scale.meta["val"] = fake_xpu_tensor((64, 1), torch.float32)
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]

        with mock.patch("torch.xpu.synchronize") as synchronize:
            count = prepack_xpu_mxfp8_weight_scales(gm, inputs, [0, 1])

        self.assertEqual(count, 0)
        self.assertFalse(has_frozen_params(gm))
        synchronize.assert_not_called()


class TestXpuMxfp8PrepackIntegration(TestCase):
    @config.patch(
        {
            "freezing": False,
            "force_disable_caches": True,
            "xpu.mxfp8_weight_scale_prepack": True,
        }
    )
    def test_compile_static_weight_scale(self, device):
        module = Mxfp8Linear(device)
        activation, activation_scale = make_mxfp8_inputs(device)
        with torch.inference_mode():
            expected = module(activation, activation_scale)
            actual, (code,) = run_and_get_code(
                torch.compile(module, fullgraph=True), activation, activation_scale
            )

        self.assertEqual(actual, expected)
        FileCheck().check("_mxfp8_packed_scale_0").check_not(
            "aten.clone.default"
        ).run(code)

    @config.patch(
        {
            "freezing": False,
            "force_disable_caches": True,
            "xpu.mxfp8_weight_scale_prepack": False,
        }
    )
    def test_config_disabled(self, device):
        module = Mxfp8Linear(device)
        activation, activation_scale = make_mxfp8_inputs(device)
        with torch.inference_mode():
            expected = module(activation, activation_scale)
            actual, (code,) = run_and_get_code(
                torch.compile(module, fullgraph=True), activation, activation_scale
            )

        self.assertEqual(actual, expected)
        FileCheck().check("aten.clone.default").check_not(
            "_mxfp8_packed_scale_"
        ).run(code)


instantiate_device_type_tests(
    TestXpuMxfp8PrepackIntegration,
    globals(),
    only_for="xpu",
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()
