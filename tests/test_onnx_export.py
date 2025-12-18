"""
Tests for ONNX export compatibility (T067).

Verifies that all operations can be exported to ONNX without
Loop, If, or other control flow nodes.
"""

import pytest
import torch
import io

# Skip if ONNX not installed
onnx = pytest.importorskip("onnx")


class TestONNXExportVGA:
    """Test ONNX export for VGA operations."""

    def test_geometric_product_no_loops(self):
        """Verify geometric_product exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, a, b):
                return self.algebra.geometric_product(a, b)

        model = Model(vga)
        a = torch.randn(1, vga.count_blade)
        b = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (a, b), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    def test_reverse_no_loops(self):
        """Verify reverse exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, mv):
                return self.algebra.reverse(mv)

        model = Model(vga)
        mv = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (mv,), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    def test_outer_product_no_loops(self):
        """Verify outer product exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, a, b):
                return self.algebra.outer(a, b)

        model = Model(vga)
        a = torch.randn(1, vga.count_blade)
        b = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (a, b), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"


class TestONNXExportCGA:
    """Test ONNX export for CGA operations."""

    def test_sandwich_rotor_no_loops(self):
        """Verify sandwich_rotor exports without control flow."""
        from fast_clifford import CGA

        cga = CGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, rotor, x):
                return self.algebra.sandwich_rotor(rotor, x)

        model = Model(cga)
        rotor = torch.randn(1, cga.count_rotor)
        x = torch.randn(1, cga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (rotor, x), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    def test_compose_rotor_no_loops(self):
        """Verify compose_rotor exports without control flow."""
        from fast_clifford import CGA

        cga = CGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, r1, r2):
                return self.algebra.compose_rotor(r1, r2)

        model = Model(cga)
        r1 = torch.randn(1, cga.count_rotor)
        r2 = torch.randn(1, cga.count_rotor)

        buffer = io.BytesIO()
        torch.onnx.export(model, (r1, r2), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    def test_exp_bivector_no_loops(self):
        """Verify exp_bivector exports without control flow."""
        from fast_clifford import CGA

        cga = CGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, B):
                return self.algebra.exp_bivector(B)

        model = Model(cga)
        B = torch.randn(1, cga.count_bivector) * 0.1

        buffer = io.BytesIO()
        torch.onnx.export(model, (B,), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    def test_transform_layer_no_loops(self):
        """Verify CliffordTransformLayer exports without control flow."""
        from fast_clifford import CGA

        cga = CGA(3)
        layer = cga.get_transform_layer()

        rotor = torch.randn(1, cga.count_rotor)
        point = torch.randn(1, cga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(
            layer,
            (rotor, point),
            buffer,
            input_names=["rotor", "point"],
            output_names=["output"],
            opset_version=14,
        )

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"


class TestONNXExportDynamicBatch:
    """Test ONNX export with dynamic batch dimensions."""

    def test_dynamic_batch_geometric_product(self):
        """Verify dynamic batch export works."""
        from fast_clifford import VGA

        vga = VGA(3)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, a, b):
                return self.algebra.geometric_product(a, b)

        model = Model(vga)
        a = torch.randn(2, vga.count_blade)
        b = torch.randn(2, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            (a, b),
            buffer,
            input_names=["a", "b"],
            output_names=["output"],
            dynamic_axes={
                "a": {0: "batch"},
                "b": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=14,
        )

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        # Check model is valid
        onnx.checker.check_model(onnx_model)

    def test_dynamic_batch_cga_transform(self):
        """Verify dynamic batch CGA transform export."""
        from fast_clifford import CGA

        cga = CGA(3)
        layer = cga.get_transform_layer()

        rotor = torch.randn(2, cga.count_rotor)
        point = torch.randn(2, cga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(
            layer,
            (rotor, point),
            buffer,
            input_names=["rotor", "point"],
            output_names=["output"],
            dynamic_axes={
                "rotor": {0: "batch"},
                "point": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=14,
        )

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        # Check model is valid
        onnx.checker.check_model(onnx_model)


class TestONNXExportExtendedOps:
    """Test ONNX export for extended operations."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_dual_no_loops(self, dim):
        """Verify dual exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(dim)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, mv):
                return self.algebra.dual(mv)

        model = Model(vga)
        mv = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (mv,), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    @pytest.mark.parametrize("dim", [2, 3])
    def test_normalize_no_loops(self, dim):
        """Verify normalize exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(dim)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, mv):
                return self.algebra.normalize(mv)

        model = Model(vga)
        mv = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (mv,), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"

    @pytest.mark.parametrize("dim", [2, 3])
    def test_left_contraction_no_loops(self, dim):
        """Verify left_contraction exports without control flow."""
        from fast_clifford import VGA

        vga = VGA(dim)

        class Model(torch.nn.Module):
            def __init__(self, algebra):
                super().__init__()
                self.algebra = algebra

            def forward(self, a, b):
                return self.algebra.contract_left(a, b)

        model = Model(vga)
        a = torch.randn(1, vga.count_blade)
        b = torch.randn(1, vga.count_blade)

        buffer = io.BytesIO()
        torch.onnx.export(model, (a, b), buffer, opset_version=14)

        buffer.seek(0)
        onnx_model = onnx.load(buffer)

        forbidden_ops = {"Loop", "If", "SequenceConstruct"}
        found_ops = {n.op_type for n in onnx_model.graph.node} & forbidden_ops
        assert not found_ops, f"Found forbidden ONNX ops: {found_ops}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
