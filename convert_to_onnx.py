import torch
import torch.nn as nn
import argparse
import os

class FakeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias

def replace_layernorm_with_fake(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln = FakeLayerNorm(module.normalized_shape, module.eps)
            ln.weight.data = module.weight.data.clone()
            ln.bias.data = module.bias.data.clone()
            parent = model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, name_parts[-1], ln)

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX with FakeLayerNorm")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--input-size", type=int, nargs=3, default=[1, 3, 224, 224], help="Input size as [B, C, H, W]")
    parser.add_argument("--model-class", type=str, required=True, help="Import path to model class (e.g., mymodule.MyModel)")
    args = parser.parse_args()

    module_path, class_name = args.model_class.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    ModelClass = getattr(module, class_name)
    model = ModelClass()

    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    replace_layernorm_with_fake(model)
    model.eval()

    dummy_input = torch.ones(*args.input_size)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=17
    )

    try:
        import onnx
        import onnxsim
        print("Simplifying ONNX...")
        model_onnx = onnx.load(args.output)
        model_simp, check = onnxsim.simplify(model_onnx)
        assert check
        onnx.save(model_simp, args.output.replace(".onnx", ".sim.onnx"))
        print(f"ONNX simplified and saved to {args.output.replace('.onnx', '.sim.onnx')}")
    except Exception as e:
        print("ONNX simplification failed:", e)

if __name__ == "__main__":
    main()
