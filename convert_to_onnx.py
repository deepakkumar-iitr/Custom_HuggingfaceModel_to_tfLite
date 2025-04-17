import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from safetensors.torch import load_file

model_dir = "./model"
safetensors_path = f"{model_dir}/model.safetensors"

# Load tokenizer, config, and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)
model = AutoModel.from_config(config)

# Load weights
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)
model.eval()

# Prepare dummy input
text = "This is a test."
inputs = tokenizer(text, return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"],),
    "custom_model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"}},
    opset_version=12
)

print("✅ ONNX model exported to custom_model.onnx")
