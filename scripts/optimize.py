import torch
import torch.nn as nn
import torch.quantization
import time
import onnx
import onnxruntime as ort
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.quantization import get_default_qconfig, prepare_fx, convert_fx



class Quantize():
    def __init__(self, model):
        self.model = model
    
    def run(self):

        # Step 2: Set quantization config
        qconfig = get_default_qconfig("fbgemm")

        # Step 3: Prepare the model using FX
        prepared_model = prepare_fx(self.model, {"": qconfig})  # Empty string "" means apply globally

        # Step 4: Calibrate with representative input
        with torch.no_grad():
            for _ in range(10):
                dummy_input = torch.randn(1, 9, 256, 256)  # match your input
                prepared_model(dummy_input)

        # Step 5: Convert to quantized model
        quantized_model = convert_fx(prepared_model)
        quantized_model.eval()

        # Step 6: Save TorchScript or ONNX
        scripted_model = torch.jit.script(quantized_model)
        scripted_model.save("unetpp_quant_scripted.pt")
        print("✅ TorchScript FX quantized model saved.")

        # Optional: Export to ONNX
        torch.onnx.export(
            quantized_model,
            dummy_input,
            "unetpp_quant.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ ONNX quantized model saved.")


class Inference():

    # Common input
    input_np = np.random.rand(1, 3, 256, 256).astype(np.float32)
    input_tensor = torch.from_numpy(input_np)

    # TorchScript Inference
    loaded_ts = torch.jit.load("unetpp_quant_scripted.pt")
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            out = loaded_ts(input_tensor)
        print("TorchScript time:", time.time() - start)

    # ONNX Runtime Inference
    session = ort.InferenceSession("unetpp_quant.onnx", providers=["CPUExecutionProvider"])
    start = time.time()
    for _ in range(10):
        result = session.run(None, {"input": input_np})
    print("ONNX time:", time.time() - start)
