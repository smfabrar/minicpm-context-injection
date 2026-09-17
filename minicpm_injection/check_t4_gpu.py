"""Verify imports and a real GPTQ operation before downloading model weights."""
import importlib.metadata


def check_gpu():
    import device_smi
    import sentencepiece
    import stepaudio2
    import torch
    import transformers
    import gptqmodel
    from gptqmodel.utils.importer import hf_select_quant_linear
    from transformers import GPTQConfig, Qwen3ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Select a T4 GPU runtime")
    config = GPTQConfig(bits=4, group_size=128, desc_act=False, sym=True,
                        backend="torch", use_exllama=False)
    if config.backend != "torch":
        raise RuntimeError("Transformers did not retain the requested PyTorch GPTQ backend")
    quant_linear = hf_select_quant_linear(bits=4, group_size=128, desc_act=False, sym=True,
                                          checkpoint_format="gptq", device_map={"": 0},
                                          backend=config.backend)
    if quant_linear.__name__ != "TorchQuantLinear":
        raise RuntimeError(f"Unexpected GPTQ backend: {quant_linear.__name__}")
    # Packed 4-bit weights are all 1, scales are 1, zero points are 0. An
    # all-ones input of length 128 must produce 128 in each output feature.
    layer = quant_linear(bits=4, group_size=128, desc_act=False, sym=True,
                         infeatures=128, outfeatures=128, bias=False).to("cuda:0")
    layer.post_init()
    with torch.inference_mode():
        layer.qweight.fill_(0x11111111)
        layer.qzeros.zero_()
        layer.scales.fill_(1)
        output = layer(torch.ones((1, 128), device="cuda:0", dtype=torch.float16))
        torch.testing.assert_close(output, torch.full_like(output, 128), rtol=0, atol=0)
    torch.cuda.synchronize()
    for name in ("torch", "transformers", "gptqmodel", "device-smi", "sentencepiece", "minicpmo-utils"):
        print(f"{name}: {importlib.metadata.version(name)}", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PASS: quantized PyTorch layer ran correctly on GPU; no custom CUDA kernel build.", flush=True)


if __name__ == "__main__":
    check_gpu()
