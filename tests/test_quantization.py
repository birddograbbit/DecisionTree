import pytest
import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.quantization import quantize_transformer

@pytest.mark.skipif(
    not torch.backends.quantized.engine == 'qnnpack' and torch.cuda.is_available() == False,
    reason="Quantization not supported on this platform"
)
def test_quantize_model():
    model = TimeSeriesTransformer(feature_size=3, seq_length=4)
    data = torch.randn(2, 4, 3)
    
    # Set to eval mode for quantization
    model.eval()
    
    try:
        q = quantize_transformer(model, data)
        assert q is not None
        # Test that quantized model still works
        with torch.no_grad():
            output = q(data)
        assert output.shape == (2, 1)
    except RuntimeError as e:
        if "NoQEngine" in str(e):
            pytest.skip("Quantization engine not available on this platform")
        else:
            raise
