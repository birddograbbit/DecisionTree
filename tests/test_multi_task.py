import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.multi_task import MultiTaskTransformer


def test_multi_task_forward():
    base = TimeSeriesTransformer()
    def get_features(x):
        feat = base.input_fc(x)
        feat = feat + base.pos_embedding[:,:feat.size(1),:]
        feat = base.transformer_encoder(feat)
        return feat[:,-1,:]
    base.get_features = get_features
    task_cfg = {
        'task1': {'input_dim':base.d_model,'hidden_dim':16,'output_dim':1},
        'task2': {'input_dim':base.d_model,'hidden_dim':8,'output_dim':2},
    }
    model = MultiTaskTransformer(base, task_cfg)
    out = model(torch.randn(2,30,9))
    assert 'task1' in out and 'task2' in out
