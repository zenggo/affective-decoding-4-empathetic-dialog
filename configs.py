from utils import dotdict


DEFAULT_MODEL_CFG = dotdict({
    'n_ctx': 256,
    'n_embd': 768,
    'n_emo_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'clf_pdrop': 0.1,
    'afn': 'gelu',
    'clf_hs': []
})


DEFAULT_OPT_CFG = dotdict({
    'max_grad_norm': 1,
    'lr': 6.25e-5,
    'lr_warmup': 0.002,
    'l2': 0.01,
    'vector_l2': 'store_true',
    'lr_schedule': 'warmup_linear',
    'b1': 0.9,
    'b2': 0.999,
    'e': 1e-8
})


EMOTION_CATES = [
    "surprised",
    "excited",
    "angry",
    "proud",
    "sad",
    "annoyed",
    "grateful",
    "lonely",
    "afraid",
    "terrified",
    "guilty",
    "impressed",
    "disgusted",
    "hopeful",
    "confident",
    "furious",
    "anxious",
    "anticipating",
    "joyful",
    "nostalgic",
    "disappointed",
    "prepared",
    "jealous",
    "content",
    "devastated",
    "embarrassed",
    "caring",
    "sentimental",
    "trusting",
    "ashamed",
    "apprehensive",
    "faithful"
]