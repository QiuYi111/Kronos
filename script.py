# script.py
from huggingface_hub import hf_hub_download
import json
from model import Kronos, KronosTokenizer, KronosPredictor

def load_cfg(repo_id, filename="config.json", **kwargs):
    cfg_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    with open(cfg_path, "r") as f:
        return json.load(f)

# 1) 加载 tokenizer
tok_repo = "NeoQuasar/Kronos-Tokenizer-base"
tok_cfg = load_cfg(tok_repo)  # 默认就叫 config.json
tokenizer = KronosTokenizer(**tok_cfg)

# 2) 加载 model（如果模型也有 config，则同理）
mdl_repo = "NeoQuasar/Kronos-small"
mdl_cfg = load_cfg(mdl_repo)
model = Kronos(**mdl_cfg)

# 3) Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 后面照常用 predictor.predict(...)
