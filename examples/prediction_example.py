import os, sys, json, inspect
import numpy as np
import pandas as pd
import torch
import safetensors.torch as st
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import Kronos, KronosTokenizer, KronosPredictor
import numpy as np
def multi_predict(predictor, x_df, x_timestamp, y_timestamp, pred_len, n_runs=5, **kwargs):
    preds = []
    for i in range(n_runs):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            **kwargs
        )
        preds.append(pred_df.values)  # 只取数值
    mean_pred = np.mean(preds, axis=0)  # 沿着多次预测平均
    # 用第一个预测的结构来恢复 DataFrame
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        **kwargs
    )
    pred_df[:] = mean_pred  # 保持原结构，只替换数值
    return pred_df
def _filtered_kwargs(cls, cfg: dict):
    sig = inspect.signature(cls.__init__)
    valid = {k for k in sig.parameters.keys() if k != "self"}
    return {k: v for k, v in cfg.items() if k in valid}

def load_tokenizer_local(dirpath: str) -> KronosTokenizer:
    cfg = json.load(open(os.path.join(dirpath, "config.json"), "r"))
    tok = KronosTokenizer(**_filtered_kwargs(KronosTokenizer, cfg))
    tok_w = os.path.join(dirpath, "model.safetensors")
    if os.path.exists(tok_w) and hasattr(tok, "load_state_dict"):
        tok.load_state_dict(st.load_file(tok_w))
    return tok

def load_model_local(dirpath: str, device: str) -> Kronos:
    cfg = json.load(open(os.path.join(dirpath, "config.json"), "r"))
    mdl = Kronos(**_filtered_kwargs(Kronos, cfg))
    w = os.path.join(dirpath, "model.safetensors")
    if hasattr(mdl, "load_state_dict"):
        mdl.load_state_dict(st.load_file(w))
    elif hasattr(mdl, "load_weights"):
        mdl.load_weights(w)
    if device.startswith("cuda") and torch.cuda.is_available():
        mdl = mdl.to(device)
    return mdl
    
def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
ASSETS = "/home/jingyi/Kronos/assets"             
TOK_DIR = f"{ASSETS}/kronos-tokenizer"            # tokenizer: 包含 config.json / 可选 model.safetensors
MODEL_DIR = f"{ASSETS}/kronos-small"
DEVICE='cuda:0' 
# 用本地目录构造
tokenizer = load_tokenizer_local(TOK_DIR)
model = load_model_local(MODEL_DIR, device=DEVICE)

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
df = pd.read_csv("/home/jingyi/Kronos/examples/data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

# 4. Make Prediction
pred_df = multi_predict(
    predictor,
    x_df,
    x_timestamp,
    y_timestamp,
    pred_len,
    n_runs=5,      # 想要几次平均就改这里
    T=1,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# visualize
plot_prediction(kline_df, pred_df)

