import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from captum.attr import IntegratedGradients
from safetensors.torch import load_file

# ─── ETİKET TANIMLARI ───────────────────────────────────────────────────
rare_drop = [
    "severe_toxicity","other_disability",
    "intellectual_or_learning_disability","other_sexual_orientation",
    "other_gender","other_race_or_ethnicity",
    "other_religion","physical_disability",
    "hindu","buddhist"
]

normal_labels = [
    "severe_toxicity","obscene","identity_attack","insult","threat"
]
normal_labels = [l for l in normal_labels if l not in rare_drop]

imbal_labels = [
    "asian","atheist","bisexual","black","buddhist","christian","female",
    "heterosexual","hindu","homosexual_gay_or_lesbian","jewish","latino",
    "male","muslim","psychiatric_or_mental_illness","transgender","white"
]
imbal_labels = [l for l in imbal_labels if l not in rare_drop]

# ─── CİHAZ ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── KAYIP ve MODEL TANIMI ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt  = torch.exp(-bce)
        return ((1-pt)**self.gamma * bce).mean()

class DualHeadBert(nn.Module):
    def __init__(self, pos_weight_n):
        super().__init__()
        self.bert   = BertModel.from_pretrained("bert-base-uncased")
        hidden     = self.bert.config.hidden_size
        self.head_n = nn.Linear(hidden, len(normal_labels))
        self.head_i = nn.Linear(hidden, len(imbal_labels))
        self.loss_n = nn.BCEWithLogitsLoss(pos_weight=pos_weight_n)
        self.loss_i = FocalLoss()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                inputs_embeds=None,
                head="n",
                labels=None):
        if inputs_embeds is not None:
            out = self.bert(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        else:
            out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled = out.pooler_output

        if head == "n":
            logits = self.head_n(pooled)
            loss   = self.loss_n(logits, labels) if labels is not None else None
        else:
            logits = self.head_i(pooled)
            loss   = self.loss_i(logits, labels) if labels is not None else None

        return {"loss": loss, "logits": logits}

# ─── MODEL YÜKLEME ───────────────────────────────────────────────────────
# pos_weight eğitiminizde kullandığınız değerleri buraya da koyabilirsiniz.
pos_weight_n = torch.ones(len(normal_labels), device=device)

# Normal-head
model_n = DualHeadBert(pos_weight_n).to(device)
sd_n = load_file("saved_model_n/model.safetensors")
model_n.load_state_dict(sd_n)
model_n.to(device)
model_n.eval()

# Imbalanced-head
model_i = DualHeadBert(pos_weight_n).to(device)
sd_i = load_file("saved_model_i/model.safetensors")
model_i.load_state_dict(sd_i)
model_i.to(device)
model_i.eval()

# Tokenizer
tok = BertTokenizer.from_pretrained("saved_tokenizer")

# ─── ATTRIBUTION FONKSİYONU ───────────────────────────────────────────────
def token_attributions(model, text, target_index, top_k=3):
    inputs = tok(text, return_tensors="pt", truncation=True,
                 padding=True, max_length=128).to(device)
    ids   = inputs["input_ids"]
    mask  = inputs["attention_mask"]

    embed_layer = model.bert.get_input_embeddings()
    embeds      = embed_layer(ids)
    embeds.requires_grad_()

    def forward_emb(x):
        out = model(inputs_embeds=x, attention_mask=mask, head="n")
        return out["logits"]

    ig   = IntegratedGradients(forward_emb)
    attr = ig.attribute(embeds, target=target_index, n_steps=50)   # 🛠️ sadece 1 değişkene alıyoruz

    scores = attr.sum(dim=-1).squeeze(0).cpu().tolist()
    toks   = tok.convert_ids_to_tokens(ids[0].cpu().tolist())[1:-1]
    top    = sorted(zip(toks, scores),
                    key=lambda x: abs(x[1]),
                    reverse=True)[:top_k]
    return [w.lstrip("##") for w,_ in top]


# ─── FASTAPI ─────────────────────────────────────────────────────────────
app = FastAPI(title="Threat & Hate-Speech Classifier")

class Query(BaseModel):
    text: str
    top_k: int = 3

@app.post("/predict")
def predict(q: Query):
    text, k = q.text, q.top_k
    inputs  = tok(text, return_tensors="pt", truncation=True,
                  padding=True, max_length=128).to(device)

    with torch.no_grad():
        out_n = model_n(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        head="n")["logits"]
        out_i = model_i(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        head="i")["logits"]

    probs_n = torch.sigmoid(out_n).cpu().numpy()[0]
    probs_i = torch.sigmoid(out_i).cpu().numpy()[0]

    result = {}
    for idx,lbl in enumerate(normal_labels):
        if probs_n[idx] > 0.5:
            result[lbl] = {"prob": float(probs_n[idx]),
                           "tokens": token_attributions(model_n, text, idx, k)}
    for idx,lbl in enumerate(imbal_labels):
        if probs_i[idx] > 0.5:
            result[lbl] = {"prob": float(probs_i[idx]),
                           "tokens": token_attributions(model_i, text, idx, k)}

    return {"input": text, "positives": result}
