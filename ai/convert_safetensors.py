# convert_safetensors.py
import argparse
import torch
import torch.nn as nn
from safetensors.torch import load_file

# —————— Buraya kendi model tanımınızı (eğitim kodunuzdaki DualHeadBert) aynen yapıştırın ——————
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, labels):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt  = torch.exp(-bce)
        return ((1-pt)**self.gamma * bce).mean()

# Etiket listelerinizi aynen eğitim kodundan alın:
normal_labels = [
    "severe_toxicity","obscene","identity_attack","insult","threat"
]
# rare_drop vs. uygulayıp filtreleyin:
rare_drop = [
    "severe_toxicity","other_disability",
    "intellectual_or_learning_disability","other_sexual_orientation",
    "other_gender","other_race_or_ethnicity",
    "other_religion","physical_disability",
    "hindu","buddhist"
]
normal_labels = [l for l in normal_labels if l not in rare_drop]

imbal_labels = [
    "asian","atheist","bisexual","black","buddhist","christian","female",
    "heterosexual","hindu","homosexual_gay_or_lesbian","jewish","latino",
    "male","muslim","psychiatric_or_mental_illness","transgender","white"
]
imbal_labels = [l for l in imbal_labels if l not in rare_drop]

class DualHeadBert(nn.Module):
    def __init__(self, pos_weight_n):
        super().__init__()
        from transformers import BertModel
        self.bert   = BertModel.from_pretrained("bert-base-uncased")
        hidden     = self.bert.config.hidden_size
        self.head_n = nn.Linear(hidden, len(normal_labels))
        self.head_i = nn.Linear(hidden, len(imbal_labels))
        self.loss_n = nn.BCEWithLogitsLoss(pos_weight=pos_weight_n)
        self.loss_i = FocalLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                inputs_embeds=None, head="n", labels=None):
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
# ————————————————————————————————————————————————————————————————————————————————————————————

def convert(src, dst):
    # 1) CPU’da aynı pos_weight ile modelinizi instantiate edin
    pos_weight_n = torch.ones(len(normal_labels))
    m = DualHeadBert(pos_weight_n)

    # 2) safetensors’ı açın
    state_dict = load_file(src)  

    # 3) yükleyip .bin’e kaydedin
    m.load_state_dict(state_dict)
    torch.save(m.state_dict(), dst)
    print(f"✅ Converted {src} → {dst}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="path to .safetensors")
    p.add_argument("--dst", required=True, help="output .bin path")
    args = p.parse_args()
    convert(args.src, args.dst)
