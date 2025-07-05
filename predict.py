# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import warnings
import logging
from model.ATAE_LSTM import ATAE_LSTM
from model.GCAE import GCAE
from model.RGAT import RGAT
from model.ASGCN import ASGCN
from model.KGNN import KGNN
from model.bert_vanill import BERT_vanilla
from model.IAN import IAN
from model.TNet import TNet_LF
from utils import build_dataset

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Set seed for reproducibility
seed = 14
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# -------- ASPECT EXTRACTION ----------
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
except:
    nlp = None

def extract_aspects(sentence):
    # Chỉ trả về noun chunks, không fallback về full câu
    if nlp is not None:
        doc = nlp(sentence)
        # Lọc ra các noun chunks có ý nghĩa (không phải stopwords và có độ dài > 1)
        aspects = []
        for chunk in doc.noun_chunks:
            # Lọc bỏ các từ stop và chỉ lấy các chunk có ý nghĩa
            meaningful_words = [token.text.lower() for token in chunk if not token.is_stop]
            if len(meaningful_words) > 0:
                aspect = " ".join(meaningful_words).strip()
                if aspect and len(aspect.split()) > 0:  # Đảm bảo aspect không rỗng
                    aspects.append(aspect)
        return list(set(aspects))  # Loại bỏ các aspect trùng lặp
    else:
        raise ImportError("Cần cài đặt spacy và en_core_web_sm để trích xuất aspect. Chạy: pip install spacy && python -m spacy download en_core_web_sm")

def _reset_params(model):
    import torch.nn as nn
    import math
    for name, p in model.named_parameters():
        if 'bert' not in name and p.requires_grad:
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def encode_single_input(sentence, aspect, vocab, args):
    # Hàm này tương tự như 1 sample trong batch train
    words = sentence.strip().split()
    seq_len = args.sent_len
    aspect_tokens = aspect.strip().split()
    wids = [vocab.get(w, vocab.get('<unk>', 0)) for w in words]
    if len(wids) < seq_len:
        wids += [0] * (seq_len - len(wids))
    else:
        wids = wids[:seq_len]
    aspect_ids = [vocab.get(w, vocab.get('<unk>', 0)) for w in aspect_tokens]
    if len(aspect_ids) < args.target_len:
        aspect_ids += [0] * (args.target_len - len(aspect_ids))
    else:
        aspect_ids = aspect_ids[:args.target_len]
    pw = [0.0] * seq_len
    adj = np.zeros((seq_len, seq_len), dtype='float32')
    mask = [1 if i < len(words) else 0 for i in range(seq_len)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        torch.tensor([wids], dtype=torch.long, device=device),
        torch.tensor([aspect_ids], dtype=torch.long, device=device),
        torch.tensor([pw], dtype=torch.float, device=device),
        torch.tensor([adj], dtype=torch.float, device=device),
        torch.tensor([mask], dtype=torch.float, device=device)
    )

def inference(args, model_path, user_sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_path = f'dataset_npy/vocab_{args.ds_name}_dep.npy'
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Không tìm thấy vocab: {vocab_path}. Hãy chạy train trước để sinh vocab.")

    vocab = np.load(vocab_path, allow_pickle=True).tolist()
    # Chỉ lấy để biết seq_len và aspect_len (lấy từ batch 1 sample)
    dataset, embeddings, graph_embeddings, n_train, n_test = build_dataset(args=args)
    args.embeddings = embeddings
    args.graph_embeddings = graph_embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])

    # Khởi tạo model
    if args.model == 'KGNN':
        model = KGNN(args)
        _reset_params(model)
    else:
        raise NotImplementedError("Predict mode hiện chỉ support model KGNN (bạn tự thêm các model khác nếu cần).")

    model = model.to(device)
    # Load weight
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"khong the tim thay model weight: {model_path}")
    save_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in save_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    model.eval()

    # --------- EXTRACT ASPECTS ----------
    aspects = extract_aspects(user_sentence)
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    print(f"\n--- Du doan ---")
    print(f"Cau: {user_sentence}")
    for aspect in aspects:
        x, xt, pw, adj, mask = encode_single_input(user_sentence, aspect, vocab, args)
        with torch.no_grad():
            logits = model(x, xt, pw, adj, mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            print(f"Aspect: {aspect:25}")
            print(f"Setiment: {label_map[pred]:8}")
            print(f"positive={probs[0]:.2f} | negative={probs[1]:.2f} | neutral={probs[2]:.2f}")
    print()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KGNN Predict')
    parser.add_argument("-ds_name", type=str, default="14semeval_laptop")
    parser.add_argument("-bs", type=int, default=32)
    parser.add_argument("-dropout_rate", type=float, default=0.5)
    parser.add_argument("-learning_rate", type=float, default=0.00003)
    parser.add_argument("-n_epoch", type=int, default=20)
    parser.add_argument('-model', type=str, default="KGNN")
    parser.add_argument("-dim_w", type=int, default=300)
    parser.add_argument("-dim_k", type=int, default=200)
    parser.add_argument("-is_test", type=int, default=0)
    parser.add_argument("-is_bert", type=int, default=0)
    parser.add_argument("-save_dir", type=str, default="model_weight/temp")
    parser.add_argument("-gcn", type=int, default=0)
    parser.add_argument("-kge", type=str, default="distmult")
    parser.add_argument('--model_path', type=str, default="model_weight/best_model_weight/KGNN_14semeval_laptop_78.91_75.21.pth")
    parser.add_argument('--input', type=str, default=None, help='Dự đoán trực tiếp 1 câu (không vào vòng lặp)')
    args = parser.parse_args()

    model_path = args.model_path

    if args.input is not None:
        inference(args, model_path, args.input)
    else:
        print("=== Du doan ABSA / KGAN ===")
        while True:
            sent = input("Nhap cau can phan tich cam xuc: ").strip()
            if not sent:
                break
            inference(args, model_path, sent)
