from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from model.KGNN import KGNN
from utils import build_dataset
import spacy
import logging
import random
from googletrans import Translator

app = Flask(__name__)

# Global variables to store model and resources
model = None
vocab = None
args = None
nlp = None
embeddings = None
graph_embeddings = None
translator = Translator()

def load_model_and_resources():
    global model, vocab, args, nlp, embeddings, graph_embeddings
    
    print("\n=== Bắt đầu load model và tài nguyên ===")
    
    # Set random seeds for reproducibility
    seed = 14
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load spaCy for aspect extraction
    print("1. Đang load model spaCy...")
    try:
        nlp = spacy.load("en_core_web_lg")
        print("   ✓ Đã load model spaCy thành công")
    except:
        logging.error("Failed to load spaCy model. Please install: pip install spacy && python -m spacy download en_core_web_lg")
        raise

    # Model configuration
    print("\n2. Đang khởi tạo cấu hình model...")
    class Args:
        def __init__(self):
            self.ds_name = "14semeval_laptop"
            self.bs = 32
            self.dropout_rate = 0.5
            self.learning_rate = 0.00003
            self.n_epoch = 20
            self.model = "KGNN"
            self.dim_w = 300
            self.dim_k = 200
            self.is_test = 0
            self.is_bert = 0
            self.save_dir = "model_weight/temp"
            self.gcn = 0
            self.kge = "distmult"
    
    args = Args()
    print("   ✓ Đã khởi tạo cấu hình model")
    
    # Load vocab
    print("\n3. Đang load từ điển từ vựng...")
    vocab_path = f'dataset_npy/vocab_{args.ds_name}_dep.npy'
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    vocab = np.load(vocab_path, allow_pickle=True).tolist()
    print(f"   ✓ Đã load từ điển từ vựng từ {vocab_path}")
    
    # Load dataset to get sequence lengths and embeddings
    print("\n4. Đang load dataset để lấy độ dài sequence và embeddings...")
    dataset, embeddings, graph_embeddings, n_train, n_test = build_dataset(args=args)
    args.embeddings = embeddings
    args.graph_embeddings = graph_embeddings
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    print(f"   ✓ Đã load dataset: {n_train} mẫu train, {n_test} mẫu test")
    print(f"   ✓ Độ dài sequence: {args.sent_len}, độ dài aspect: {args.target_len}")
    
    # Initialize model
    print("\n5. Đang khởi tạo model KGNN...")
    model = KGNN(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Đang sử dụng device: {device}")
    model = model.to(device)
    
    # Load model weights
    print("\n6. Đang load trọng số model...")
    model_path = "./model_weight/best_model_weight/KGNN_14semeval_laptop_78.91_75.21.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    save_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in save_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print(f"   ✓ Đã load trọng số model từ {model_path}")
    print("\n=== Hoàn thành load model và tài nguyên ===\n")

def encode_single_input(sentence, aspect, vocab, args):
    # Reset random seeds for each encoding to ensure reproducibility
    seed = 14
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    words = sentence.strip().split()
    seq_len = args.sent_len
    aspect_tokens = aspect.strip().split()
    
    # Convert words to ids using vocab
    wids = [vocab.get(w, vocab.get('<unk>', 0)) for w in words]
    if len(wids) < seq_len:
        wids += [0] * (seq_len - len(wids))
    else:
        wids = wids[:seq_len]
    
    # Convert aspect tokens to ids
    aspect_ids = [vocab.get(w, vocab.get('<unk>', 0)) for w in aspect_tokens]
    if len(aspect_ids) < args.target_len:
        aspect_ids += [0] * (args.target_len - len(aspect_ids))
    else:
        aspect_ids = aspect_ids[:args.target_len]
    
    # Create position weights and adjacency matrix
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

def extract_aspects(sentence):
    if nlp is None:
        raise ImportError("spaCy model not loaded")
    doc = nlp(sentence)
    aspects = []
    for chunk in doc.noun_chunks:
        meaningful_words = [token.text.lower() for token in chunk if not token.is_stop]
        if len(meaningful_words) > 0:
            aspect = " ".join(meaningful_words).strip()
            if aspect and len(aspect.split()) > 0:
                aspects.append(aspect)
    return list(set(aspects))

def load_test_data():
    test_file = './dataset/14semeval_laptop/test_clean.txt'
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Xóa các nhãn sentiment (p/n/0) và aspect
            clean_line = ' '.join([word.split('/')[0] for word in line.strip().split()])
            test_data.append(clean_line)
    return test_data

@app.route('/')
def index():
    test_data = load_test_data()
    return render_template('index.html', test_data=test_data)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    if model is None or vocab is None or args is None:
        return jsonify({"results": ["Model chưa được load. Vui lòng khởi động lại server."]}), 200
        
    user_text = request.form.get('text')
    if not user_text:
        return jsonify({"results": ["Vui lòng nhập văn bản!"]}), 200

    try:
        print(f"\n=== Bắt đầu phân tích câu: {user_text} ===")
        
        # Translate input text to Vietnamese
        try:
            translated_text = translator.translate(user_text, src='en', dest='vi').text
            print(f"   ✓ Đã dịch văn bản sang tiếng Việt")
        except Exception as e:
            print(f"   ✗ Lỗi khi dịch văn bản: {str(e)}")
            translated_text = "Không thể dịch văn bản"
        
        print("1. Đang trích xuất aspect...")
        aspects = extract_aspects(user_text)
        if not aspects:
            print("   ✗ Không tìm thấy aspect nào")
            return jsonify({"results": ["Không tìm thấy aspect nào trong câu."]}), 200
        print(f"   ✓ Đã tìm thấy {len(aspects)} aspect: {', '.join(aspects)}")

        # Translate aspects to Vietnamese
        translated_aspects = []
        for aspect in aspects:
            try:
                translated_aspect = translator.translate(aspect, src='en', dest='vi').text
                translated_aspects.append(translated_aspect)
                print(f"   ✓ Đã dịch aspect '{aspect}' sang '{translated_aspect}'")
            except Exception as e:
                print(f"   ✗ Lỗi khi dịch aspect '{aspect}': {str(e)}")
                translated_aspects.append("Không thể dịch")

        results = []
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        sentiment_map = {
            'positive': 'Tích cực',
            'negative': 'Tiêu cực',
            'neutral': 'Trung lập'
        }
        
        print("\n2. Đang dự đoán sentiment cho từng aspect...")
        for i, (aspect, translated_aspect) in enumerate(zip(aspects, translated_aspects), 1):
            print(f"\n   Aspect {i}/{len(aspects)}: {aspect}")
            
            # Reset model to eval mode for each prediction
            model.eval()
            
            # Encode input
            x, xt, pw, adj, mask = encode_single_input(user_text, aspect, vocab, args)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Make prediction
            with torch.no_grad():
                logits = model(x, xt, pw, adj, mask)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                sentiment = label_map[pred]
                print(f"   ✓ Sentiment: {sentiment}")
                print(f"   ✓ Xác suất: positive={probs[0]:.2f}, negative={probs[1]:.2f}, neutral={probs[2]:.2f}")
                results.append({
                    "aspect": aspect,
                    "translated_aspect": translated_aspect,
                    "sentiment": sentiment,
                    "translated_sentiment": sentiment_map[sentiment],
                    "probabilities": {
                        "positive": float(probs[0]),
                        "negative": float(probs[1]),
                        "neutral": float(probs[2])
                    }
                })
        
        print("\n=== Hoàn thành phân tích câu ===\n")
        return jsonify({
            "results": results,
            "translated_text": translated_text
        }), 200
        
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        print(f"\n✗ Lỗi xử lý: {str(e)}\n")
        return jsonify({"results": [f"Lỗi xử lý: {str(e)}"]}), 200

@app.route('/delete_test_line', methods=['POST'])
def delete_test_line():
    try:
        data = request.get_json()
        line_index = data.get('index')
        
        # Read all lines from test_clean.txt
        with open('dataset/14semeval_laptop/test_clean.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Remove the line at specified index
        if 0 <= line_index < len(lines):
            lines.pop(line_index)
            
            # Write back to file
            with open('dataset/14semeval_laptop/test_clean.txt', 'w', encoding='utf-8') as f:
                f.writelines(lines)
                
            return jsonify({'success': True, 'message': 'Line deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Invalid line index'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    try:
        print("\n=== Khởi động server ===")
        load_model_and_resources()
        print("\n=== Server đã sẵn sàng, đang chạy tại http://localhost:5000 ===")
        print("Để truy cập giao diện web, mở trình duyệt và truy cập: http://localhost:5000")
        print("Nhấn Ctrl+C để dừng server\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
        print(f"\n✗ Không thể khởi động server: {str(e)}\n")
        sys.exit(1)
