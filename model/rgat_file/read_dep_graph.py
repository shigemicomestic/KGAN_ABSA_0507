# -*- coding: utf-8 -*-
# File: read_dep_graph.py
# Mục đích: Xử lý và tạo đồ thị phụ thuộc cú pháp (dependency graph) và đồ thị tri thức (knowledge graph)
# Các chức năng chính:
# 1. Tokenization dựa trên khoảng trắng
# 2. Tạo cấu trúc cây cho dependency graph
# 3. Tạo ma trận kề cho cả dependency và knowledge graph
# 4. Kết hợp thông tin cú pháp và ngữ nghĩa

import numpy as np
import spacy
from model.rgat_file.senticnet5 import senticnet
from spacy.tokens import Doc

# Lớp WhitespaceTokenizer
# Mục đích: Tùy chỉnh cách tokenization của spaCy để xử lý chính xác văn bản tiếng Anh
# Cách hoạt động: Tách câu thành các từ dựa trên khoảng trắng và giữ nguyên cấu trúc từ
class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab  # Từ điển từ vựng của spaCy

    def __call__(self, text):
        words = text.split()  # Tách câu thành các từ
        spaces = [True] * len(words)  # Đánh dấu vị trí khoảng trắng
        return Doc(self.vocab, words=words, spaces=spaces)  # Tạo đối tượng Doc của spaCy

# Khởi tạo spaCy model với tokenizer tùy chỉnh
# Sử dụng model 'en_core_web_lg' - model tiếng Anh lớn với word vectors
nlp = spacy.load('en_core_web_lg')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

# Lớp Tree
# Mục đích: Biểu diễn cấu trúc cây cho dependency graph
# Cấu trúc: Mỗi nút có thể có nhiều nút con và một nút cha
class Tree:
    def __init__(self, value, parent=None):
        if isinstance(value, list):
            # Khởi tạo nút gốc của cây
            self.value = 0  # Giá trị mặc định cho nút gốc
            self.parent = None  # Nút gốc không có nút cha
            self.children = []  # Danh sách các nút con
            
            # Xây dựng cây từ danh sách các đường dẫn
            for path in value:
                parent = self
                for i,v in enumerate(path):
                    node = None
                    # Kiểm tra xem nút con đã tồn tại chưa
                    for child in parent.children:
                        if v == child.getValue():
                            node = child
                            break
                    # Tạo nút mới nếu chưa tồn tại
                    if node == None:
                        node = Tree(v, parent)
                        parent.children.append(node)
                    parent = node
        else:
            # Khởi tạo nút lá
            self.value = value  # Giá trị của nút
            self.parent = parent  # Nút cha
            self.children = []  # Nút lá không có nút con

    # Các phương thức getter
    def getValue(self):
        return self.value  # Lấy giá trị của nút

    def getChildren(self):
        return self.children  # Lấy danh sách các nút con

    def getParent(self):
        return self.parent  # Lấy nút cha

# Hàm tạo cây từ SenticNet
# Mục đích: Xây dựng cấu trúc cây cho knowledge graph từ dữ liệu SenticNet
# SenticNet: Cơ sở dữ liệu ngữ nghĩa chứa thông tin về cảm xúc và ý nghĩa của từ
def get_senticnet_tree():
    values = []
    # Lấy dữ liệu từ SenticNet
    for k,v in senticnet.items():
        value = []
        value.append(k)  # Thêm từ gốc
        for i,j in enumerate(v):
            if '#' in j:
                value.append(j.replace('#',''))  # Xử lý các ký tự đặc biệt
            if i>=8:
                value.append(j)  # Thêm các thông tin ngữ nghĩa
        values.append(value)
    # Tạo cây với 2 phần tử đầu tiên
    get_senticnet_tree = Tree(value=values[:2], parent='a_little')

# Hàm tạo ma trận kề cho dependency graph
# Mục đích: Biểu diễn mối quan hệ phụ thuộc cú pháp giữa các từ trong câu
# Input: text - câu cần phân tích
# Output: ma trận kề n x n (n là số từ trong câu)
def dependency_adj_matrix(text):
    tokens = nlp(text)  # Phân tích cú pháp câu bằng spaCy
    words = text.split()
    # Khởi tạo ma trận kề với kích thước len(words) x len(words)
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))  # Kiểm tra tính nhất quán

    # Xây dựng ma trận kề
    for num,token in enumerate(tokens):
        matrix[token.i][token.i] = 1  # Tự kết nối (self-loop)
        if num!=0 and num!=(len(tokens)-1):  # Bỏ qua token đầu và cuối
            # Thêm kết nối với các từ con
            for child in token.children:
                if child.string != '[CLS] ' and child.string != '[SEP]':  # Bỏ qua các token đặc biệt
                    matrix[token.i][child.i] = 1  # Kết nối hai chiều
                    matrix[child.i][token.i] = 1
    return matrix

# Hàm tạo ma trận kề cho knowledge graph
# Mục đích: Kết hợp thông tin cú pháp và ngữ nghĩa từ SenticNet
# Input: text - câu cần phân tích
# Output: ma trận kề mở rộng và danh sách từ + tag ngữ nghĩa
def knowledge_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    assert len(words) == len(list(tokens))
    tags = {}  # Lưu trữ các tag ngữ nghĩa
    tag = {}   # Lưu trữ mapping giữa token và tag
    num = len(words)
    
    # Lấy thông tin ngữ nghĩa từ SenticNet
    for token in tokens:
        try:
            for sem in senticnet[token.text][8:]:  # Lấy các thông tin ngữ nghĩa
                if sem not in tags.keys():
                    tags[sem] = num
                    num += 1
            tag[token] = senticnet[token.text][8:]
        except KeyError:
            continue  # Bỏ qua các từ không có trong SenticNet
            
    # Khởi tạo ma trận kề mở rộng
    # Kích thước: (số từ + số tag) x (số từ + số tag)
    matrix = np.zeros((len(words)+len(tags.keys()), len(words)+len(tags.keys()))).astype('float32')
    
    # Xây dựng ma trận kề
    for token in tokens:
        matrix[token.i][token.i] = 1  # Tự kết nối
        if token in tag.keys():
            # Thêm kết nối với các tag ngữ nghĩa
            for t in tag[token]:
                if t != '[CLS] ' and t != '[SEP]':
                    matrix[token.i][tags[t]] = 1  # Kết nối hai chiều
                    matrix[tags[t]][token.i] = 1
    return matrix, words+list(tags.keys())

# Hàm xử lý đồ thị tổng hợp
# Mục đích: Kết hợp cả dependency graph và knowledge graph
# Input: text - câu cần phân tích
# Output: 
#   - syntactic_adj_matrix: ma trận kề cú pháp
#   - common_adj_matrix: ma trận kề ngữ nghĩa
#   - words_know: danh sách từ và tag ngữ nghĩa
def process_graph(text):
    # Tạo ma trận kề cú pháp
    syntactic_adj_matrix = dependency_adj_matrix(text)
    common_adj_matrix, words_know = knowledge_adj_matrix(text)
    return syntactic_adj_matrix, common_adj_matrix, words_know

