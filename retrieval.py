import faiss
import numpy as np
import re
import sys
import time
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


class Retrieval():
    def __init__(self, dataset_name, dataset):
        self.dataset_name = dataset_name
        self.dataset = dataset

        if self.dataset_name == 'FeTaQA':
            self.header_lake = [_[0] for _ in self.dataset['table_array']]
            self.value_lake = [_[1:] for _ in self.dataset['table_array']]
            self.title_lake = [_ + ' - ' + __ for _, __ in zip(self.dataset['table_page_title'], self.dataset['table_section_title'])]
            self.questions = self.dataset['question']
            self.answers = self.dataset['answer']
        elif self.dataset_name == 'QTSumm':
            self.header_lake = [_['header'] for _ in self.dataset['table']]
            self.value_lake = [_['rows'] for _ in self.dataset['table']]
            self.title_lake = [_['title'] for _ in self.dataset['table']]
            self.questions = self.dataset['query']
            self.answers = self.dataset['summary']


class SparseRetrieval(Retrieval):
    def __init__(self, dataset_name, dataset):
        super().__init__(dataset_name, dataset)

        self.tokenize_header = self.preprocessing('header')
        print("Finish header tokenizing!")
        self.tokenize_value = self.preprocessing('value')
        print("Finish value tokenizing!")
        self.tokenize_title = self.preprocessing('title')
        print("Finish title tokenizing!")
        self.tokenize_question = self.preprocessing('question')
        print("Finish question tokenizing!")

    def text_preprocessing(self, text):
        text = text.strip()
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', ' ', text)
        text = text.split()

        stop_words = stopwords.words('english')
        return np.array([word for word in text if word not in stop_words])

    def list_preprocessing(self, input_list):
        return np.hstack([self.text_preprocessing(text) for text in input_list])

    def preprocessing(self, type):
        if type == 'header':
            result = np.array([self.list_preprocessing(header) for header in self.header_lake], dtype=object)
        elif type == 'value':
            result = np.array([self.list_preprocessing(row) for rows in self.value_lake for row in rows], dtype=object)
        elif type == 'title':
            result = np.array([self.text_preprocessing(title) for title in self.title_lake], dtype=object)
        elif type == 'question':
            result = np.array([self.text_preprocessing(question) for question in self.questions], dtype=object)
        else:
            print("Invalid type!")
            sys.exit()
        return result
    
    def bm25_idf(self, data, query):
        N = len(data)
        n_qi = np.array([sum(1 for doc in data if token in doc) for token in query])
        return np.log(((N - n_qi + 0.5) / (n_qi + 0.5)) + 1)

    def bm25_tf(self, query, data):
        qi_cnt = np.array([[np.sum(doc == token) for token in query] for doc in data])
        return qi_cnt

    def bm25_score(self, data, query, k1=1.2, b=0.75):
        doc_lens = np.array([len(doc) for doc in data])
        avgdl = np.mean(doc_lens)

        idf = self.bm25_idf(data, query)
        tf = self.bm25_tf(query, data)

        numerator = idf * (tf * (k1 + 1))
        denominator = tf + k1 * (1 - b + b * (doc_lens[:, np.newaxis] / avgdl))
        score = numerator / denominator

        return np.sum(score, axis=1)

    def get_bm25_scores(self, data, queries):
        scores = []
        gt_time = time.time()
        start_time = time.time()
        for i, query in enumerate(queries):
            if i % 100 == 0:
                end_time = time.time()
                print(f"Finish {i}th query! ({end_time - start_time}sec, total {end_time - gt_time}sec)")
                start_time = end_time
            scores.append(self.bm25_score(data, query))
        return np.array(scores)
    
    def bm25_retrieval(self, retrieval_range):
        if retrieval_range == 'title':
            data = self.tokenize_title
        elif retrieval_range == 'header':
            data = self.tokenize_header
        elif retrieval_range == 'metadata':
            data = [np.concatenate([title, header]) for title, header in zip(self.tokenize_title, self.tokenize_header)]
        elif retrieval_range == 'table':
            data = [np.concatenate([header, value]) for header, value in zip(self.tokenize_header, self.tokenize_value)]
        elif retrieval_range == 'full':
            data = [np.concatenate([title, header, value]) for title, header, value in zip(self.tokenize_title, self.tokenize_header, self.tokenize_value)]
        else:
            print("Invalid retireval range!")
            sys.exit()
        queries = self.tokenize_question

        bm25_scores = self.get_bm25_scores(data, queries)
        sorted_indices = np.argsort(bm25_scores, axis=1)[:, ::-1]

        print(f"Finish score calculation with {retrieval_range} serialization!")
        return sorted_indices


class DenseRetrieval(Retrieval):
    def __init__(self, dataset_name, dataset):
        super().__init__(dataset_name, dataset)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base').to(self.device)

        self.serialize_header = self.serialization('header')
        print("Finish header serialization!")
        self.serialize_value = self.serialization('value')
        print("Finish value serialization!")
        self.serialize_title = self.serialization('title')
        print("Finish title serialization!")
        self.serialize_question = self.serialization('question')
        print("Finish question serialization!")
    
    def text_encoding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = self.model(**inputs).pooler_output
        return embeddings.squeeze().cpu().numpy()

    def list_encoding(self, text_list):
        embeddings_list = []
        for text in tqdm(text_list, desc="pretrained BERT - encoding"):
            embeddings = self.text_encoding(text)
            embeddings_list.append(embeddings)
        return np.array(embeddings_list)

    def serialization(self, type):
        if type == 'header':
            result = np.array([f"col: | {' | '.join(header)} |" for header in self.header_lake], dtype=object)
        elif type == 'value':
            result = np.array([' '.join([f"row {i}: | {' | '.join(row)} |" for i, row in enumerate(rows)]) for rows in self.value_lake], dtype=object)
        elif type == 'title':
            result = np.array([f"title: {title}" for title in self.title_lake], dtype=object)
        elif type == 'question':
            result = np.array(self.questions, dtype=object)
        else:
            print("Invalid type!")
            sys.exit()
        return result

    def dpr_retrieval(self, retrieval_range):
        if retrieval_range == 'title':
            data = self.list_encoding(self.serialize_title)
        elif retrieval_range == 'header':
            data = self.list_encoding(self.serialize_header)
        elif retrieval_range == 'metadata':
            data = self.list_encoding(self.serialize_title + " " + self.serialize_header)
        elif retrieval_range == 'table':
            data = self.list_encoding(self.serialize_header + " " + self.serialize_value)
        elif retrieval_range == 'full':
            data = self.list_encoding(self.serialize_title + " " + self.serialize_header + " " + self.serialize_value)
        else:
            print("Invalid retireval range!")
            sys.exit()
        queries = self.list_encoding(self.serialize_question)
        print(f"Finish {retrieval_range} encoding!")

        index = faiss.IndexFlatL2(data.shape[1])
        # index = faiss.index_cpu_to_all_gpus(index)
        index.add(data)

        num_docs = data.shape[0]
        sorted_indices = []
        for query in tqdm(queries, desc="Faiss library - retrieval"):
            _, sorted_index_list = index.search(np.array([query]), k=num_docs)
            sorted_indices.append(sorted_index_list[0])

        print(f"Finish score calculation with {retrieval_range} serialization!")
        return sorted_indices
