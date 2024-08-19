from datasets import load_dataset


class FeTaQADataset():
    def __init__(self):
        self.dataset = load_dataset('DongfuJiang/FeTaQA')
        self.train = self.dataset['train']
        self.validation = self.dataset['validation']
        self.test = self.dataset['test']


class QTSummDataset():
    def __init__(self):
        self.auth_token = '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'
        self.dataset = load_dataset('yale-nlp/QTSumm', token=self.auth_token)
        self.train = self.dataset['train']
        self.validation = self.dataset['validation']
        self.test = self.dataset['test']
