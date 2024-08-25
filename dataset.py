import json
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


class OpenWikiTableDataset():
    def __init__(self):
        self.table_dict = {}
        with open(f'source_data/openwikitable/tables.json', 'r') as file:
            table_json = json.load(file)
            for i in range(len(table_json['original_table_id'])):
                new_key = table_json['original_table_id'][str(i)]
                table_data = {}
                for key in table_json:
                    if key != 'original_table_id':
                        table_data[key] = table_json[key][str(i)]
                self.table_dict[new_key] = table_data

        self.train, self.validation, self.test = {}, {}, {}
        for split in ['train', 'validation', 'test']:
            data_dict = getattr(self, split)
            with open(f'source_data/openwikitable/{split}.json', 'r') as file:
                data_json = json.load(file)
                data_dict['question'] = list(data_json['question'].values())
                data_dict['answer'] = list(data_json['answer'].values())
                data_dict['sql'] = list(data_json['sql'].values())
                
                data_dict['tid'], data_dict['header'], data_dict['value'], data_dict['title'] = [], [], [], []
                for _, tid in data_json['original_table_id'].items():
                    data_dict['tid'].append(tid)
                    data_dict['header'].append(self.table_dict[tid]['header'])
                    data_dict['value'].append(self.table_dict[tid]['rows'])
                    data_dict['title'].append(' - '.join([self.table_dict[tid][_] for _ in ['page_title', 'section_title', 'caption']]))

        self.dataset = {'train': self.train, 'validation': self.validation, 'test': self.test}
