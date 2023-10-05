import logging
import json

from typing import List, Dict, Union
from abc import ABC
from openprompt.data_utils import InputExample

class Processor(ABC):
    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__()
        self.template = None
        if template_id is not None and hasattr(self, 'templates'):
            self.template = self.templates[template_id]
        self.labelwords = None
        if verbalizer_id is not None and hasattr(self, 'verbalizers'):
            verbalizer = self.verbalizers[verbalizer_id]
        if verbalizer_file is not None:
            with open(verbalizer_file, 'r') as f:
                verbalizer_all = json.load(f)
            if isinstance(verbalizer_all, list):
                verbalizer = verbalizer_all[verbalizer_id if verbalizer_id is not None else 0]
            else:
                verbalizer = verbalizer_all
        if verbalizer_id is not None or verbalizer_file is not None:
            if isinstance(verbalizer, list):
                self.labelwords = verbalizer
            elif isinstance(verbalizer, dict):
                self.labelwords = list(verbalizer.values())
           
class AGNewsProcessor(Processor):
    templates = [
        '{"mask"} news: {"placeholder": "text_a", "shortenable": True}',
        '{"placeholder": "text_a", "shortenable": True} This topic is about {"mask"}',
        '[Category: {"mask"}] {"placeholder": "text_a", "shortenable": True}',
        '[Topic: {"mask"}] {"placeholder": "text_a", "shortenable": True}', 
    ]

    verbalizers = [
        [['world', 'politics'], ['sports'], ['business'], ['science', 'technology']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['World', 'Sports', 'Business', 'Sci/Tech']
        self.metrics = ['acc']
        self.inputs = ['text']
        self.output = 'label'
        # self.labelwords = [['world', 'politics'], ['sports'], ['business'], ['science', 'technology']]
        # self.template = None if template_id is None else AGNewsProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['text'].replace('\\', ' ')
        label = data['label']
        return InputExample(meta=data, text_a=text_a, label=label)
        
class YahooProcessor(Processor):
    templates = [
        '{"mask"} question: {"meta": "question_title"} {"meta": "question_content", "shortenable": True} {"meta": "best_answer", "shortenable": True}',
        '{"meta": "question_title"} {"meta": "question_content", "shortenable": True} {"meta": "best_answer", "shortenable": True} This topic is about {"mask"}',
        '[Topic: {"mask"}] {"meta": "question_title"} {"meta": "question_content", "shortenable": True} {"meta": "best_answer", "shortenable": True}',
        '[Category: {"mask"}] {"meta": "question_title"} {"meta": "question_content", "shortenable": True} {"meta": "best_answer", "shortenable": True}' 
    ]

    verbalizers = [
        [
            ['society', 'culture'], 
            ['science', 'mathematics'], 
            ['health'],
            ['education', 'reference'], 
            ['computers', 'internet'],
            ['sports'],
            ['business', 'finance'],
            ['entertainment', 'music'],
            ['family', 'relationships'],
            ['politics', 'government']
        ]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = [
            'Society & Culture', 
            'Science & Mathematics', 
            'Health',
            'Education & Reference', 
            'Computers & Internet',
            'Sports',
            'Business & Finance',
            'Entertainment & Music',
            'Family & Relationships',
            'Politics & Government'
        ]
        self.metrics = ['acc']
        self.inputs = ['question_title', 'question_content', 'best_answer']
        self.output = 'topic'
        # self.labelwords = [
        #     ['society', 'culture'], 
        #     ['science', 'mathematics'], 
        #     ['health'],
        #     ['education', 'reference'], 
        #     ['computers', 'internet'],
        #     ['sports'],
        #     ['business', 'finance'],
        #     ['entertainment', 'music'],
        #     ['family', 'relationships'],
        #     ['politics', 'government']
        # ]
        # self.template = YahooProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['question_title'].replace('\\', ' ') + ' '
        text_a += data['question_content'].replace('\\', ' ') + ' '
        text_a += data['best_answer'].replace('\\', ' ')
        label = data['topic']
        idx = data['id']
        return InputExample(guid=idx, meta=data, text_a=text_a, label=label)

    
class DBpediaProcessor(Processor):
    templates = [
        '{"meta": "title"}. {"meta": "content", "shortenable": True} In this sentence, {"meta": "title"} is {"mask"}.',
        '{"meta": "title"}. {"meta": "content", "shortenable": True} {"meta": "title"} is {"mask"}.',
        '{"meta": "title"}. {"meta": "content", "shortenable": True} The category of {"meta": "title"} is {"mask"}.',
        '{"meta": "title"}. {"meta": "content", "shortenable": True} The type of {"meta": "title"} is {"mask"}.'
    ]

    verbalizers = [
        [
            ['company'], 
            ['educational', 'institution'], 
            ['artist'], 
            ['athlete', 'sport'], 
            ['office'], 
            ['transportaion'], 
            ['building'], 
            ['natural', 'place'], 
            ['village'], 
            ['animal'], 
            ['plant'], 
            ['album'], 
            ['film'], 
            ['written', 'work']
        ]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = [
            "Company", 
            "EducationalInstitution", 
            "Artist", 
            "Athlete", 
            "OfficeHolder", 
            "MeanOfTransportation", 
            "Building", 
            "NaturalPlace", 
            "Village", 
            "Animal", 
            "Plant", 
            "Album", 
            "Film", 
            "WrittenWork"
        ]
        self.metrics = ['acc']
        self.inputs = ['title', 'content']
        self.output = 'label'
        # self.labelwords = [
        #     ['company'], 
        #     ['educational', 'institution'], 
        #     ['artist'], 
        #     ['athlete', 'sport'], 
        #     ['office'], 
        #     ['transportaion'], 
        #     ['building'], 
        #     ['natural', 'place'], 
        #     ['village'], 
        #     ['animal'], 
        #     ['plant'], 
        #     ['album'], 
        #     ['film'], 
        #     ['written', 'work']
        # ]
        # self.template = DBpediaProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['title'] + ' ' + data['content']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, label=label)


class IMDbProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} All in all, it was {"mask"}.'
        'It was {"mask"}. {"placeholder": "text_a"}'
        'Just {"mask"} ! {"placeholder": "text_a"}'
        '{"placeholder": "text_a"} In summary, the film was {"mask"}'
    ]
    verbalizers = [
        [['bad'], ['good']],
        [['terrible'], ['great']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['neg', 'pos']
        self.metrics = ['acc', 'f1']
        self.inputs = ['text']
        self.output = 'label'
        # self.labelwords = IMDbProcessor.verbalizers[verbalizer_id]
        # self.template =IMDbProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['text']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, label=label)

class SST2Processor(Processor):
    templates = [
        '{"meta": "sentence"} All in all, it was {"mask"}.'
        'It was {"mask"}. {"meta": "sentence"}'
        'Just {"mask"} ! {"meta": "sentence"}'
        '{"meta": "sentence"} A {"mask"} one.'
    ]
    verbalizers = [
        [['terrible'], ['great']],
        [['bad'], ['good']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['negative', 'positive']
        self.metrics = ['acc', 'f1']
        self.inputs = ['sentence']
        self.output = 'label'
        # self.labelwords = SST2Processor.verbalizers[verbalizer_id]
        # self.template = SST2Processor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['sentence']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, label=label)


class SST5Processor(Processor):
    templates = [
        '{"meta": "text"} All in all, it was {"mask"}.'
        'It was {"mask"}. {"meta": "text"}'
        'Just {"mask"} ! {"meta": "text"}'
        '{"meta": "text"} A {"mask"} one.'
    ]

    verbalizers = [
        [['terrible'], ['bad'], ['okay'], ['good'], ['great']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        self.metrics = ['acc']
        self.inputs = ['text']
        self.output = 'label'
        # self.labelwords = ['terrible', 'bad', 'okay', 'good', 'great']
        # self.template = SST5Processor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['text']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, label=label)
    
class YelpProcessor(Processor):
    templates = [
        '{"meta": "text"} All in all, it was {"mask"}.'
        'It was {"mask"}. {"meta": "text"}'
        'Just {"mask"} ! {"meta": "text"}'
        '{"meta": "text"} In summary, the restaurant is {"mask"}.'
    ]

    verbalizers = [
        [['terrible'], ['bad'], ['okay'], ['good'], ['great']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
        self.metrics = ['acc']
        self.inputs = ['text']
        self.output = 'label'
        # self.labelwords = ['terrible', 'bad', 'okay', 'good', 'great']
        # self.template = YelpProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['text']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, label=label)


class MNLIProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} ? {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" ? {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['maybe'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['entailment', 'neutral', 'contradiction']
        self.metrics = ['acc']
        self.inputs = ['premise', 'hypothesis']
        self.output = 'label'
        # self.labelwords = ['yes', 'maybe', 'no']
        # self.template = MNLIProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['premise']
        text_b = data['hypothesis']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, text_b=text_b, label=label)


class SNLIProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} ? {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" ? {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['maybe'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['entailment', 'neutral', 'contradiction']
        self.metrics = ['acc']
        self.inputs = ['premise', 'hypothesis']
        self.output = 'label'
        # self.labelwords = ['yes', 'maybe', 'no']
        # self.template = SNLIProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['premise']
        text_b = data['hypothesis']
        label = data['label']
        return InputExample(meta=data, text_a=text_a, text_b=text_b, label=label)


class QNLIProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} ? {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" ? {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['entailment', 'not_entailment']
        self.metrics = ['acc']
        self.inputs = ['question', 'sentence']
        self.output = 'label'
        # self.labelwords = ['yes', 'no']
        # self.template = QNLIProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['question']
        text_b = data['sentence']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, text_b=text_b, label=label)


class RTEProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} ? {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" ? {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['entailment', 'not_entailment']
        self.metrics = ['acc']
        self.inputs = ['sentence1, sentence2']
        self.output = 'label'
        # self.labelwords = ['yes', 'no']
        # self.template = RTEProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['sentence']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, label=label)


class MRPCProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['not_equivalent', 'equivalent']
        self.metrics = ['acc', 'f1']
        self.inputs = ['sentence1', 'sentence2']
        self.output = 'label'
        self.labelwords = ['yes', 'no']
        # self.template = MRPCProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['sentence1']
        text_b = data['sentence2']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, text_b=text_b, label=label)


class QQPProcessor(Processor):
    templates = [
        '{"placeholder": "text_a"} {"mask"}, {"place_holder": "text_b"}',
        '"{"placeholder": "text_a"}" {"mask"}, "{"place_holder": "text_b"}"'
    ]

    verbalizers = [
        [['yes'], ['no']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['not_duplicate', 'duplicate']
        self.metrics = ['acc', 'f1']
        self.inputs = ['question1', 'question2']
        self.output = 'label'
        self.labelwords = ['yes', 'no']
        # self.template = QQPProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['question1']
        text_b = data['question2']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, text_b=text_b, label=label)


class COLAProcessor(Processor):
    templates = [
        '{"meta": "text"} This is {"mask"}.',
        'It is {"mask"}. {"meta": "text"}'
    ]

    verbalizers = [
        [['incorrect'], ['correct']],
        [['wrong'], ['right']]
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = ['unacceptable', 'acceptable']
        self.metrics = ['matcor']
        self.inputs = ['sentence']
        self.output = 'label'
        # self.labelwords = COLAProcessor.verbalizers[verbalizer_id]
        # self.template = COLAProcessor.templates[template_id]
        
    def get_example(self, data):
        text_a = data['sentence']
        label = data['label']
        idx = data['idx']
        return InputExample(guid=idx, meta=data, text_a=text_a, label=label)
    
class FactivaProcessor(Processor):

    templates = [
        'Nouvelle {"mask"}: {"placeholder": "text_a", "shortenable": True}',
        'Actualité {"mask"}: {"placeholder": "text_a", "shortenable": True}',
        '{"mask"}: {"placeholder": "text_a", "shortenable": True}',
        '[Catégorie: {"mask"}] {"placeholder": "text_a", "shortenable": True}', 
    ]

    verbalizers = [
        {
            'ADMINISTRATION': ['administration'],
            'AERONAUTIQUE / ARMEMENT': ['aéronautique', 'armement'],
            'AGRO-ALIMENTAIRE': ['alimentaire', 'agriculture'],
            'AUTOMOBILE': ['automobile'],
            'BIENS DE CONSOMMATION (Production / Fabrication)': ['consommation', 'production', 'fabrication'],
            'BTP': ['bâtiment', 'travaux'],
            'COMMERCE INTERNATIONALE DES MATIERES PREMIERES': ['commerce', 'matière'],
            'COMMUNICATION': ['communication'],
            'CONSTRUCTION MECANIQUE ET ELECTRIQUE': ['construction', 'mécanique', 'électrique'],
            'DISTRIBUTION-COMMERCE': ['distribution', 'commerce'],
            'ELECTRICITE': ['électricité'],
            'FINANCE': ['finance'],
            'HOLDINGS': ['holdings'],
            'INDUSTRIE DE BASE': ['industrie', 'base'],
            'INFORMATIQUE et TECHNOLOGIES': ['informatique', 'technologies'],
            'INGENIERIE ET RECHERCHE': ['ingénerie', 'recherche'],
            'LOCATIONS ET SERVICES IMMOBILIERS': ['location', 'immobilier'],
            'METAUX': ['métaux'],
            'PETROLE - GAZ': ['pétrole', 'gaz'],
            'PIM': ['promotion', 'immobilier'],
            'SANTE - INDUSTRIE PHARMACEUTIQUE': ['santé', 'pharmacie'],
            'SERVICES': ['services'],
            'SERVICES AUX COLLECTIVITES': ['collectives'],
            'TELECOMMUNICATIONS': ['télécommunication'],
            'TOURISME-HOTELLERIE-RESTAURATION': ['tourisme', 'hôtellerie', 'restauration'],
            'TRANSPORT': ['transport'],
        }
    ]

    def __init__(self, template_id=None, verbalizer_id=None, verbalizer_file=None, **kwargs):
        super().__init__(template_id, verbalizer_id, verbalizer_file)
        self.labels = kwargs['classes']
        self.metrics = ['acc', 'f1a']
        self.inputs = ['title', 'body', 'snippet']
        self.output = 'sector'
        if verbalizer_id is not None:
            self.labelwords = [FactivaProcessor.verbalizers[verbalizer_id][label] for label in self.labels]
        elif verbalizer_file is None:
            self.labelwords = None
        # self.template = FactivaProcessor.templates[template_id]     

    def get_example(self, data):
        try:
            text_a = data['title'] + data['snippet'] + data['body']
        except KeyError:
            text_a = data['text']
            self.inputs = ['text']
        label = data['sector']
        guid = data['index']
        return InputExample(guid=guid, meta=data, text_a=text_a, label=label)
    
PROCESSORS = {
    'ag_news': AGNewsProcessor,
    'yahoo': YahooProcessor,
    'dbpedia': DBpediaProcessor,
    'imdb': IMDbProcessor,
    'sst2': SST2Processor,
    'sst5': SST5Processor,
    'yelp': YelpProcessor,
    'mnli': MNLIProcessor,
    'snli': SNLIProcessor,
    'qnli': QNLIProcessor,
    'rte': RTEProcessor,
    'mrpc': MRPCProcessor,
    'qqp': QQPProcessor,
    'cola': COLAProcessor,
    'factiva': FactivaProcessor,
}

def get_processor(name):
    return PROCESSORS[name]

def process_dataset(processor, dataset):
    logging.debug(dataset)
    return [processor.get_example(data) for data in dataset]

def has_multiple_verbalizers(name):
    return hasattr(PROCESSORS[name], 'verbalizers')

def set_template(processor, template_id=None, template_file=None):
    assert (template_id is None) or (template_file is None), "Cannot set template from both id and file."
    if template_id is not None:
        processor.template = processor.templates[template_id]
    if template_file is not None:
        with open(template_file, 'r') as f:
            processor.template = f.read()

def set_verbalizer(processor, verbalizer_id=None, verbalizer_file=None):
    print(verbalizer_id, verbalizer_file)
    assert (verbalizer_id is None) or (verbalizer_file is None), "Cannot set verbalizer from both id and file."
    if verbalizer_id is not None:
        if hasattr(processor, 'verbalizers'):
            processor.labelwords = processor.verbalizers[verbalizer_id]
        else:
            assert verbalizer_id==0, "Processor has only one default verbalizer"
    if verbalizer_file is not None:
        with open(verbalizer_file, 'r') as f:
            label_words = []
            for line in f:
                label_words.append([w.strip() for w in line.split(',')])
            processor.labelwords = label_words
