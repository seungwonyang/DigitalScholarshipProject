STANFORD_CORE_LOCAL_PATH = r'C:\Users\febrah1\AppData\Local\Programs\Python\Python37\Lib\site-packages\stanfordcorenlp\stanford-corenlp-full-2018-10-05'
class Preprocessing:
    def __init__(self, dataPath):
        self.path = dataPath
        self.stanfordNlp = StanfordCoreNLP(STANFORD_CORE_LOCAL_PATH)
        with open(self.path) as file:
            self.sentences = list(file)


    def GetTokens(self, startIndex = 0, endIndex = 0):
        if endIndex == 0:
            endIndex = len(self.sentences)
        tokenize_list = []
        for index in range(startIndex, endIndex):
            tokenize_list.append(self.stanfordNlp.word_tokenize(self.sentences[index]))
        return tokenize_list

    def GetPosTags(self, startIndex = 0, endIndex = 0):
        if endIndex == 0:
            endIndex = len(self.sentences)
        pos_list = []
        for index in range(startIndex, endIndex):
            pos_list.append(self.stanfordNlp.pos_tag(self.sentences[index]))
        return pos_list

    def GetNER(self, startIndex = 0, endIndex = 0):
        if endIndex == 0:
            endIndex = len(self.sentences)
        ner_list = []
        for index in range(startIndex, endIndex):
            ner_list.append(self.stanfordNlp.ner(self.sentences[index]))
        return ner_list

    def GetParses(self, startIndex = 0, endIndex = 0):
        if endIndex == 0:
            endIndex = len(self.sentences)
        parse_list = []
        for index in range(startIndex, endIndex):
            parse_list.append(self.stanfordNlp.parse(self.sentences[index]))
        return parse_list

    def GetDependencyParse(self, startIndex = 0, endIndex = 0):
        if endIndex == 0:
            endIndex = len(self.sentences)
        dependency_list = []
        for index in range(startIndex, endIndex):
            dependency_list.append(self.stanfordNlp.dependency_parse(self.sentences[index]))
        return dependency_list