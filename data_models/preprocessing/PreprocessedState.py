class PreprocessedState:
    def __init__(self, concatenatedText, graph):
        self.concatenatedText: str = concatenatedText
        self.graphWithTriples: str = graph
