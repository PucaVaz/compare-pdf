from transformers import AutoTokenizer, AutoModel

class Model:
    _instance = None
    _model = None
    _tokenizer = None

    @staticmethod
    def get_instance(device):
        if Model._instance is None:
            Model._instance = Model(device)
        return Model._instance

    def __init__(self, device):
        if Model._instance is not None:
            raise Exception("Model is a singleton and has already been initialized!")
        Model._instance = self
        self._device = device
        self.__load_model()

    def __load_model(self):
        self._tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', clean_up_tokenization_spaces=False)
        self._model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self._device)

    def get_model(self):
        return self._model

    def get_tokenizer(self):
        return self._tokenizer
