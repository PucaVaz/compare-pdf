import torch
import numpy as np

from src.embedding_system.device import Device
from src.embedding_system.model import Model

# DefineDeviceHandler: Retrieves the singleton device
class DefineDeviceHandler:
    def __init__(self):
        self.device = Device.get_instance().get_device()
    
    def handle(self):
        return self.device

# LoadModelHandler: Loads the tokenizer and model based on the device
class LoadModelHandler:
    def __init__(self):
        self.device = Device.get_instance().get_device()
        self.tokenizer, self.model = self.load_model()
    
    def load_model(self):
        model_instance = Model.get_instance(self.device)
        return model_instance.get_tokenizer(), model_instance.get_model()
    
    def handle(self):
        return self.tokenizer, self.model

# ComputeEmbeddings: Responsible for embedding computation
class ComputeEmbeddings:
    def __init__(self):
        self.tokenizer, self.model = LoadModelHandler().handle()
        self.device = Device.get_instance().get_device()

    def compute_embeddings(self, chunks):
        """
        Compute embeddings for the provided chunks of text.
        
        Args:
        - chunks (list): List of text chunks to compute embeddings for.
        
        Returns:
        - numpy array: Embeddings for the provided chunks.
        """
        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Pass inputs to the model
                outputs = self.model(**inputs, return_dict=True)
                
                # Compute the mean embedding and move to CPU
                chunk_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(chunk_embedding)
        
        return np.vstack(embeddings)