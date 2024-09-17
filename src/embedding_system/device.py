import torch

class Device:
    _instance = None

    @staticmethod
    def get_instance():
        if Device._instance is None:
            Device._instance = Device()
        return Device._instance

    def __init__(self):
        if Device._instance is not None:
            raise Exception("Device is a singleton and has already been initialized!")
        self.__define_device()

    def __define_device(self):
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            print("Using CUDA for GPU acceleration")
        elif torch.backends.mps.is_available():
            self._device = torch.device('mps')
            print("Using MPS for Apple Silicon Mac")
        else:
            self._device = torch.device('cpu')
            print("Using CPU")
    
    def get_device(self):
        return self._device
