from abc import ABC, abstractmethod

class F0Predictor(ABC):
    @abstractmethod
    def compute_f0(self, wav, p_len):
        """
        input: wav:[signal_length]
        p_len:int
        output: f0:[signal_length//hop_length]
        """
        pass

    @abstractmethod
    def compute_f0_uv(self, wav, p_len):
        """
        input: wav:[signal_length]
        p_len:int
        output: f0:[signal_length//hop_length],uv:[signal_length//hop_length]
        """
        pass
