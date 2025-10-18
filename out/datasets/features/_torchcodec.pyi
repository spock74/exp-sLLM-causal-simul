from torchcodec.decoders import AudioDecoder as _AudioDecoder

class AudioDecoder(_AudioDecoder):
    def __getitem__(self, key: str): ...
