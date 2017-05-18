class Config:
    __values = {}

    def __init__(self, flags):
        self.__values = flags

    def fetch(self, index=None):
        if index is not None:
            value = getattr(self.__values, index, None)
        else:
            value = getattr(self, '__values', {})

        return value
