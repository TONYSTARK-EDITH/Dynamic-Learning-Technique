class NoArgumentException(AttributeError):
    def __init__(self, msg):
        super(AttributeError, self).__init__(msg)


class InvalidMachineLearningModel(AssertionError):
    def __init__(self, msg):
        super(AssertionError, self).__init__(msg)
