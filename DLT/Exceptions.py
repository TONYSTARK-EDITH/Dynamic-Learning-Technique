class NoArgumentException(AttributeError):
    def __init__(self, msg):
        super(AttributeError, self).__init__(msg)
