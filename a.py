import abc
from abc import ABC


def main():
    class A(abc.ABC):
        def __init__(self):
            self.__a = 1

        @abc.abstractmethod
        def abstract(self):
            print(self.__a)

    class B(A, ABC):
        def __init__(self):
            super(B, self).__init__()
            self.__b = 1

        def abstract(self):
            print(self.__b)

    print(dir(A()))


if __name__ == "__main__":
    main()
