from copy import deepcopy
import random


class CodeMaker:
    '''
        CodeMaker class is used to create a random secret code that consists
        of self._colors and is of the length size

        Args:
            size: int, indicating the length of the code

        Methods:
            output_code(): returns a deepcopy of the generated code
            create_code(): creates a random code and saves it to self.__code
            output_size(): returns a deepcopy of self.__size which\
            idicates the real length of the code
        '''
    def __init__(self, size: int = 4):
        self._colors = ['W', 'K', 'Y', 'G', 'R', 'B']
        self.__size = size
        self.__code = []

    def _output_code(self):
        '''
        returns:
            deepcopy of the self.__code
        '''
        return deepcopy(self.__code)

    def _create_code(self):
        '''
        create_code() method generates a random code from self._colors of the\
        length self.__size and saves it into self.__code.
        '''
        # creates a random code consisting of self._colors
        # of the length self.__size
        self.__code = random.sample(self._colors, self.__size)

    def _output_size(self):
        '''
        returns:
            deepcopy of the self.__size
        '''
        return deepcopy(self.__size)
