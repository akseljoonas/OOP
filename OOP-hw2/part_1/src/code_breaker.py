from copy import deepcopy
import sys


class CodeBreaker:
    '''
    CodeBreaker is used to get an user input for the
    code of the correct size and content.

    methods:
            make_guess(): method used get a user input and save it to\
            self._user_guess
            output_size(): method to output the size of the produced code
            output_user_guess(): method to return a deepcopy of the\
            user's guess.
    '''
    def __init__(self, size: int = 4):
        self._colors = ['W', 'K', 'Y', 'G', 'R', 'B']
        self.__size = size
        self._user_guess = []

    def make_guess(self):
        '''
        make_guess() method is used to get a user input\
        for the code of the correct size and content.
        '''
        user_input = input("Please enter four letters: ")
        sys.stdin.flush()  # Flush the input buffer

        # Remove extra spaces
        letters = []

        # Append the individual characters to the array
        for char in user_input:
            letters.append(char.upper())

        # checking if the provided input is of the correct size
        # and consists of allowed letters
        length_correct = self._check_input_length(letters)
        correctness_correct = self._check_input_correctness(letters)
        if length_correct is False or correctness_correct is False:
            if length_correct is False:
                print("Please enter exactly four letters!")
            else:
                print("Use only the allowed letters: 'W' 'K' 'Y' 'G' 'R' 'B'")
            self.make_guess()
        else:
            self._user_guess = letters

    def _output_size(self):
        '''
        returns:
            deepcopy of the self.__size.
        '''
        return deepcopy(self.__size)

    def _output_user_guess(self):
        '''
        returns:
            deepcopy of the self._user_guess.
        '''
        return deepcopy(self._user_guess)

    def _check_input_length(self, letters):
        if len(letters) == self.__size:
            return True
        else:
            return False

    def _check_input_correctness(self, letters):
        for letter in letters:
            if letter not in self._colors:
                return False
        return True
