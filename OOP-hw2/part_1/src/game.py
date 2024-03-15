from src.game_master import GameMaster
from src.code_maker import CodeMaker
from src.code_breaker import CodeBreaker


class Game():
    '''
    Game class is user to create an instance of the game by specifying\
    code_length and game_length in the initialization.
    '''
    def __init__(self, code_length: int = 4, game_length: int = 10):
        self.code_length = code_length
        self.game_length = game_length

    def play(self) -> None:
        '''
        play() method initialises the game cycle, by using CodeMaker to
        create a code of the size code_length, and lets the user guess
        the code for game_length number of rounds.

        Args:
            code_length: (int) number of letters in a code\n
            game_length: (int) number of rounds in a game
        '''
        # create a secret code of the length self.code_length
        code_maker = CodeMaker(size=self.code_length)
        code_maker._create_code()
        correct_code = code_maker._output_code()
        # create code_braker (for getting user input) and game_master
        # (to evaluate user input)
        code_breaker = CodeBreaker(size=self.code_length)
        game_master = GameMaster()
        game_master._save_correct_code(correct_code)
        print(f"we are looking for a code of the length {self.code_length}.")
        print("The code consists of these letters: 'W' 'K' 'Y' 'G' 'R' 'B'")
        print(f"You have {self.game_length} guesses!")
        # play the game for <game_length> number of iterations
        for i in range(0, self.game_length):
            print(f"ROUND {i+1}")
            code_breaker.make_guess()  # get the input from the user
            # save the user guess into the game master
            user_guess = code_breaker._output_user_guess()
            game_master._save_user_input(user_guess)
            # give feedback to the user
            feedback = game_master._give_feedback()

            # if the guess was correct end the game
            if feedback is True:
                print(f"Correct! The right code is indeed {correct_code}!")
                return
            # if the guess was not correct and the game is still going,
            # give user feedback
            elif i < self.game_length - 1:
                print(f"You have {feedback[0]} correct at the right position.")
                print(f"You have {feedback[1]} correct at the wrong position.")
            # if the user has not guessed correctly and no iterations left,
            # end the game and give the user the right code.
            else:
                print(f"Game Over! The right code was {correct_code}!")
                return
