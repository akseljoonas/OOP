from src.game import Game


def main():
    game = Game(code_length=4, game_length=10)
    game.play()


if __name__ == '__main__':
    main()
