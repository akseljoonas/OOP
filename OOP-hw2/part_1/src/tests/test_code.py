import unittest
import sys
import os
sys.path.append(os.getcwd() + "/part_1/src/")
from code_maker import CodeMaker as CM
from code_breaker import CodeBreaker as CB
from game_master import GameMaster as GM
from unittest.mock import patch


class TestCodeMaker(unittest.TestCase):
    def test_code_exists(self):
        cm = CM()
        cm._create_code()
        self.assertIsNotNone(cm._output_code)

    def test_code_length(self):
        cm = CM()
        cm._create_code()
        self.assertEqual(cm._output_size(), len(cm._output_code()))

    def test_code_colors(self):
        cm = CM()
        cm._create_code()
        code = cm._output_code()
        colors = cm._colors
        for i in range(0, cm._output_size()):
            self.assertIn(code[i], colors)


class TestInputGetter(unittest.TestCase):
    @patch('builtins.input', side_effect=['WKYG'])
    def test_guess_registration(self, mock_input):
        test_instance = CB()
        test_instance.make_guess()
        expected_output = ['W', 'K', 'Y', 'G']
        self.assertEqual(test_instance._user_guess, expected_output)

    def test_input_length_tester(self):
        cb = CB()
        input = ['W', 'K', 'Y', 'G']
        self.assertTrue(cb._check_input_length(input))

    def test_right_colors(self):
        cb = CB()
        input = ['W', 'K', 'Y', 'G']
        self.assertTrue(cb._check_input_correctness(input))


class GameMaster(unittest.TestCase):
    def test_correct_code_saving(self):
        gm = GM()
        correct_code = ['W', 'K', 'Y', 'G']
        gm._save_correct_code(correct_code)
        self.assertEqual(gm._correct_code, correct_code)

    def test_right_position_output(self):
        gm = GM()
        correct_code = ['W', 'K', 'Y', 'G']
        user_code = ['W', 'Y', 'K', 'G']
        gm._save_user_input(user_code)
        gm._save_correct_code(correct_code)
        self.assertEqual(gm.__output_correct_position_num(), 2)
        return

    def test_wrong_position_output(self):
        gm = GM()
        correct_code = ['W', 'K', 'Y', 'G']
        user_code = ['W', 'Y', 'K', 'G']
        gm._save_user_input(user_code)
        gm._save_correct_code(correct_code)
        gm.__output_correct_position_num()
        self.assertEqual(gm.__output_wrong_position_num(), 2)

    def test_feedback(self):
        gm = GM()
        correct_code = ['W', 'K', 'Y', 'G']
        user_code = ['W', 'Y', 'K', 'G']
        gm._save_user_input(user_code)
        gm._save_correct_code(correct_code)
        feedback = gm._give_feedback()
        correct_feedback = [2, 2]
        self.assertEqual(feedback, correct_feedback)


if __name__ == '__main__':
    unittest.main()
