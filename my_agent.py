__author__ = "Luke Webb"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "weblu938@student.otago.ac.nz"

import random
from typing import List, Tuple, Any

import numpy as np
from itertools import product


def get_random_actions(self):
    """ Function for generating a random sequence of colours.
    """
    return np.random.choice(self.colours, size=self.code_length)


def get_initial_list(self):
    """ Generates and returns the list of all possible solutions as a list of tuples.
        This list is based on the initialized colour and code-length of the game.
    """
    return self.true_list.copy()


def initial_guess(self):
    """
        Based on the optimal guess by Donald Knuth, where contrary to what might seem intuitive,
        we guess in a pattern using only two colours.
            e.g. in a sequence of four colours we guess XXYY,
                where X is any colour and Y is a different colour

        See https://en.wikipedia.org/wiki/Mastermind_(board_game)

        :return: The best starting guess
    """
    guess = []
    c = 0
    for i in range(self.code_length):
        if i == (self.code_length//2):
            c += 1
        guess.append(self.colours[c])
    return guess


def update_list_zero(self, current_list, remove_sequence):
    """
        Used to remove all sequences of a particular colour. Only to be called when
        the last guess results in 0 in-place and 0 in-colour.

        :param  current_list: The current list of possibilities.

                remove_sequence: The last guess, where we want to remove all colours in the sequence.

        :return: The updated list, where all guesses with the given colours are removed.
    """
    colours_to_remove = set(remove_sequence)
    remove_list = []
    for colour in colours_to_remove:
        i = 0
        single_colour = []
        while i < self.code_length:
            single_colour.append(colour)
            i += 1
        remove_list.append(single_colour)
    new_set = set()
    for sequence in remove_list:
        for guess in current_list:
            if sequence[0] in guess:
                new_set.add(guess)

    return list(set.difference( set(current_list), new_set))


def update_list(current_list, sequence, score_of_sequence=tuple()):
    """
        Used to update the current list of possible guesses based on the last guess.
        Iterates over all guesses in the current_list and compares them to the input
        sequence as if it were the solution.

        :param  current_list: the current list of possibilities in this state of the game.

                sequence: the previous guess.

                score_of_sequence: a tuple containing the evaluation of the last guess as (in_place, in_colour).

        :return: The updated list, with all impossible guesses removed from the list of possibilities.
    """
    new_set = set()
    for guess in current_list:
        if evaluate_guess(guess, sequence) != score_of_sequence:
            new_set.add(guess)
    return list(set.difference(set(current_list), new_set))


def get_best_guess(current_list):
    return minimax(current_list)


def minimax(current_list):
    """ Classic implementation of the mini-max algoritm for mastermind

    """
    score_occurrences = {}
    # Iterate through each item in the list, as guess
    for guess in current_list:
        guess_score_occurrence = {}
        guess_min_score = tuple()
        # Iterate through each item in the list, as sub_guess,
        # Get the evaluation of the sub_guess as if guess were the goal.
        for sub_guess in current_list:
            score = evaluate_guess(sub_guess, guess)
            # Keep track of the min score for this loop.
            if score < guess_min_score:
                guess_min_score = score
            # Maintain the dictionary of scores and their occurrences.
            if score in guess_score_occurrence:
                guess_score_occurrence[score] += 1
            else:
                guess_score_occurrence[score] = 1
        # Retrieve the lowest score from the associated occurrence.
        guess_min_score_frequency = min(guess_score_occurrence.items(), key=lambda x: x[0])[1]
        # Add the min to the main dictionary of scores and occurrences.
        score_occurrences[guess] = guess_min_score_frequency
    # Get the guess with the most frequent and lowest score.
    min_score = max(score_occurrences, key=lambda x: x[1])
    return min_score


def minimax_lazy(current_list):
    """WIP"""
    score_occurrences = {}
    min_score_guess = []
    min_score_guess = tuple()
    # Iterate through each item in the list, as guess
    for guess in current_list:
        guess_score_occurrence = {}
        guess_min_score = tuple()
        # Iterate through each item in the list, as sub_guess,
        # Get the evaluation of the sub_guess as if guess were the goal.
        for sub_guess in current_list:
            score = evaluate_guess(sub_guess, guess)
            # Keep track of the min score for this loop.
            if score < guess_min_score:
                guess_min_score = score
            # Maintain the dictionary of scores and their occurrences.
            if score in guess_score_occurrence:
                guess_score_occurrence[score] += 1
            else:
                guess_score_occurrence[score] = 1
        # Retrieve the lowest score from the associated occurrence.
        guess_min_score_frequency = min(guess_score_occurrence.items(), key=lambda x: x[0])[1]

        # Add the min to the main dictionary of scores and occurrences.
        # If the frequency of a given min score is less than the one that already occurs more often,
        # then don't add this to our dictionary.
        if not guess_min_score_frequency < max(score_occurrences):
            score_occurrences[guess] = guess_min_score_frequency
    # Get the guess with the most frequent and lowest score.
    min_score = max(score_occurrences, key=lambda x: x[1])
    return min_score


def evaluate_guess(guess, target):
    """ Evaluates a guess against a target, borrowed from Mastermind.py

          :param guess: a R x C numpy array of valid colour characters that constitutes a guess

                 target: a R x C numpy array of valid colour characters that constitutes target solution


          :return: a tuple of 4 vectors:

                   R-dimensional vector that gives the number of correct colours in place in each row of the
                                 guess against the target

                   R-dimensional vector that gives the number of correct colours out of place in each row of the
                                 guess against the target

                   C-dimensional vector that gives the number of correct colours in place in each column of the
                                 guess against the target

                   C-dimensional vector that gives the number of correct colours out of place in each column of the
                                 guess against the target

          """

    guess = np.reshape(guess, (-1))
    target = np.reshape(target, (-1))

    I = np.where(guess == target)[0]
    in_place = len(I)
    I = np.where(guess != target)[0]
    state = np.zeros(np.shape(target))

    in_colour = 0
    for i in I:
        a = target[i]
        for j in I:
            if state[j] != 0:
                continue

            b = guess[j]

            if a == b:
                in_colour += 1
                state[j] = -1
                break

    return in_place, in_colour


class MastermindAgent():
    """
             A class that encapsulates the code dictating the
             behaviour of the agent playing the game of Mastermind.

             ...

             Attributes
             ----------
             code_length: int
                 the length of the code to guess
             colours : list of char
                 a list of colours represented as characters
             num_guesses : int
                 the max. number of guesses per game

             Methods
             -------
             AgentFunction(percepts)
                 Returns the next guess of the colours on the board
             """

    def __init__(self, code_length, colours, num_guesses):
        """
      :param code_length: the length of the code to guess
      :param colours: list of letter representing colours used to play
      :param num_guesses: the max. number of guesses per game
      """

        self.code_length = code_length
        self.colours = colours
        self.num_guesses = num_guesses
        self.true_list = list(product(list(self.colours), repeat=self.code_length))


    def AgentFunction(self, percepts):
        """Returns the next board guess given state of the game in percepts

            :param percepts: a tuple of four items: guess_counter, last_guess, in_place, in_colour

                     , where

                     guess_counter - is an integer indicating how many guesses have been made, starting with 0 for
                                     initial guess;

                     last_guess - is a num_rows x num_cols structure with the copy of the previous guess

                     in_place - is the number of character in the last guess of correct colour and position

                     in_colour - is the number of characters in the last guess of correct colour but not in the
                                 correct position

            :return: list of chars - a list of code_length chars constituting the next guess
            """
        global possibles
        # Extract different parts of percepts.
        guess_counter, last_guess, in_place, in_colour = percepts

        score = in_place + in_colour
        score_tuple = (in_place, in_colour)

        # Check the state of the game, creates our list of combinations.
        if guess_counter == 0:
            possibles = get_initial_list(self)
            actions = initial_guess(self)
            possibles.remove(tuple(actions))

            print(len(possibles))
            return actions

        # If the guess is terrible, remove those colours from the guess table.
        elif score == 0:
            possibles = update_list_zero(self, possibles, last_guess)
            print(len(possibles))
        else:
            possibles = update_list(possibles, last_guess, score_tuple)
            print(len(possibles))

        actions = list(get_best_guess(possibles))
        possibles.remove(tuple(actions))
        return actions
