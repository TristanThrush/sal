import dill
import os
from learner import Learner


class InnereseAndEnglish:
    def __init__(self, innerese, english):
        self.innerese = innerese
        self.english = english

    def __hash__(self):
        return hash(self.innerese)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.innerese == other.innerese
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '[ ' + self.innerese + ', ' + self.english + ' ]'


class Atom:
    def __init__(self, name, arguments=[]):
        self.name = name
        self.arguments = arguments
        self.sal = None

    def function(self, args):
        pass

    def innerese_and_english(self, args):
        pass


class Problems:
    def __init__(self):
        pass

    def names(self):
        return self.problems.keys()

    def choice(self, name):
        pass


class Atoms:
    def __init__(self):
        pass

    def names(self):
        return self.atoms.keys()

    def choice(self, name):
        return self.atoms[name]

    def connect(self, sal):
        for atom in self.atoms.values():
            atom.sal = sal


class SAL():
    def __init__(self):
        self.learner = Learner(self)
        self.atoms = None
        self.problems = None

    def learn(self, epochs):
        self.perform(epochs, optimize=True, help=True)

    def perform(self, epochs, optimize=False, help=False, cutoff=float('inf')):
        self.atoms.connect(self)
        self.optimize = optimize
        self.help = help
        self.operator_traces = []
        if self.optimize:
            self.learner.merge_module.train()
        else:
            self.learner.merge_module.eval()
        for epoch in range(epochs):
            for problem_name in self.problems.names():
                steps = 0
                self.operator_traces.append((problem_name, []))
                self.problem_state_and_goal, reward = self.problems.choice(
                    problem_name)
                while reward < 0 and steps < cutoff:
                    reward = self.step()
                    steps += 1
                if reward > 0:
                    print('\nSolved: ', problem_name)
                else:
                    print('\nCutt off on: ', problem_name)
                print('Epoch:', epoch)
                print()

    def step(self, forced_operator=None):
        innerese_problem_state_and_goal = list(map(
            lambda innerese_and_english: innerese_and_english.innerese,
            self.problem_state_and_goal))
        self.problem_state_and_goal, reward, operator =\
            self.learner.operate(
                innerese_problem_state_and_goal,
                forced_operator=forced_operator)
        print(
            operator.innerese_and_english().innerese + ',',
            end=' ',
            flush=True)
        if self.optimize:
            self.learner.optimize()
        return reward

    def save(self, directory):
        os.makedirs(directory)
        dill.dump(
            list(
                map(
                    lambda operator_trace: [
                        operator_trace[0],
                        list(
                            map(
                                lambda operator:
                                    operator.innerese_and_english().innerese,
                                operator_trace[1]))],
                    self.operator_traces)),
            open(
                directory + '/innerese_operator_traces.b',
                'wb'))
        self.learner.save(directory)

    def load(self, directory):
        self.learner.load(directory)
