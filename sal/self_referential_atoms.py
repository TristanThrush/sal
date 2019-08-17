from sal import InnereseAndEnglish, Atom
from generator import OperatorGenerator
import random


class Then(Atom):
    def __init__(self, name):
        super(Then, self).__init__(name, arguments=['p', 'p'])

    def function(self, args):
        self.sal.step(args[0])
        reward = self.sal.step(args[1])
        return self.sal.problem_state_and_goal, reward

    def innerese_and_english(self, args):
        return InnereseAndEnglish(
            '( then ' + args[0].innerese_and_english().innerese + ' '
            + args[1].innerese_and_english().innerese + ' )',
            args[0].innerese_and_english().english + ' then '
            + args[1].innerese_and_english().english)


class Forget(Atom):
    def __init__(self, name):
        super(Forget, self).__init__(name, arguments=['p'])

    def function(self, args):
        new_problem_state_and_goal = []
        for innerese_and_english in self.sal.problem_state_and_goal:
            if innerese_and_english.innerese != '( remember i ' + \
                    args[0].innerese_and_english().innerese + ' )':
                new_problem_state_and_goal.append(
                    innerese_and_english)
        return new_problem_state_and_goal, -0.1

    def innerese_and_english(self, args):
        return InnereseAndEnglish(
            '( forget ' + args[0].innerese_and_english().innerese + ' )',
            'forget ' + args[0].innerese_and_english().english)


class Remember(Atom):
    def __init__(self, name):
        super(Remember, self).__init__(name, arguments=['p'])

    def function(self, args):
        innerese_operator_traces =\
            list(map(lambda operator: operator.innerese_and_english().innerese,
                     self.sal.operator_traces[-1][1]))
        for innerese_operator in set(innerese_operator_traces):
            if args[0].innerese_and_english().innerese == innerese_operator:
                self.sal.problem_state_and_goal =\
                    self.sal.atoms.choice('forget').function(args)[0]
                self.sal.problem_state_and_goal.append(InnereseAndEnglish(
                    '( remember i ' + args[0].innerese_and_english().innerese
                    + ' )',
                    'i remember ' + args[0].innerese_and_english().english))
        return self.sal.problem_state_and_goal, -0.1

    def innerese_and_english(self, args):
        return InnereseAndEnglish(
            '( remember ' + args[0].innerese_and_english().innerese + ' )',
            'remember ' + args[0].innerese_and_english().english)


class Externalize(Atom):
    def __init__(self, name):
        super(Externalize, self).__init__(name)

    def function(self, args):
        print()
        for item in self.sal.problem_state_and_goal:
            print(item.english + '.', end=" ", flush=True)
        print()
        return self.sal.problem_state_and_goal, -0.1

    def innerese_and_english(self, args):
        return InnereseAndEnglish('externalize', 'externalize')


class Internalize(Atom):
    def __init__(self, name):
        super(Internalize, self).__init__(name)
        self.automatic_help_list = ['place oil jug on stool', 'pick oil jug']

    def function(self, args):
        if self.sal.help:
            help_english = None
            if self.sal.operator_traces[-1][0] == 'train':
                for operator in self.sal.operator_traces[-1][1]:
                    if operator.innerese_and_english().innerese ==\
                            'externalize':
                        help_english = random.choice(self.automatic_help_list)
        else:
            help_english = None
        # Try to generate an operator that corresponds to the english input.
        help_operator = None
        innerese_problem_state_and_goal = list(map(
            lambda innerese_and_english: innerese_and_english.innerese,
            self.sal.problem_state_and_goal))

        self.sal.learner.merge_module(innerese_problem_state_and_goal)
        terminals = list(self.sal.learner.merge_module.constituent_set)

        if help_english is not None:
            iter = 0
            beam = 5000
            OperatorGenerator.temporary_off_limit_productions = set()
            while iter < beam and help_operator is None:
                iter += 1
                operator = OperatorGenerator.generate(
                    list(map(lambda name: self.sal.atoms.choice(name),
                             self.sal.atoms.names())),
                    terminals)
                OperatorGenerator.temporary_off_limit_productions.add(
                    operator.innerese_and_english().innerese)
                if operator.innerese_and_english().english == help_english:
                    help_operator = operator
            OperatorGenerator.temporary_off_limit_productions = set()
        reward = -0.1
        if help_operator is not None:
            print('help: ', help_operator.innerese_and_english().innerese)
            reward = self.sal.step(help_operator)
        else:
            print('FAILED TO INTERNALIZE')
        return self.sal.problem_state_and_goal, reward

    def innerese_and_english(self, args):
        return InnereseAndEnglish('internalize', 'internalize')
