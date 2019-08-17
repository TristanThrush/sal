from sal import Atom, Atoms, Problems, InnereseAndEnglish
from self_referential_atoms import Forget, Remember, Externalize, Internalize,\
    Then
import copy
import time
import os
from random import randint
import atexit


class Simulator:
    def __init__(self, state_dict, goal, simulator_visual_file):
        self.state_dict = state_dict
        self.goal = goal
        self.types = ['movables', 'immovables']
        self.left_gripper = InnereseAndEnglish('( left gripper )',
                                               'left gripper')
        self.right_gripper = InnereseAndEnglish('( right gripper )',
                                                'right gripper')
        self.simulator_visual_file = simulator_visual_file
        self.save_visual()

    def innerese_and_english(self):
        innerese_and_english = []
        for item in self.state_dict:
            for resting in self.state_dict[item]:
                if item not in self.types:
                    innerese_and_english.append(InnereseAndEnglish(
                        '( on ' + resting.innerese + ' ' + item.innerese
                        + ' )',
                        resting.english + ' is on ' + item.english))
            if len(self.state_dict[item]) == 0 and item not in\
                self.state_dict['movables'] and item != self.left_gripper\
                    and item != self.right_gripper and item not in\
                    self.types:
                innerese_and_english.append(InnereseAndEnglish(
                    '( on nothing ' + item.innerese + ' )', 'nothing is on '
                    + item.innerese))
        innerese_and_english.append(self.goal[1])
        return innerese_and_english

    def save_visual(self):
        def string_form(item):
            return item.english[:5] + ' '*(5-len(item.english[:5]))
        vis = ''
        for immovable in self.state_dict['immovables']:
            reverse_sub_vis =\
                ['|' + '-'*6*len(self.state_dict['movables']) + '|']\
                + ['']*len(self.state_dict['movables'])
            items_to_draw = self.state_dict[immovable]
            for item in items_to_draw:
                spaces = False
                for iter in range(len(self.state_dict['movables'])):
                    if not spaces:
                        reverse_sub_vis[iter+1] += string_form(item) + ' '
                        if len(self.state_dict[item]) != 0:
                            # Movable objects can only fit one thing on top.
                            item = self.state_dict[item][0]
                        else:
                            spaces = True
                    else:
                        reverse_sub_vis[iter+1] += ' '*6
            vis += '\n'.join(reversed(reverse_sub_vis)) + '\n'
        if len(self.state_dict[self.left_gripper]) != 0:
            vis += string_form(self.state_dict[self.left_gripper][0]) + ' '
        else:
            vis += ' '*6
        if len(self.state_dict[self.right_gripper]) != 0:
            vis += string_form(self.state_dict[self.right_gripper][0]) + ' '
        else:
            vis += ' '*6
        vis += '\n[_]   [_]'
        visual_save_location = os.path.dirname(os.path.abspath(__file__)) +\
            '/../' + self.simulator_visual_file
        open(visual_save_location, 'w+').write(vis)

    def reward(self):
        if self.goal[0][0] in self.state_dict[self.goal[0][1]]:
            return 1
        return -0.1

    def find_and_remove(self, x):
        for item in self.state_dict:
            if item not in self.types:
                for resting in self.state_dict[item]:
                    if resting == x:
                        self.state_dict[item].remove(x)

    def find_top(self, x):
        if len(self.state_dict[x]) == 0:
            return x
        return self.find_top(self.state_dict[x][0])

    def pick(self, x):
        if x in self.state_dict['movables'] and len(self.state_dict[x]) == 0:
            if len(self.state_dict[self.left_gripper]) == 0:
                self.find_and_remove(x)
                self.state_dict[self.left_gripper].append(x)
            elif len(self.state_dict[self.right_gripper]) == 0:
                self.find_and_remove(x)
                self.state_dict[self.right_gripper].append(x)
        self.save_visual()
        return self.innerese_and_english(), self.reward()

    def place(self, x, y):
        if x in self.state_dict[self.left_gripper] +\
                self.state_dict[self.right_gripper] and y not in\
                self.state_dict[self.left_gripper] +\
                self.state_dict[self.right_gripper]:
            if y in self.state_dict['immovables']:
                self.find_and_remove(x)
                self.state_dict[y].append(x)
            if y in self.state_dict['movables']:
                if len(self.state_dict[y]) == 0:
                    self.find_and_remove(x)
                    self.state_dict[y].append(x)
        self.save_visual()
        return self.innerese_and_english(), self.reward()


class SimulatorVisualizer:

    @staticmethod
    def visualize(simulator_visual_file):
        vis = None
        while True:
            time.sleep(0.1)
            if vis is not None:
                print('\033[K\033[A'*(vis.count('\n')+2))
            vis = open(os.path.dirname(
                os.path.abspath(__file__)) + '/../'
                    + simulator_visual_file, 'r').read()
            print(vis)


class Pick(Atom):
    def __init__(self, name):
        super(Pick, self).__init__(name, arguments=['t'])

    def function(self, args):
        _, _, recovered_english = self.sal.learner.merge_module.parse(
            args[0])
        problem_state_and_goal, reward =\
            self.sal.problems.active_problem.pick(InnereseAndEnglish(
                args[0], recovered_english))
        # Don't erase info about the problem that is not about the state of the
        # simulator.
        for production in self.sal.problem_state_and_goal:
            if production.innerese.\
                    startswith('( remember i'):
                problem_state_and_goal.append(production)
        return problem_state_and_goal, reward

    def innerese_and_english(self, args):
        innerese = '( ' + self.name + ' ' + args[0] + ' )'
        _, _, recovered_english = self.sal.learner.merge_module.parse(
            innerese)
        return InnereseAndEnglish(innerese, recovered_english)


class Place(Atom):
    def __init__(self, name):
        super(Place, self).__init__(name, arguments=['t', 't'])

    def function(self, args):
        _, _, recovered_english_0 = self.sal.learner.merge_module.parse(
            args[0])
        _, _, recovered_english_1 = self.sal.learner.merge_module.parse(
            args[1])
        problem_state_and_goal, reward =\
            self.sal.problems.active_problem.place(InnereseAndEnglish(
                args[0], recovered_english_0), InnereseAndEnglish(
                args[1], recovered_english_1))
        # Don't erase info about the problem that is not about the state of the
        # simulator.
        for production in self.sal.problem_state_and_goal:
            if production.innerese.\
                    startswith('( remember i'):
                problem_state_and_goal.append(production)
        return problem_state_and_goal, reward

    def innerese_and_english(self, args):
        innerese = '( ' + self.name + ' ( on ' + args[0] + ' '\
            + args[1] + ' ) )'
        _, _, recovered_english = self.sal.learner.merge_module.parse(
            innerese)
        return InnereseAndEnglish(innerese, recovered_english)


class ManipulationProblems(Problems):
    def __init__(self):
        super(Problems, self).__init__()

        self.simulator_visual_file = '.' + str(randint(10000, 99999)) + '.txt'
        print('Simulator visual file:', self.simulator_visual_file)
        atexit.register(lambda: os.remove(
            os.path.dirname(os.path.abspath(__file__))
            + '/../' + self.simulator_visual_file))

        tire = InnereseAndEnglish('tire', 'tire')
        oil_jug = InnereseAndEnglish('( oil jug )', 'oil jug')
        workbench = InnereseAndEnglish('workbench', 'workbench')
        stool = InnereseAndEnglish('stool', 'stool')
        left_gripper = InnereseAndEnglish('( left gripper )', 'left gripper')
        right_gripper = InnereseAndEnglish('( right gripper )',
                                           'right gripper')
        self.problems = {

            'train': [{
                'movables': [tire, oil_jug],
                tire: [],
                oil_jug: [],
                'immovables': [workbench, stool],
                workbench: [oil_jug],
                stool: [tire],
                left_gripper: [],
                right_gripper: []},
                [[oil_jug, stool], InnereseAndEnglish(
                    '( want i ( on ( oil jug ) stool ) )',
                    'i want oil jug on stool')]]}

    def choice(self, name):
        self.active_problem = Simulator(copy.deepcopy(self.problems[name][0]),
                                        copy.deepcopy(self.problems[name][1]),
                                        self.simulator_visual_file)
        return (self.active_problem.innerese_and_english(),
                self.active_problem.reward())


class ManipulationAtoms(Atoms):
    def __init__(self):
        self.atoms = {
            'place': Place('place'),
            'pick': Pick('pick'),
            'externalize': Externalize('externalize'),
            'internalize': Internalize('internalize'),
            'forget': Forget('forget'),
            'remember': Remember('remember'),
            'then': Then('then')}
