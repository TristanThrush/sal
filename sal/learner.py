import random
import gc
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import os
import dill
from generator import OperatorGenerator
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


Transition = namedtuple('Transition',
                        ('innerese_mental_state',
                         'next_innerese_problem_state_and_goal', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EmbeddingsNavigator():
    def __init__(self, embeddings_path):
        self.words_to_indices, self.embeddings =\
            self.load_embeddings(embeddings_path)
        self.vocab_size, self.embedding_dim = self.embeddings.size()

    def load_embeddings(self, embeddings_path):
        lines = open(embeddings_path).readlines()
        words_to_indices = {}
        embeddings = []
        index = 0
        for line in lines:
            line_list = line.split()
            words_to_indices[line_list[0]] = index
            embeddings.append(list(map(lambda line_list_number:
                                       float(line_list_number),
                                       line_list[1:])))
            index += 1
        if torch.cuda.is_available():
            # Load into CPU first for speed.
            embeddings = torch.FloatTensor(embeddings).cuda()
        else:
            embeddings = torch.FloatTensor(embeddings)
        return words_to_indices, embeddings


class MergeModule(nn.Module):
    def __init__(self, lstm_hidden_state_size, embeddings_path):
        super(MergeModule, self).__init__()
        self.tokenizer = EmbeddingsNavigator(embeddings_path)
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.word_embeddings = nn.Embedding(self.tokenizer.vocab_size,
                                            self.tokenizer.embedding_dim)
        self.word_embeddings.weight.data.copy_(self.tokenizer.embeddings)
        self.word_embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(
            self.tokenizer.embedding_dim,
            self.lstm_hidden_state_size,
            bidirectional=True)
        self.rescale = nn.Linear(self.lstm_hidden_state_size * 2,
                                 self.tokenizer.embedding_dim)
        self.memory = {}

    def init_hidden(self):
        self.hidden = (Variable(FloatTensor(
                            [[[0] * self.lstm_hidden_state_size] * 1] * 2)),
                       Variable(FloatTensor(
                            [[[0] * self.lstm_hidden_state_size] * 1] * 2)))

    def merge(self, constituents):
        if len(constituents) == 1:
            return constituents[0]
        self.init_hidden()
        x = torch.cat(constituents)
        x, self.hidden = self.lstm(x.view(len(constituents), 1, -1),
                                   self.hidden)
        x = F.max_pool1d(
            torch.t(x.view(len(constituents), -1)).unsqueeze(0),
            kernel_size=len(constituents))
        return self.rescale(x.squeeze().unsqueeze(0))

    def parse(self, innerese):
        if innerese in self.memory:
            return self.memory[innerese]
        constituent_set = set()
        constituent_vectors = []
        open_parens = 0
        closed_parens = 0
        constituent = ''
        length_counter = len(innerese)
        recovered_english = []
        for character in innerese:
            length_counter -= 1
            constituent += character
            if character == '(':
                open_parens += 1
            if character == ')':
                closed_parens += 1
            if (character == ' ' or length_counter == 0)\
                    and open_parens == closed_parens:
                if character == ' ':
                    constituent = constituent[:-1]
                if '(' in constituent:
                    constituent_vector, sub_constituent_set,\
                        recovered_sub_english = self.parse(constituent[2:-2])
                    constituent_vectors.append(constituent_vector)
                    constituent_set |= sub_constituent_set
                    recovered_english.append(recovered_sub_english)
                else:
                    constituent_index = Variable(LongTensor(
                        [self.tokenizer.words_to_indices[constituent]]))
                    constituent_vectors.append(
                        self.word_embeddings(constituent_index))
                    recovered_english.append(constituent)
                constituent_set.add(constituent)
                constituent = ''
        if len(recovered_english) == 3:
            recovered_english = recovered_english[1] + ' ' +\
                recovered_english[0] + ' ' + recovered_english[2]
        else:
            recovered_english = ' '.join(recovered_english)
        self.memory[innerese] = (self.merge(constituent_vectors),
                                 constituent_set, recovered_english)
        return self.parse(innerese)

    def forward(self, state):
        constituent_vector, self.constituent_set, self.recovered_english =\
            self.parse('( ' + ' '.join(state) + ' )')
        return constituent_vector.squeeze()[:1]


class Learner:
    def __init__(self, sal, target=False):

        self.sal = sal

        self.EMBEDDINGS_PATH = os.path.dirname(os.path.abspath(__file__)) + \
            '/glove.6B.100d.txt'
        self.TARGET_UPDATE = 10
        self.GAMMA = 0.99
        self.G = 500
        self.GE = 2
        self.LR = 0.0003
        self.BATCH_SIZE = 20
        self.REPLAY_MEMORY_SIZE = 200
        self.LSTM_HIDDEN_STATE_SIZE = 100

        self.target_counter = 0

        self.merge_module = MergeModule(self.LSTM_HIDDEN_STATE_SIZE,
                                        self.EMBEDDINGS_PATH)
        if torch.cuda.is_available():
            self.merge_module.cuda()

        self.optimizer = optim.Adam(filter(
            lambda parameter: parameter.requires_grad,
            list(self.merge_module.parameters())), lr=self.LR)

        self.replay_memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)

        if not target:
            self.target = Learner(sal, target=True)
            self.target.merge_module.eval()

    def generate_and_choose_operator(self, innerese_problem_state_and_goal,
                                     beam, forced_operator=None,
                                     return_generations=False):

        self.merge_module(innerese_problem_state_and_goal)
        terminals = list(self.merge_module.constituent_set)

        if forced_operator is None:
            OperatorGenerator.temporary_off_limit_productions = set()
            operator_and_innerese_mental_state_list = []
            for i in range(beam):
                operator = OperatorGenerator.generate(
                    list(map(lambda name: self.sal.atoms.choice(name),
                             self.sal.atoms.names())),
                    terminals)
                OperatorGenerator.temporary_off_limit_productions.add(
                    operator.innerese_and_english().innerese)
                operator_and_innerese_mental_state_list.append((
                    operator,
                    innerese_problem_state_and_goal
                    + ['( will i ' + operator.innerese_and_english().innerese
                       + ' )']))
            sorted_generations = sorted(map(
                lambda operator_and_innerese_mental_state:
                (self.merge_module(operator_and_innerese_mental_state[1]),
                 operator_and_innerese_mental_state[0]),
                    operator_and_innerese_mental_state_list),
                key=lambda item: item[0])
            OperatorGenerator.temporary_off_limit_productions = set()
            if return_generations:
                return sorted_generations
            return sorted_generations[-1]
        else:
            return (self.merge_module(
                innerese_problem_state_and_goal + ['( will i '
                                                   + forced_operator.
                                                   innerese_and_english().
                                                   innerese + ' )']),
                    forced_operator)

    def operate(self, innerese_problem_state_and_goal, forced_operator=None):
        value_approximation, operator = self.generate_and_choose_operator(
                innerese_problem_state_and_goal, random.choice(
                    (self.GE, self.GE, self.G)), forced_operator)
        next_innerese_problem_state_and_goal, reward = operator.function()
        self.sal.operator_traces[-1][1].append(operator)
        self.replay_memory.push(
            innerese_problem_state_and_goal + ['( will i '
                                               + operator.
                                               innerese_and_english().innerese
                                               + ' )'],
            list(map(lambda innerese_and_english: innerese_and_english.
                     innerese, next_innerese_problem_state_and_goal)),
            FloatTensor([reward]))
        return next_innerese_problem_state_and_goal, reward, operator

    def optimize(self):

        # Uptate target network, if it is time.
        self.target_counter += 1
        if self.target_counter == self.TARGET_UPDATE:
            self.target.merge_module.load_state_dict(
                self.merge_module.state_dict())
            self.target_counter = 0

        # Get and transpose the batch.
        transitions = self.replay_memory.sample(
            min(self.BATCH_SIZE, len(self.replay_memory)))
        batch = Transition(*zip(*transitions))

        # Get policy network value estimates for the states.
        innerese_mental_state_values = torch.cat(tuple(map(
            self.merge_module, batch.innerese_mental_state)))

        # Get target network value estimates for the next states.
        next_innerese_mental_state_values = torch.cat(
            tuple(
                map(
                    lambda next_innerese_problem_state_and_goal:
                    self.target.generate_and_choose_operator(
                        next_innerese_problem_state_and_goal,
                        self.G)[0],
                    batch.next_innerese_problem_state_and_goal)))

        # Compute the expected values.
        with torch.no_grad():
            reward_batch = Variable(torch.cat(batch.reward))
            expected_innerese_mental_state_values = (
                next_innerese_mental_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss.
        loss = F.smooth_l1_loss(
            innerese_mental_state_values,
            expected_innerese_mental_state_values)

        # Optimize the model.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Free memory
        self.merge_module.memory = {}
        self.target.merge_module.memory = {}
        gc.collect()

    def save(self, directory):
        temp_memory = copy.deepcopy(self.replay_memory.memory)
        self.replay_memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        [self.replay_memory.push(transition[0], transition[1],
                                 transition[2].type(torch.FloatTensor))
         for transition in temp_memory]
        dill.dump(self.replay_memory, open(
            directory + '/replay_memory.b', 'wb'))
        torch.save(self.merge_module.state_dict(),
                   directory + '/policy_merge_module.pt')
        torch.save(self.target.merge_module.state_dict(),
                   directory + '/target_merge_module.pt')

    def load(self, directory):
        self.replay_memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        [self.replay_memory.push(transition[0], transition[1],
                                 transition[2].type(FloatTensor))
         for transition in dill.load(open(
            directory + '/replay_memory.b', 'rb')).memory]
        if torch.cuda.is_available():
            self.merge_module.load_state_dict(
                    torch.load(directory + '/policy_merge_module.pt'))
            self.target.merge_module.load_state_dict(
                    torch.load(directory + '/target_merge_module.pt'))
        else:
            self.merge_module.load_state_dict(
                    torch.load(directory + '/policy_merge_module.pt',
                               map_location=torch.device('cpu')))
            self.target.merge_module.load_state_dict(
                    torch.load(directory + '/target_merge_module.pt',
                               map_location=torch.device('cpu')))

        # Free memory
        self.merge_module.memory = {}
        self.target.merge_module.memory = {}
        gc.collect()
