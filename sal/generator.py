import dill
import random


class OperatorGenerator:
    ARGUMENT_UPPER_LIMIT = 8
    off_limit_productions = set()
    temporary_off_limit_productions = set()

    @staticmethod
    def load_off_limit_productions(file):
        operator_traces = dill.load(open(file, 'rb'))
        for operator_trace in operator_traces:
            OperatorGenerator.off_limit_productions |= set(
                operator_trace[1])

    @staticmethod
    def generate(atoms, terminals):
        OperatorGenerator.current_arguments = 0
        operator = OperatorGenerator.generate_recursive(
            atoms, terminals)
        while operator is None or operator.innerese_and_english().innerese in\
            OperatorGenerator.off_limit_productions.union(
                OperatorGenerator.temporary_off_limit_productions):
            OperatorGenerator.current_arguments = 0
            operator = OperatorGenerator.generate_recursive(
                atoms, terminals)
        return operator

    @staticmethod
    def generate_recursive(atoms, terminals):
        symbol = random.choice(atoms)
        arguments = []
        new_name = symbol.name + ' ('
        for argument in symbol.arguments:
            OperatorGenerator.current_arguments += 1
            if OperatorGenerator.current_arguments >\
                    OperatorGenerator.ARGUMENT_UPPER_LIMIT:
                return None
            if argument == 'p':
                next_symbol = OperatorGenerator.generate_recursive(
                    atoms, terminals)
                if next_symbol is None:
                    return None
            if argument == 't':
                next_symbol = random.choice(terminals)
            if argument == 'tl':
                next_symbol = random.choice(list(filter(
                    lambda terminal: '(' not in terminal, terminals)))
            arguments.append(next_symbol)
            if next_symbol not in terminals:
                new_name += ' ' + next_symbol.name
            else:
                new_name += ' ' + next_symbol
        new_name += ' )'

        if new_name == symbol.name + ' ( )':
            new_name = symbol.name

        class Operator():
            def __init__(self, name):
                self.name = name

            def function(self):
                # Note that this is a different method than an atomic
                # operator's function() method.
                return symbol.function(arguments)

            def innerese_and_english(self):
                return symbol.innerese_and_english(arguments)

        new_operator = Operator(new_name)
        return new_operator
