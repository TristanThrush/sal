import os


def exact_p_value(independents, observation):
    def all_binary_combinations(independents):
        probabilities = []
        if len(independents) == 1:
            probabilities.append((independents[0][0], [independents[0][1]]))
            probabilities.append(
                ((1 - independents[0][0]), ['not ' + independents[0][1]]))
        else:
            for probability in all_binary_combinations(independents[1:]):
                probabilities.append(
                    (independents[0][0] * probability[0],
                        [independents[0][1]] + probability[1]))
                probabilities.append(
                    ((1 - independents[0][0]) * probability[0],
                     ['not ' + independents[0][1]] + probability[1]))
        return probabilities
    probabilities = all_binary_combinations(independents)
    p = 0
    for probability in probabilities:
        if probability[1] == observation:
            extreme = probability[0]
    for probability in probabilities:
        if probability[0] <= extreme:
            p += probability[0]
    return p


def test_knowledge(s):
    import sal
    import copy
    for word_pair in (('grab', 'set'), ('pluck', 'rest'), ('grip', 'put'),
                      ('get', 'leave'), ('acquire', 'deposit'),
                      ('grasp', 'lay'), ('clasp', 'position'),
                      ('clutch', 'situate'), ('grapple', 'settle'),
                      ('take', 'sit'), ('guiltiest', 'phishers'),
                      ('98-93', 'kranhold'), ('swineflu', 'fanck'),
                      ('toner', 'assizes'), ('titanosaur', 'märjamaa'),
                      ('archeparchy', 'grella'), ('bacteroidetes', 'cowered'),
                      ('maritza', 'stylinski'), ('siniora', 'maurycy'),
                      ('scrophularia', 'attests')):
        s.atoms.atoms['pick'].name = word_pair[0]
        s.atoms.atoms['place'].name = word_pair[1]
        s.perform(1, cutoff=2)
    s.atoms.atoms['pick'].name = 'pick'
    s.atoms.atoms['place'].name = 'place'
    replacement = 'oil'
    for word in ('petroleum', 'gasoline', 'petrol', 'fuel',
                 'lubricant', 'grease', 'lubrication', 'kerosene', 'diesel',
                 'napalm', 'ivic', 'showed', 'murigande', 'chelios',
                 'aricie', 'ligule', 'oom', 'fedorchenko', 'haugesund',
                 'compilers'):
        original_state_dict = copy.deepcopy(s.problems.problems['train'][0])
        original_goal = copy.deepcopy(s.problems.problems['train'][1])
        for item in original_state_dict:
            if isinstance(item, sal.InnereseAndEnglish):
                if replacement in item.innerese or replacement in item.english:
                    value = original_state_dict[item]
                    del s.problems.problems['train'][0][item]
                    new_item = sal.InnereseAndEnglish('', '')
                    new_item.innerese = item.innerese.replace(replacement,
                                                              word)
                    new_item.english = item.english.replace(replacement, word)
                    s.problems.problems['train'][0][new_item] = value
        for item in s.problems.problems['train'][0]:
            new_values = []
            for value in s.problems.problems['train'][0][item]:
                new_values.append(sal.InnereseAndEnglish(
                    value.innerese.replace(replacement, word),
                    value.english.replace(replacement, word)))
            s.problems.problems['train'][0][item] = new_values
        s.problems.problems['train'][1][0][0] = sal.InnereseAndEnglish(
            s.problems.problems['train'][1][0][0].innerese.replace(
                replacement, word),
            s.problems.problems['train'][1][0][0].english.replace(replacement,
                                                                  word))
        s.problems.problems['train'][1][0][1] = sal.InnereseAndEnglish(
            s.problems.problems['train'][1][0][1].innerese.replace(
                replacement, word),
            s.problems.problems['train'][1][0][1].english.replace(replacement,
                                                                  word))
        s.problems.problems['train'][1][1] = sal.InnereseAndEnglish(
            s.problems.problems['train'][1][1].innerese.replace(replacement,
                                                                word),
            s.problems.problems['train'][1][1].english.replace(replacement,
                                                               word))
        s.perform(1, cutoff=2)
        s.problems.problems['train'][0] = original_state_dict
        s.problems.problems['train'][1] = original_goal


def latexable_innerese_operator_traces(dir):
    import dill
    traces = dill.load(open(
        os.path.dirname(os.path.abspath(__file__))
        + '/' + dir + '/innerese_operator_traces.b',
        'rb'))
    string = ''
    for epoch in traces:
        i = 0
        externalized = False
        while i < len(epoch[1]):
            if epoch[1][i] == 'externalize':
                externalized = True
                i += 1
            elif externalized and epoch[1][i] == 'internalize':
                epoch[1][i] = r'\{ ' + epoch[1][i] + \
                    ', ' + epoch[1][i - 1] + r' \}'
                del epoch[1][i - 1]
            elif epoch[1][i].startswith('( then'):
                epoch[1][i] = r'\{ [ ' + epoch[1][i - 2] + ', ' + epoch[
                    1][i - 1] + ']' + ', ( then ' + epoch[
                    1][i - 2] + ' ' + epoch[1][i - 1] + r' ) \}'
                del epoch[1][i - 1]
                del epoch[1][i - 2]
                i -= 1
            else:
                i += 1
        string += r'\scriptsize' + '\n'
        for operator in epoch[1]:
            string += operator + ', '
        string = string[:-2]
        string += r' \\ \hline ' + '\n'
    return string
