First, add all of the directories (sal/sal, sal/trained_models, sal/domains) to your Python path.
You may also need to install the relevant Python package dependencies for this project.
The following commands, entered in a Python 3 terminal, show how to use SAL.
Note that everywhere you see <desired model directory>, an absolute path must be typed.


To initialize SAL with self referential operators:

```python
import sal
import manipulation
s = sal.SAL()
s.atoms = manipulation.ManipulationAtoms()
s.problems = manipulation.ManipulationProblems()
```


To initialize SAL without the self-referential operators:

```python
import sal
import manipulation
s = sal.SAL()
s.atoms = manipulation.ManipulationAtoms()
s.problems = manipulation.ManipulationProblems()
del s.atoms.atoms['then']
del s.atoms.atoms['remember']
del s.atoms.atoms['forget']
del s.atoms.atoms['internalize']
del s.atoms.atoms['externalize']
```


Note that after SAL's problems are initialized, a simulator visual file will be printed, (such as '.49456.txt'). After SAL starts solving problems, you can open up another terminal and enter the following command to see a real time visualization of the problem that SAL is solving:

```python
import manipulation
manipulation.SimulatorVisualizer.visualize('.<number given>.txt')
```


Trained SAL instances are provided in sal/trained_models/manipulation_self_referential and sal/trained_models/manipulation_not_self_referential.
To load a saved SAL instance:

```python
s.load(<desired model directory>)
```


To evaluate SAL's performance (first making |G(.,.)| and |G(.,.)_e| something reasonable):

```python
s.learner.G = 1000  # Change to 150 if evaluating SAL without self-referential operators
s.learner.GE = 1000  # Change to 150 if evaluating SAL without self-referential operators
s.perform(1)
```


To train SAL yourself for 40 epochs, followed by saving your model:

```python
s.learner.G = 500  # Change to 150 if training SAL without self-referential operators
s.learner.GE = 2
s.learn(40)
s.save(<desired model directory>)
```


To run the generalization tests from my thesis, when SAL has the self-referential operators:

```python
import analysis
analysis.test_generalizability(s)
```


To print a latex-friendly readout of the entire training history of a saved SAL instance:

```python
import analysis
analysis.latexable_innerese_operator_traces(<desired model directory>)
```
