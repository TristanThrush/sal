First, add all of the directories (sal/sal, sal/trained_models, sal/domains) to your Python path.
You may also need to install the relevant Python package dependencies for this project.
The following commands, entered in a Python 3 terminal, show how to use SAL.


To initialize SAL with self referential operators:

```python
import sal
import manipulation
s = sal.SAL()
s.atoms = manipulation.ManipulationAtoms()
s.atoms = manipulation.ManipulationProblems()
```


To initialize SAL without the self-referential operators:

```python
import sal
import manipulation
s = sal.SAL()
s.atoms = manipulation.ManipulationAtoms()
s.atoms = manipulation.ManipulationProblems()
del s.atoms.atoms['then']
del s.atoms.atoms['remember']
del s.atoms.atoms['forget']
del s.atoms.atoms['internalize']
del s.atoms.atoms['externalize']
```


Trained SAL instances are provided in sal/trained_models/manipulation_self_referential and sal/trained_models/manipulation_not_self_referential.
To load a saved SAL instance:

```python
s.load(<desired directory>)
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
s.save(<desired directory>)
```


To run the generalization tests from my thesis, when SAL has the self-referential operators:

```python
import analysis
analysis.test_generalizability(s)
```


To print a latex-friendly readout of the entire training history of a saved SAL instance:

```python
import analysis
analysis.latexable_innerese_operator_traces(s)
```
