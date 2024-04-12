# AutoVal - Automatic Assignment Evaluation Utility

`autoval` is an automatic assignment evaluation utility, specifically tailored for the DS-207: Introduction to NLP Course offered at the [Department of Computational and Data Sciences (CDS)](https://cds.iisc.ac.in), [Indian Institute of Science](https://isc.ac.in).

`autoval` can work directly with archive files obtained from Microsoft Teams, and is easily configurable to adapt to specific needs.

For operation, the following components are required:
- A configuration file (e.g.: [`config/ds207-assignment-3.json`](/config/ds207-assignment-3.json)) that specifies the files to extract, the marks distribution for different tests and other details.
- A `templates` folder, that specifies what modules to generate from parsed student code.
- A `tests` folder, that specifies tests to evaluate student modules against.

The only requirement with student code is that code to be evaluated and parsed must be demarcated by the special comments:

```python
# ==== BEGIN EVALUATION PORTION
# ==== END EVALUATION PORTION
```
for a code block, and

```python
# BEGIN CODE : <segment_name>
# END CODE
```

for a code segment, preferably inside a code-block comment block.

### Templates

A template contains special `# >>> {...} <<<` comments that specify the part of student code to populate with. Populated templates are executed to validate student code.
There are three types of components currently supported (to be mentioned inside the `{...}`):

- `block`: Loads an entire code block. Syntax: `# >>> {block:<block_name>} <<<`.
- `segment`: Loads a specific part of a code block. Syntax: `# >>> {segment:<segment_name>} <<<`.
- `variable`: Loads a variable (optionally as a declaration/statement or just its value).
  -   Load as a statement: `# >>> {variable:<var_name>:stmt} <<<`
  -   Load just the value: `# >>> {variable:<var_name>:value} <<<`

An example template looks like the following, and gets populated with student code for all students to be graded:

```python

import os
...

# >>> {block:tokenizer} <<<

def init_tokenizers():
    # >>> {segment:tokenizer.create} <<<

    return src_tokenizer, tgt_tokenizer

```

For more concrete examples, look at [some templates](/templates) implemented for [Assignment 3 for DS-207](https://colab.research.google.com/drive/1LUbHW5wf2l2WjMTtEg4dc5vij8fOJecG).

### Tests

A test module is a simple python file that declares test functions to evaluate student modules against. Each populated template is passed as a module to the test module which can use the implemented functions.
A test module must define a `Context` class, of the following form:

```python
class Context:
    def __init__(self, module, record, data_dir):
        # Module refers to the student-specific implementation, as a compiled python module (executable).
        # record refers to parsed student data, such as additionally submitted files.
        # data_dir is the root directory where intermediate files are stored.
        ...
```

Any test functions that need to be defined must start with the prefix `test_`, and must accept only a context object as a parameter. Example:

```python

def test_tokenizer_functionality(context: Context):
    ...
```

Tests utilize assert statements or other exceptions to signify failure. Further, test functions may return a dictionary of values, which is cached in the student record.

For concrete examples, look at [some test modules](/tests) implemented for [Assignment 3 for DS-207](https://colab.research.google.com/drive/1LUbHW5wf2l2WjMTtEg4dc5vij8fOJecG).
