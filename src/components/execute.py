import os
import sys
import glob
import inspect
import traceback
import importlib.util

import regex

from .task import Task
from ..utils import io, common

def load_module(module_name, source_file):
    """ Loads a module from a source .py file.

    Args:
        module_name (str): Name of the module to use.
        source_file (str): Path to the source file.

    Returns:
        ModuleType: Eagerly loaded module.
    """

    spec = importlib.util.spec_from_file_location(module_name, source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestExecutorTask(Task):
    """ Sub-Task for executing tests against student codes. """

    def __init__(self, data_dir, test_dir, students_list, modules, functions, skip_existing=False) -> None:
        super().__init__("TEST", [
            self.load_test_modules,
            self.execute_tests
        ])

        self.test_dir = test_dir
        self.data_dir = data_dir
        self.students_list = students_list

        self.skip_existing = skip_existing
        self.record_dir    = os.path.join(self.data_dir, "records")

        if self.students_list is None:
            self.students_list = [ file[:-5] for file in os.listdir(self.record_dir) ]

        self.tests = {}
        self.test_modules = modules
        self.test_functions = [ regex.compile(rf"(?ui).*{pattern}.*") for pattern in functions ] if functions else None

    def in_function_spec(self, name):
        if self.test_functions is None: return True
        for pattern in self.test_functions:
            if pattern.search(name) is not None: return True
        return False

    def rank_wise_order(self, functions):
        if self.test_functions is None: return functions
        idx, rank = 0, {}
        for pattern in self.test_functions:
            for fx in functions:
                if pattern.search(fx.__name__[5:]) is not None:
                    rank[fx.__name__[5:]] = idx
        return sorted(functions, key=lambda fx: rank[fx.__name__[5:]])

    def load_test_modules(self):
        self.print("Loading test modules from", self.test_dir, "...", end=' ', flush=True)

        tests = glob.glob(os.path.join(self.test_dir, "*.py"))

        for test_path in tests:
            suite_name = os.path.splitext(os.path.basename(test_path))[0]
            if suite_name == 'common': continue
            if self.test_modules is not None and suite_name not in self.test_modules: continue
            test_module = load_module("tests." + suite_name, test_path)
            self.tests[suite_name] = {
                'module' : test_module,
                'context': test_module.Context,
                'tests'  : self.rank_wise_order([
                    function for name, function in inspect.getmembers(test_module, inspect.isfunction)
                    if name.startswith('test_') and self.in_function_spec(name)
                ])
            }

        print("done")

        self.print("Discovered test suites:")
        for suite_name, test in self.tests.items():
            self.print(" TEST SUITE:", suite_name)
            for fx in test['tests']:
                self.print("    ", "TEST:", fx.__name__[5:])

    def execute_tests(self):
        self.print("Executing test modules using student implementations ...")

        for test_name, test in self.tests.items():
            self.print("Executing test suite:", test_name, "...")

            for student in (pbar := common.tqdm(self.students_list)):
                pbar.set_description(student)
                student_data = io.Record.load(os.path.join(self.record_dir, f"{student}.json"))

                missing_tests = student_data.deepget(("tests", test_name)) is None
                missing_tests = missing_tests or any(
                    fx.__name__[5:] not in student_data['tests'][test_name] for fx in test['tests']
                )
                if not self.skip_existing or missing_tests:
                    module_file = student_data.deepget(f"meta.code.{test_name}")
                    if module_file is None or not os.path.exists(module_file):
                        self.print(f"Student {student}: missing {test_name} module implementation")
                        continue

                    try:
                        student_module = load_module("student." + test_name, module_file)
                    except:
                        self.print(f"Student {student}: {test_name} module cannot be imported")
                        if len(self.students_list) == 1:
                            for line_set in traceback.format_exception(exc):
                                for line in line_set.split('\n'): self.print("   ", line)
                        continue

                    context = test['context'](student_module, student_data, self.data_dir)

                    net_result  = student_data.deepget(('tests'  , test_name), {})
                    net_outputs = student_data.deepget(('outputs', test_name), {})
                    for test_fx in test['tests']:
                        fn_name = test_fx.__name__[5:]
                        pbar.set_postfix(test=fn_name)

                        if not self.skip_existing or student_data.deepget(('tests', test_name, fn_name)) is None:
                            old_main_module = sys.modules['__main__']
                            sys.modules['__main__'] = student_module
                            try:
                                output = test_fx(context)
                                if isinstance(output, dict): net_outputs.update(output)
                                net_result[fn_name] = { 'status': True }
                            except Exception as exc:
                                if len(self.students_list) == 1:
                                    for line_set in traceback.format_exception(exc):
                                        for line in line_set.split('\n'): self.print("   ", line)
                                net_result[fn_name] = { 'status': False, 'reason': str(exc) }
                                if output := getattr(exc, 'outputs', None) is not None:
                                    if isinstance(output, dict): net_outputs.update(output)
                            sys.modules['__main__'] = old_main_module

                    student_data.deepset(f"outputs.{test_name}", net_outputs)
                    student_data.deepset(f"tests.{test_name}", net_result)
                    student_data.save(os.path.join(self.record_dir, f"{student}.json"))

                elif len(self.students_list) == 1:
                    for test_fx in test['tests']:
                        fn_name = test_fx.__name__[5:]
                        result = student_data.deepget(('tests', test_name, fn_name))
                        if result['status'] == False:
                            self.print(f"Student {student}: failed test:", fn_name, ":", result['reason'])

