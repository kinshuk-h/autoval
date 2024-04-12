import os
import sys
import inspect
import argparse
import importlib.util

import traceback

from src.utils import io

def load_module(module_name, source_file):
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_test_module(test_path):
    test_name = os.path.splitext(os.path.basename(test_path))[0]
    test_module = load_module("tests." + test_name, test_path)
    return test_name, {
        'module' : test_module,
        'context': test_module.Context,
        'tests'  : [
            function for name, function in inspect.getmembers(test_module, inspect.isfunction)
            if name.startswith('test_')
        ]
    }

def execute_tests(test_path, module_file, student_data):
    test_name, test = load_test_module(test_path)

    print("Executing test suite:", test_name, "...")

    try:
        student_module = load_module("student." + test_name, module_file)
    except:
        print(f"{test_name} module cannot be imported")

    context = test['context'](student_module, student_data, "data")
    net_result, net_outputs = {}, {}

    for test_fx in test['tests']:
        fn_name = test_fx.__name__[5:]

        old_main_module = sys.modules['__main__']
        sys.modules['__main__'] = student_module
        try:
            output = test_fx(context)
            if isinstance(output, dict): net_outputs.update(output)
            net_result[fn_name] = { 'status': True }
        except Exception as exc:
            traceback.print_exc()
            net_result[fn_name] = { 'status': False, 'reason': str(exc) }
        sys.modules['__main__'] = old_main_module

    io.jprint(net_result)
    io.jprint(net_outputs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execute tests against a specified dummy module")
    parser.add_argument("module", type=str, help="module to test with and for.")

    args = parser.parse_args()

    TEST_MODULE = f"tests/{args.module}.py"
    CODE_MODULE = f"dump/{args.module}.py"
    RECORD = io.Record.load(f"dump/SAPName.json")

    execute_tests(TEST_MODULE, CODE_MODULE, RECORD)