class Task:
    def __init__(self, task_prefix, task_functions) -> None:
        self.task_prefix = task_prefix
        self.task_functions = task_functions

    @staticmethod
    def get_task_executor(task_fn):
        def new_task_fn(*args, **kwargs):
            result = task_fn(*args, **kwargs)
            if result is None: return {}
            if not isinstance(result, dict):
                result = { task_fn.__name__: result }
            return result
        return new_task_fn

    def execute(self):
        result = {}
        for task_function in self.task_functions:
            task_function = self.get_task_executor(task_function)
            task_result = task_function(**result)
            result.update(task_result)
            print()
        return result

    def print(self, *args, **kwargs):
        return print(f"{self.task_prefix}:", *args, **kwargs)