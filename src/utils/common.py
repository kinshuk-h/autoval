import tqdm.auto as true_tqdm

class FauxTQDM:
    def __init__(self, *args, **kwargs) -> None:
        self.iterable = args[0]

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

def tqdm(*args, **kwargs):
    if len(args[0]) > 1:
        return true_tqdm.tqdm(*args, **kwargs)
    else:
        return FauxTQDM(*args, **kwargs)