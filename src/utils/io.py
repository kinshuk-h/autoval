import os
import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as ifile:
        return json.load(ifile)

def write_json(data, path, **kwargs):
    kwargs['indent']       = kwargs.get('indent', 2)
    kwargs['ensure_ascii'] = kwargs.get('ensure_ascii', False)

    with open(path, 'w', encoding='utf-8') as ofile:
        json.dump(data, ofile, **kwargs)

def jprint(*args, indent=2, **kwargs):
    return print(*[ json.dumps(obj, ensure_ascii=False, indent=indent) for obj in args ], **kwargs)

def get_file(directory, extension, index=0):
    files = [
        file for file in os.listdir(directory)
        if os.path.splitext(file)[1] == "."+extension
    ]
    return files[index] if len(files) > index else None

class Record(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @staticmethod
    def load(path):
        return Record(read_json(path))

    @staticmethod
    def load_if(path):
        if os.path.exists(path):
            return Record(read_json(path))
        else:
            return Record()

    def __repr__(self) -> str:
        return f"Record({super().__repr__()})"

    def save(self, path):
        write_json(self, path)

    def deepget(self, deepkey, default=None):
        retval = self
        for key in (deepkey if isinstance(deepkey, (list, tuple)) else deepkey.split('.')):
            if key not in retval: return default
            retval = retval[key]
        return retval

    def deepset(self, deepkey, value):
        keys = deepkey if isinstance(deepkey, (list, tuple)) else deepkey.split('.')
        ref = self
        for i, key in enumerate(keys):
            if i == len(keys)-1: break
            if key not in ref:
                ref[key] = {}
            ref = ref[key]
        ref[keys[-1]] = value