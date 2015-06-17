from functools import partial

def with_name(obj, name):
    obj.name = name
    return obj

def named(name):
    return partial(with_name, name=name)
