from collections import OrderedDict
from functools import partial

class VarType:
    def __init__(self, **methods):
        if "should_ignore" not in methods:
            methods["should_ignore"] = lambda: False
        for name, fun in methods.items():
            setattr(self, name, fun)


Integer = VarType(
    from_str = int
)

String = VarType(
    from_str = lambda x:x
)
Ignore = VarType(
    should_ignore = lambda: True
)



def _from_tuple(members, r_type, items):
    if len(items) != len(members):
        raise IndexError(\
            "items len should be {} but it's {}".format(len(members), len(items)))
    res = r_type()
    for ((var_name, var_type), value) in zip(members.items(), items):
        if not var_type.should_ignore():
            setattr(res, var_name, var_type.from_str(value))
    return res

def _make_to_str(members):
    def to_str(self):
        return "{" + ", ".join("{}: {}".format(a_name, getattr(self, a_name)) \
            for a_name, a_type in members.items() if not a_type.should_ignore()) +"}"
    return to_str

class ModelMeta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kws):
        return OrderedDict()

    def __new__(cls, name, bases, namespace, **kws):
        members = OrderedDict(x for x in namespace.items() \
            if isinstance(x[1], VarType))
        namespace = dict(x for x in namespace.items()\
            if not isinstance(x[1], VarType))
        namespace["__slots__"] = tuple(members)
        namespace["__str__"] = _make_to_str(members)
        result = type.__new__(cls, name, bases, namespace)
        result.from_tuple = \
            staticmethod(partial(_from_tuple, members, result))
        return result

class Model(metaclass=ModelMeta):
    pass
