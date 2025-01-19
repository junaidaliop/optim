"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
from functools import partialmethod

from .. import (Adam, AdamW, NAdam, SGD, RAdam, RMSprop, Rprop, ASGD, Adamax, Adadelta, Adagrad)


def partialclass(cls, *args, **kwargs):  # noqa: D103
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


Adam = partialclass(Adam, foreach=True)
AdamW = partialclass(AdamW, foreach=True)
NAdam = partialclass(NAdam, foreach=True)
SGD = partialclass(SGD, foreach=True)
RAdam = partialclass(RAdam, foreach=True)
RMSprop = partialclass(RMSprop, foreach=True)
Rprop = partialclass(Rprop, foreach=True)
ASGD = partialclass(ASGD, foreach=True)
Adamax = partialclass(Adamax, foreach=True)
Adadelta = partialclass(Adadelta, foreach=True)
Adagrad = partialclass(Adagrad, foreach=True)
