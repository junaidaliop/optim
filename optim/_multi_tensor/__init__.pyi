from functools import partial

from .. import (Adam, AdamW, NAdam, SGD, RAdam, RMSprop, Rprop, ASGD, Adamax, Adadelta, Adagrad)

Adam = partial(Adam, foreach=True)
AdamW = partial(AdamW, foreach=True)
NAdam = partial(NAdam, foreach=True)
SGD = partial(SGD, foreach=True)
RAdam = partial(RAdam, foreach=True)
RMSprop = partial(RMSprop, foreach=True)
Rprop = partial(Rprop, foreach=True)
ASGD = partial(ASGD, foreach=True)
Adamax = partial(Adamax, foreach=True)
Adadelta = partial(Adadelta, foreach=True)
Adagrad = partial(Adagrad, foreach=True)
