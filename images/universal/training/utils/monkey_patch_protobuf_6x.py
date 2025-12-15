# Works with protobuf 6.30 – 6.31 (and probably later 6.x)

import google.protobuf.message_factory as _mf

# https://protobuf.dev/news/v30/#remove-deprecated
if not hasattr(_mf.MessageFactory, "GetPrototype"):
    _mf.MessageFactory.GetPrototype = staticmethod(_mf.GetMessageClass)