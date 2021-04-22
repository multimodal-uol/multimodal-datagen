# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    List as typing___List,
    NewType as typing___NewType,
    Tuple as typing___Tuple,
    cast as typing___cast,
)


builtin___int = int
builtin___str = str


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

AuthTypeValue = typing___NewType('AuthTypeValue', builtin___int)
type___AuthTypeValue = AuthTypeValue
class AuthType(object):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    @classmethod
    def Name(cls, number: builtin___int) -> builtin___str: ...
    @classmethod
    def Value(cls, name: builtin___str) -> AuthTypeValue: ...
    @classmethod
    def keys(cls) -> typing___List[builtin___str]: ...
    @classmethod
    def values(cls) -> typing___List[AuthTypeValue]: ...
    @classmethod
    def items(cls) -> typing___List[typing___Tuple[builtin___str, AuthTypeValue]]: ...
    undef = typing___cast(AuthTypeValue, 0)
    NoAuth = typing___cast(AuthTypeValue, 1)
    KeyAuth = typing___cast(AuthTypeValue, 2)
    SessionTokenAuth = typing___cast(AuthTypeValue, 3)
    AdminAuth = typing___cast(AuthTypeValue, 4)
    PATAuth = typing___cast(AuthTypeValue, 5)
undef = typing___cast(AuthTypeValue, 0)
NoAuth = typing___cast(AuthTypeValue, 1)
KeyAuth = typing___cast(AuthTypeValue, 2)
SessionTokenAuth = typing___cast(AuthTypeValue, 3)
AdminAuth = typing___cast(AuthTypeValue, 4)
PATAuth = typing___cast(AuthTypeValue, 5)
type___AuthType = AuthType
