from common import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ScheduleCallRequest(_message.Message):
    __slots__ = ("functionID", "data")
    FUNCTIONID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    functionID: _common_pb2.FunctionID
    data: bytes
    def __init__(self, functionID: _Optional[_Union[_common_pb2.FunctionID, _Mapping]] = ..., data: _Optional[bytes] = ...) -> None: ...

class ScheduleCallResponse(_message.Message):
    __slots__ = ("data", "error")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    error: _common_pb2.Error
    def __init__(self, data: _Optional[bytes] = ..., error: _Optional[_Union[_common_pb2.Error, _Mapping]] = ...) -> None: ...

class CreateFunctionRequest(_message.Message):
    __slots__ = ("image_tag", "config")
    IMAGE_TAG_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    image_tag: _common_pb2.ImageTag
    config: _common_pb2.Config
    def __init__(self, image_tag: _Optional[_Union[_common_pb2.ImageTag, _Mapping]] = ..., config: _Optional[_Union[_common_pb2.Config, _Mapping]] = ...) -> None: ...

class CreateFunctionResponse(_message.Message):
    __slots__ = ("functionID",)
    FUNCTIONID_FIELD_NUMBER: _ClassVar[int]
    functionID: _common_pb2.FunctionID
    def __init__(self, functionID: _Optional[_Union[_common_pb2.FunctionID, _Mapping]] = ...) -> None: ...
