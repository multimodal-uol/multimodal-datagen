# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/clarifai/api/status/status.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from clarifai_grpc.grpc.auth.util import extension_pb2 as proto_dot_clarifai_dot_auth_dot_util_dot_extension__pb2
from clarifai_grpc.grpc.api.status import status_code_pb2 as proto_dot_clarifai_dot_api_dot_status_dot_status__code__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/clarifai/api/status/status.proto',
  package='clarifai.api.status',
  syntax='proto3',
  serialized_options=b'\n\034com.clarifai.grpc.api.statusP\001Z\006status\242\002\004CAIP',
  serialized_pb=b'\n&proto/clarifai/api/status/status.proto\x12\x13\x63larifai.api.status\x1a(proto/clarifai/auth/util/extension.proto\x1a+proto/clarifai/api/status/status_code.proto\"\xdb\x01\n\x06Status\x12-\n\x04\x63ode\x18\x01 \x01(\x0e\x32\x1f.clarifai.api.status.StatusCode\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x0f\n\x07\x64\x65tails\x18\x03 \x01(\t\x12\x19\n\x0bstack_trace\x18\x04 \x03(\tB\x04\x80\x9c\'\x01\x12\x19\n\x11percent_completed\x18\x05 \x01(\r\x12\x16\n\x0etime_remaining\x18\x06 \x01(\r\x12\x0e\n\x06req_id\x18\x07 \x01(\t\x12\x1e\n\x10internal_details\x18\x08 \x01(\tB\x04\x80\x9c\'\x01\";\n\x0c\x42\x61seResponse\x12+\n\x06status\x18\x01 \x01(\x0b\x32\x1b.clarifai.api.status.StatusB/\n\x1c\x63om.clarifai.grpc.api.statusP\x01Z\x06status\xa2\x02\x04\x43\x41IPb\x06proto3'
  ,
  dependencies=[proto_dot_clarifai_dot_auth_dot_util_dot_extension__pb2.DESCRIPTOR,proto_dot_clarifai_dot_api_dot_status_dot_status__code__pb2.DESCRIPTOR,])




_STATUS = _descriptor.Descriptor(
  name='Status',
  full_name='clarifai.api.status.Status',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='clarifai.api.status.Status.code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='clarifai.api.status.Status.description', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='details', full_name='clarifai.api.status.Status.details', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stack_trace', full_name='clarifai.api.status.Status.stack_trace', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\200\234\'\001', file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='percent_completed', full_name='clarifai.api.status.Status.percent_completed', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_remaining', full_name='clarifai.api.status.Status.time_remaining', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='req_id', full_name='clarifai.api.status.Status.req_id', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='internal_details', full_name='clarifai.api.status.Status.internal_details', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\200\234\'\001', file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=370,
)


_BASERESPONSE = _descriptor.Descriptor(
  name='BaseResponse',
  full_name='clarifai.api.status.BaseResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='clarifai.api.status.BaseResponse.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=372,
  serialized_end=431,
)

_STATUS.fields_by_name['code'].enum_type = proto_dot_clarifai_dot_api_dot_status_dot_status__code__pb2._STATUSCODE
_BASERESPONSE.fields_by_name['status'].message_type = _STATUS
DESCRIPTOR.message_types_by_name['Status'] = _STATUS
DESCRIPTOR.message_types_by_name['BaseResponse'] = _BASERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'proto.clarifai.api.status.status_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.status.Status)
  })
_sym_db.RegisterMessage(Status)

BaseResponse = _reflection.GeneratedProtocolMessageType('BaseResponse', (_message.Message,), {
  'DESCRIPTOR' : _BASERESPONSE,
  '__module__' : 'proto.clarifai.api.status.status_pb2'
  # @@protoc_insertion_point(class_scope:clarifai.api.status.BaseResponse)
  })
_sym_db.RegisterMessage(BaseResponse)


DESCRIPTOR._options = None
_STATUS.fields_by_name['stack_trace']._options = None
_STATUS.fields_by_name['internal_details']._options = None
# @@protoc_insertion_point(module_scope)
