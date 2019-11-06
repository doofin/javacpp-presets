// Targeted by JavaCPP version 1.5.2: DO NOT EDIT THIS FILE

package org.bytedeco.onnxruntime.global;

import org.bytedeco.onnxruntime.*;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class onnxruntime extends org.bytedeco.onnxruntime.presets.onnxruntime {
    static { Loader.load(); }

// Parsed from onnxruntime/core/session/onnxruntime_c_api.h

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// #pragma once
// #include <stdlib.h>
// #include <stdint.h>
// #include <string.h>

// This value is used in structures passed to ORT so that a newer version of ORT will still work with
public static final int ORT_API_VERSION = 1;

// #ifdef __cplusplus
// #endif

// SAL2 Definitions
// #ifndef _WIN32
// #define _In_
// #define _In_opt_
// #define _Out_
// #define _Outptr_
// #define _Out_opt_
// #define _Inout_
// #define _Inout_opt_
// #define _Frees_ptr_opt_
// #define ORT_ALL_ARGS_NONNULL __attribute__((nonnull))
// #else
// #include <specstrings.h>
// #define ORT_ALL_ARGS_NONNULL
// #endif

// #ifdef _WIN32
// Define ORT_DLL_IMPORT if your program is dynamically linked to Ort.
// dllexport is not used, we use a .def file.
// #ifdef ORT_DLL_IMPORT
// #define ORT_EXPORT __declspec(dllimport)
// #else
// #define ORT_EXPORT
// #endif
// #define ORT_API_CALL _stdcall
// #define ORT_MUST_USE_RESULT
// #define ORTCHAR_T wchar_t
// #else
// #define ORT_EXPORT
// #define ORT_API_CALL
// #define ORT_MUST_USE_RESULT __attribute__((warn_unused_result))
// #define ORTCHAR_T char
// #endif

// #ifndef ORT_TSTR
// #ifdef _WIN32
// #define ORT_TSTR(X) L##X
// #else
// #define ORT_TSTR(X) (X)
// #endif
// #endif

// Any pointer marked with _In_ or _Out_, cannot be NULL.

// #ifdef __cplusplus
// Windows users should use unicode paths when possible to bypass the MAX_PATH limitation
// Every pointer marked with _In_ or _Out_, cannot be NULL. Caller should ensure that.
// for ReleaseXXX(...) functions, they can accept NULL pointer.
// #define NO_EXCEPTION noexcept
// #else
// #define NO_EXCEPTION
// #endif

// Copied from TensorProto::DataType
// Currently, Ort doesn't support complex64, complex128, bfloat16 types
/** enum ONNXTensorElementDataType */
public static final int
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,   // maps to c type float
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,   // maps to c type uint8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,    // maps to c type int8_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,  // maps to c type uint16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,   // maps to c type int16_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,   // maps to c type int32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,   // maps to c type int64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,  // maps to c++ type std::string
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,      // maps to c type double
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,      // maps to c type uint32_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,      // maps to c type uint64_t
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,   // complex with float32 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,  // complex with float64 real and imaginary components
  ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16;    // Non-IEEE floating-point format based on IEEE754 single-precision

// Synced with onnx TypeProto oneof
/** enum ONNXType */
public static final int
  ONNX_TYPE_UNKNOWN = 0,
  ONNX_TYPE_TENSOR = 1,
  ONNX_TYPE_SEQUENCE = 2,
  ONNX_TYPE_MAP = 3,
  ONNX_TYPE_OPAQUE = 4,
  ONNX_TYPE_SPARSETENSOR = 5;

/** enum OrtLoggingLevel */
public static final int
  ORT_LOGGING_LEVEL_VERBOSE = 0,
  ORT_LOGGING_LEVEL_INFO = 1,
  ORT_LOGGING_LEVEL_WARNING = 2,
  ORT_LOGGING_LEVEL_ERROR = 3,
  ORT_LOGGING_LEVEL_FATAL = 4;

/** enum OrtErrorCode */
public static final int
  ORT_OK = 0,
  ORT_FAIL = 1,
  ORT_INVALID_ARGUMENT = 2,
  ORT_NO_SUCHFILE = 3,
  ORT_NO_MODEL = 4,
  ORT_ENGINE_ERROR = 5,
  ORT_RUNTIME_EXCEPTION = 6,
  ORT_INVALID_PROTOBUF = 7,
  ORT_MODEL_LOADED = 8,
  ORT_NOT_IMPLEMENTED = 9,
  ORT_INVALID_GRAPH = 10,
  ORT_SHAPE_INFERENCE_NOT_REGISTERED = 11,
  ORT_REQUIREMENT_NOT_REGISTERED = 12;

// __VA_ARGS__ on Windows and Linux are different
// #define ORT_API(RETURN_TYPE, NAME, ...)
//   ORT_EXPORT RETURN_TYPE ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

// #define ORT_API_STATUS(NAME, ...)
//   ORT_EXPORT OrtStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION ORT_MUST_USE_RESULT

// Used in *.cc files. Almost as same as ORT_API_STATUS, except without ORT_MUST_USE_RESULT
// #define ORT_API_STATUS_IMPL(NAME, ...)
//   ORT_EXPORT OrtStatus* ORT_API_CALL NAME(__VA_ARGS__) NO_EXCEPTION

// #define ORT_RUNTIME_CLASS(X)
//   struct Ort##X;
//   typedef struct Ort##X Ort##X;
//   ORT_API(void, OrtRelease##X, _Frees_ptr_opt_ Ort##X* input);
// Targeting ../OrtEnv.java


  public static native void OrtReleaseEnv( OrtEnv input);
// Targeting ../OrtStatus.java


  public static native void OrtReleaseStatus( OrtStatus input);
// Targeting ../OrtProvider.java


  public static native void OrtReleaseProvider( OrtProvider input);
// Targeting ../OrtAllocatorInfo.java


  public static native void OrtReleaseAllocatorInfo( OrtAllocatorInfo input);
// Targeting ../OrtSession.java


  public static native void OrtReleaseSession( OrtSession input);
// Targeting ../OrtValue.java


  public static native void OrtReleaseValue( OrtValue input);
// Targeting ../OrtValueList.java


  public static native void OrtReleaseValueList( OrtValueList input);
// Targeting ../OrtRunOptions.java


  public static native void OrtReleaseRunOptions( OrtRunOptions input);
// Targeting ../OrtTypeInfo.java


  public static native void OrtReleaseTypeInfo( OrtTypeInfo input);
// Targeting ../OrtTensorTypeAndShapeInfo.java


  public static native void OrtReleaseTensorTypeAndShapeInfo( OrtTensorTypeAndShapeInfo input);
// Targeting ../OrtSessionOptions.java


  public static native void OrtReleaseSessionOptions( OrtSessionOptions input);
// Targeting ../OrtCallback.java


  public static native void OrtReleaseCallback( OrtCallback input);
// Targeting ../OrtCustomOpDomain.java


  public static native void OrtReleaseCustomOpDomain( OrtCustomOpDomain input);
  public static native void OrtReleaseAllocator( OrtAllocator input);
// Targeting ../OrtAllocator.java


// Targeting ../OrtLoggingFunction.java



/**
 * @param out Should be freed by {@code OrtReleaseEnv} after use
 */
public static native OrtStatus OrtCreateEnv( @Cast("OrtLoggingLevel") int default_logging_level, @Cast("const char*") BytePointer logid, @Cast("OrtEnv**") PointerPointer out);
public static native OrtStatus OrtCreateEnv( @Cast("OrtLoggingLevel") int default_logging_level, @Cast("const char*") BytePointer logid, @ByPtrPtr OrtEnv out);
public static native OrtStatus OrtCreateEnv( @Cast("OrtLoggingLevel") int default_logging_level, String logid, @ByPtrPtr OrtEnv out);

/**
 * @param out Should be freed by {@code OrtReleaseEnv} after use
 */
public static native OrtStatus OrtCreateEnvWithCustomLogger( OrtLoggingFunction logging_function,
               Pointer logger_param, @Cast("OrtLoggingLevel") int default_logging_level,
               @Cast("const char*") BytePointer logid,
               @Cast("OrtEnv**") PointerPointer out);
public static native OrtStatus OrtCreateEnvWithCustomLogger( OrtLoggingFunction logging_function,
               Pointer logger_param, @Cast("OrtLoggingLevel") int default_logging_level,
               @Cast("const char*") BytePointer logid,
               @ByPtrPtr OrtEnv out);
public static native OrtStatus OrtCreateEnvWithCustomLogger( OrtLoggingFunction logging_function,
               Pointer logger_param, @Cast("OrtLoggingLevel") int default_logging_level,
               String logid,
               @ByPtrPtr OrtEnv out);

// TODO: document the path separator convention? '/' vs '\'
// TODO: should specify the access characteristics of model_path. Is this read only during the
// execution of OrtCreateSession, or does the OrtSession retain a handle to the file/directory
// and continue to access throughout the OrtSession lifetime?
//  What sort of access is needed to model_path : read or read/write?
public static native OrtStatus OrtCreateSession( OrtEnv env, @Cast("const char*") BytePointer model_path,
               @Const OrtSessionOptions options, @Cast("OrtSession**") PointerPointer out);
public static native OrtStatus OrtCreateSession( OrtEnv env, @Cast("const char*") BytePointer model_path,
               @Const OrtSessionOptions options, @ByPtrPtr OrtSession out);
public static native OrtStatus OrtCreateSession( OrtEnv env, String model_path,
               @Const OrtSessionOptions options, @ByPtrPtr OrtSession out);

public static native OrtStatus OrtCreateSessionFromArray( OrtEnv env, @Const Pointer model_data, @Cast("size_t") long model_data_length,
               @Const OrtSessionOptions options, @Cast("OrtSession**") PointerPointer out);
public static native OrtStatus OrtCreateSessionFromArray( OrtEnv env, @Const Pointer model_data, @Cast("size_t") long model_data_length,
               @Const OrtSessionOptions options, @ByPtrPtr OrtSession out);

public static native OrtStatus OrtRun( OrtSession sess,
               @Const OrtRunOptions run_options,
               @Cast("const char*const*") PointerPointer input_names, @Cast("const OrtValue*const*") PointerPointer input, @Cast("size_t") long input_len,
               @Cast("const char*const*") PointerPointer output_names, @Cast("size_t") long output_names_len, @Cast("OrtValue**") PointerPointer out);
public static native OrtStatus OrtRun( OrtSession sess,
               @Const OrtRunOptions run_options,
               @Cast("const char*const*") @ByPtrPtr BytePointer input_names, @Const @ByPtrPtr OrtValue input, @Cast("size_t") long input_len,
               @Cast("const char*const*") @ByPtrPtr BytePointer output_names, @Cast("size_t") long output_names_len, @ByPtrPtr OrtValue out);
public static native OrtStatus OrtRun( OrtSession sess,
               @Const OrtRunOptions run_options,
               @Cast("const char*const*") @ByPtrPtr ByteBuffer input_names, @Const @ByPtrPtr OrtValue input, @Cast("size_t") long input_len,
               @Cast("const char*const*") @ByPtrPtr ByteBuffer output_names, @Cast("size_t") long output_names_len, @ByPtrPtr OrtValue out);
public static native OrtStatus OrtRun( OrtSession sess,
               @Const OrtRunOptions run_options,
               @Cast("const char*const*") @ByPtrPtr byte[] input_names, @Const @ByPtrPtr OrtValue input, @Cast("size_t") long input_len,
               @Cast("const char*const*") @ByPtrPtr byte[] output_names, @Cast("size_t") long output_names_len, @ByPtrPtr OrtValue out);

/**
 * @return A pointer of the newly created object. The pointer should be freed by OrtReleaseSessionOptions after use
 */
public static native OrtStatus OrtCreateSessionOptions( @Cast("OrtSessionOptions**") PointerPointer options);
public static native OrtStatus OrtCreateSessionOptions( @ByPtrPtr OrtSessionOptions options);

// create a copy of an existing OrtSessionOptions
public static native OrtStatus OrtCloneSessionOptions( OrtSessionOptions in_options, @Cast("OrtSessionOptions**") PointerPointer out_options);
public static native OrtStatus OrtCloneSessionOptions( OrtSessionOptions in_options, @ByPtrPtr OrtSessionOptions out_options);
public static native OrtStatus OrtEnableSequentialExecution( OrtSessionOptions options);
public static native OrtStatus OrtDisableSequentialExecution( OrtSessionOptions options);

// Enable profiling for this session.
public static native OrtStatus OrtEnableProfiling( OrtSessionOptions options, @Cast("const char*") BytePointer profile_file_prefix);
public static native OrtStatus OrtEnableProfiling( OrtSessionOptions options, String profile_file_prefix);
public static native OrtStatus OrtDisableProfiling( OrtSessionOptions options);

// Enable the memory pattern optimization.
// The idea is if the input shapes are the same, we could trace the internal memory allocation
// and generate a memory pattern for future request. So next time we could just do one allocation
// with a big chunk for all the internal memory allocation.
// Note: memory pattern optimization is only available when SequentialExecution enabled.
public static native OrtStatus OrtEnableMemPattern( OrtSessionOptions options);
public static native OrtStatus OrtDisableMemPattern( OrtSessionOptions options);

// Enable the memory arena on CPU
// Arena may pre-allocate memory for future usage.
// set this option to false if you don't want it.
public static native OrtStatus OrtEnableCpuMemArena( OrtSessionOptions options);
public static native OrtStatus OrtDisableCpuMemArena( OrtSessionOptions options);

// < logger id to use for session output
public static native OrtStatus OrtSetSessionLogId( OrtSessionOptions options, @Cast("const char*") BytePointer logid);
public static native OrtStatus OrtSetSessionLogId( OrtSessionOptions options, String logid);

// < applies to session load, initialization, etc
public static native OrtStatus OrtSetSessionLogVerbosityLevel( OrtSessionOptions options, int session_log_verbosity_level);

// Set Graph optimization level.
// Available options are : 0, 1, 2.
// 0 -> Disable all optimizations
// 1 -> Enable basic optimizations
// 2 -> Enable all optimizations
public static native OrtStatus OrtSetSessionGraphOptimizationLevel( OrtSessionOptions options, int graph_optimization_level);

// How many threads in the session thread pool.
public static native OrtStatus OrtSetSessionThreadPoolSize( OrtSessionOptions options, int session_thread_pool_size);

/**
  * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
  * functions to enable them in the session:
  *   OrtSessionOptionsAppendExecutionProvider_CPU
  *   OrtSessionOptionsAppendExecutionProvider_CUDA
  *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
  * The order they care called indicates the preference order as well. In other words call this method
  * on your most preferred execution provider first followed by the less preferred ones.
  * If none are called Ort will use its internal CPU execution provider.
  */

public static native OrtStatus OrtSessionGetInputCount( @Const OrtSession sess, @Cast("size_t*") SizeTPointer out);
public static native OrtStatus OrtSessionGetOutputCount( @Const OrtSession sess, @Cast("size_t*") SizeTPointer out);

/**
 * @param out  should be freed by OrtReleaseTypeInfo after use
 */
public static native OrtStatus OrtSessionGetInputTypeInfo( @Const OrtSession sess, @Cast("size_t") long index, @Cast("OrtTypeInfo**") PointerPointer type_info);
public static native OrtStatus OrtSessionGetInputTypeInfo( @Const OrtSession sess, @Cast("size_t") long index, @ByPtrPtr OrtTypeInfo type_info);

/**
 * @param out  should be freed by OrtReleaseTypeInfo after use
 */
public static native OrtStatus OrtSessionGetOutputTypeInfo( @Const OrtSession sess, @Cast("size_t") long index, @Cast("OrtTypeInfo**") PointerPointer type_info);
public static native OrtStatus OrtSessionGetOutputTypeInfo( @Const OrtSession sess, @Cast("size_t") long index, @ByPtrPtr OrtTypeInfo type_info);

/**
 * @param value  is set to a null terminated string allocated using 'allocator'. The caller is responsible in freeing it.
 */
public static native OrtStatus OrtSessionGetInputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") PointerPointer value);
public static native OrtStatus OrtSessionGetInputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr BytePointer value);
public static native OrtStatus OrtSessionGetInputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr ByteBuffer value);
public static native OrtStatus OrtSessionGetInputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr byte[] value);
public static native OrtStatus OrtSessionGetOutputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") PointerPointer value);
public static native OrtStatus OrtSessionGetOutputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr BytePointer value);
public static native OrtStatus OrtSessionGetOutputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr ByteBuffer value);
public static native OrtStatus OrtSessionGetOutputName( @Const OrtSession sess, @Cast("size_t") long index,
               OrtAllocator allocator, @Cast("char**") @ByPtrPtr byte[] value);

/**
 * @return A pointer to the newly created object. The pointer should be freed by OrtReleaseRunOptions after use
 */
public static native OrtStatus OrtCreateRunOptions( @Cast("OrtRunOptions**") PointerPointer out);
public static native OrtStatus OrtCreateRunOptions( @ByPtrPtr OrtRunOptions out);

public static native OrtStatus OrtRunOptionsSetRunLogVerbosityLevel( OrtRunOptions options, int value);
public static native OrtStatus OrtRunOptionsSetRunTag( OrtRunOptions arg0, @Cast("const char*") BytePointer run_tag);
public static native OrtStatus OrtRunOptionsSetRunTag( OrtRunOptions arg0, String run_tag);

public static native OrtStatus OrtRunOptionsGetRunLogVerbosityLevel( @Const OrtRunOptions options, IntPointer out);
public static native OrtStatus OrtRunOptionsGetRunLogVerbosityLevel( @Const OrtRunOptions options, IntBuffer out);
public static native OrtStatus OrtRunOptionsGetRunLogVerbosityLevel( @Const OrtRunOptions options, int[] out);
public static native OrtStatus OrtRunOptionsGetRunTag( @Const OrtRunOptions arg0, @Cast("const char**") PointerPointer out);
public static native OrtStatus OrtRunOptionsGetRunTag( @Const OrtRunOptions arg0, @Cast("const char**") @ByPtrPtr BytePointer out);
public static native OrtStatus OrtRunOptionsGetRunTag( @Const OrtRunOptions arg0, @Cast("const char**") @ByPtrPtr ByteBuffer out);
public static native OrtStatus OrtRunOptionsGetRunTag( @Const OrtRunOptions arg0, @Cast("const char**") @ByPtrPtr byte[] out);

// Set a flag so that any running OrtRun* calls that are using this instance of OrtRunOptions
// will exit as soon as possible if the flag is true.
// flag can be either 1 (true) or 0 (false)
public static native OrtStatus OrtRunOptionsSetTerminate( OrtRunOptions options, int flag);

/**
 * Create a tensor from an allocator. OrtReleaseValue will also release the buffer inside the output value
 * @param out Should be freed by calling OrtReleaseValue
 * @param type must be one of TENSOR_ELEMENT_DATA_TYPE_xxxx
 */
public static native OrtStatus OrtCreateTensorAsOrtValue( OrtAllocator allocator,
               @Cast("const int64_t*") LongPointer shape, @Cast("size_t") long shape_len, @Cast("ONNXTensorElementDataType") int type,
               @Cast("OrtValue**") PointerPointer out);
public static native OrtStatus OrtCreateTensorAsOrtValue( OrtAllocator allocator,
               @Cast("const int64_t*") LongPointer shape, @Cast("size_t") long shape_len, @Cast("ONNXTensorElementDataType") int type,
               @ByPtrPtr OrtValue out);
public static native OrtStatus OrtCreateTensorAsOrtValue( OrtAllocator allocator,
               @Cast("const int64_t*") LongBuffer shape, @Cast("size_t") long shape_len, @Cast("ONNXTensorElementDataType") int type,
               @ByPtrPtr OrtValue out);
public static native OrtStatus OrtCreateTensorAsOrtValue( OrtAllocator allocator,
               @Cast("const int64_t*") long[] shape, @Cast("size_t") long shape_len, @Cast("ONNXTensorElementDataType") int type,
               @ByPtrPtr OrtValue out);

/**
 * Create a tensor with user's buffer. You can fill the buffer either before calling this function or after.
 * p_data is owned by caller. OrtReleaseValue won't release p_data.
 * @param out Should be freed by calling OrtReleaseValue
 */
public static native OrtStatus OrtCreateTensorWithDataAsOrtValue( @Const OrtAllocatorInfo info,
               Pointer p_data, @Cast("size_t") long p_data_len, @Cast("const int64_t*") LongPointer shape, @Cast("size_t") long shape_len,
               @Cast("ONNXTensorElementDataType") int type, @Cast("OrtValue**") PointerPointer out);
public static native OrtStatus OrtCreateTensorWithDataAsOrtValue( @Const OrtAllocatorInfo info,
               Pointer p_data, @Cast("size_t") long p_data_len, @Cast("const int64_t*") LongPointer shape, @Cast("size_t") long shape_len,
               @Cast("ONNXTensorElementDataType") int type, @ByPtrPtr OrtValue out);
public static native OrtStatus OrtCreateTensorWithDataAsOrtValue( @Const OrtAllocatorInfo info,
               Pointer p_data, @Cast("size_t") long p_data_len, @Cast("const int64_t*") LongBuffer shape, @Cast("size_t") long shape_len,
               @Cast("ONNXTensorElementDataType") int type, @ByPtrPtr OrtValue out);
public static native OrtStatus OrtCreateTensorWithDataAsOrtValue( @Const OrtAllocatorInfo info,
               Pointer p_data, @Cast("size_t") long p_data_len, @Cast("const int64_t*") long[] shape, @Cast("size_t") long shape_len,
               @Cast("ONNXTensorElementDataType") int type, @ByPtrPtr OrtValue out);

// This function doesn't work with string tensor
// this is a no-copy method whose pointer is only valid until the backing OrtValue is free'd.
public static native OrtStatus OrtGetTensorMutableData( OrtValue value, @Cast("void**") PointerPointer out);
public static native OrtStatus OrtGetTensorMutableData( OrtValue value, @Cast("void**") @ByPtrPtr Pointer out);

/**
 * \Sets *out to 1 iff an OrtValue is a tensor, 0 otherwise
 */
public static native OrtStatus OrtIsTensor( @Const OrtValue value, IntPointer out);
public static native OrtStatus OrtIsTensor( @Const OrtValue value, IntBuffer out);
public static native OrtStatus OrtIsTensor( @Const OrtValue value, int[] out);

/**
 * @param value A tensor created from OrtCreateTensor... function.
 * @param s each A string array. Each string in this array must be null terminated.
 * @param s_len length of s
 */
public static native OrtStatus OrtFillStringTensor( OrtValue value, @Cast("const char*const*") PointerPointer s, @Cast("size_t") long s_len);
public static native OrtStatus OrtFillStringTensor( OrtValue value, @Cast("const char*const*") @ByPtrPtr BytePointer s, @Cast("size_t") long s_len);
public static native OrtStatus OrtFillStringTensor( OrtValue value, @Cast("const char*const*") @ByPtrPtr ByteBuffer s, @Cast("size_t") long s_len);
public static native OrtStatus OrtFillStringTensor( OrtValue value, @Cast("const char*const*") @ByPtrPtr byte[] s, @Cast("size_t") long s_len);
/**
 * @param value A tensor created from OrtCreateTensor... function.
 * @param len total data length, not including the trailing '\0' chars.
 */
public static native OrtStatus OrtGetStringTensorDataLength( @Const OrtValue value, @Cast("size_t*") SizeTPointer len);

/**
 * @param s string contents. Each string is NOT null-terminated.
 * @param value A tensor created from OrtCreateTensor... function.
 * @param s_len total data length, get it from OrtGetStringTensorDataLength
 */
public static native OrtStatus OrtGetStringTensorContent( @Const OrtValue value, Pointer s, @Cast("size_t") long s_len,
               @Cast("size_t*") SizeTPointer offsets, @Cast("size_t") long offsets_len);

/**
 * Create an OrtValue in CPU memory from a serialized TensorProto
 * @param input           serialized TensorProto object
 * @param input_len       length of 'input'.
 * @param input_file_path A local file path of where the input was loaded from. Can be NULL if the tensor proto doesn't
 *                        have any external data or it was loaded from current working dir. This path could be either a
 *                        relative path or an absolute path.
 * @param preallocated A preallocated buffer for the tensor. It should be allocated from CPU memory
 * @param preallocated_size Length of the preallocated buffer in bytes, can be computed from
 *          the OrtGetTensorMemSizeInBytesFromTensorProto function. This function will return an error if the
 *          preallocated_size is not enough.
 * @param out
 * @return
 */
public static native OrtStatus OrtTensorProtoToOrtValue( @Const Pointer input, int input_len,
               @Cast("const char*") BytePointer input_file_path, Pointer preallocated, @Cast("size_t") long preallocated_size,
               @Cast("OrtValue**") PointerPointer out, @Cast("OrtCallback**") PointerPointer deleter);
public static native OrtStatus OrtTensorProtoToOrtValue( @Const Pointer input, int input_len,
               @Cast("const char*") BytePointer input_file_path, Pointer preallocated, @Cast("size_t") long preallocated_size,
               @ByPtrPtr OrtValue out, @ByPtrPtr OrtCallback deleter);
public static native OrtStatus OrtTensorProtoToOrtValue( @Const Pointer input, int input_len,
               String input_file_path, Pointer preallocated, @Cast("size_t") long preallocated_size,
               @ByPtrPtr OrtValue out, @ByPtrPtr OrtCallback deleter);

/**
 *  f will be freed in this call
 */
public static native void OrtRunCallback( OrtCallback f);

/**
 * calculate the memory requirement for the OrtTensorProtoToOrtValue function
 */
public static native OrtStatus OrtGetTensorMemSizeInBytesFromTensorProto( @Const Pointer input, int input_len, @Cast("size_t") long alignment,
               @Cast("size_t*") SizeTPointer out);

/**
 * Don't free the 'out' value
 */
public static native OrtStatus OrtCastTypeInfoToTensorInfo( OrtTypeInfo arg0, @Cast("const OrtTensorTypeAndShapeInfo**") PointerPointer out);
public static native OrtStatus OrtCastTypeInfoToTensorInfo( OrtTypeInfo arg0, @Const @ByPtrPtr OrtTensorTypeAndShapeInfo out);

/**
 * Return OnnxType from OrtTypeInfo
 */
public static native OrtStatus OrtOnnxTypeFromTypeInfo( @Const OrtTypeInfo arg0, @Cast("ONNXType*") IntPointer out);
public static native OrtStatus OrtOnnxTypeFromTypeInfo( @Const OrtTypeInfo arg0, @Cast("ONNXType*") IntBuffer out);
public static native OrtStatus OrtOnnxTypeFromTypeInfo( @Const OrtTypeInfo arg0, @Cast("ONNXType*") int[] out);

/**
 * The 'out' value should be released by calling OrtReleaseTensorTypeAndShapeInfo
 */
public static native OrtStatus OrtCreateTensorTypeAndShapeInfo( @Cast("OrtTensorTypeAndShapeInfo**") PointerPointer out);
public static native OrtStatus OrtCreateTensorTypeAndShapeInfo( @ByPtrPtr OrtTensorTypeAndShapeInfo out);

public static native OrtStatus OrtSetTensorElementType( OrtTensorTypeAndShapeInfo arg0, @Cast("ONNXTensorElementDataType") int type);

/**
 * @param info Created from OrtCreateTensorTypeAndShapeInfo() function
 * @param dim_values An array with length of {@code dim_count}. Its elements can contain negative values.
 * @param dim_count length of dim_values
 */
public static native OrtStatus OrtSetDimensions( OrtTensorTypeAndShapeInfo info, @Cast("const int64_t*") LongPointer dim_values, @Cast("size_t") long dim_count);
public static native OrtStatus OrtSetDimensions( OrtTensorTypeAndShapeInfo info, @Cast("const int64_t*") LongBuffer dim_values, @Cast("size_t") long dim_count);
public static native OrtStatus OrtSetDimensions( OrtTensorTypeAndShapeInfo info, @Cast("const int64_t*") long[] dim_values, @Cast("size_t") long dim_count);

public static native OrtStatus OrtGetTensorElementType( @Const OrtTensorTypeAndShapeInfo arg0, @Cast("ONNXTensorElementDataType*") IntPointer out);
public static native OrtStatus OrtGetTensorElementType( @Const OrtTensorTypeAndShapeInfo arg0, @Cast("ONNXTensorElementDataType*") IntBuffer out);
public static native OrtStatus OrtGetTensorElementType( @Const OrtTensorTypeAndShapeInfo arg0, @Cast("ONNXTensorElementDataType*") int[] out);
public static native OrtStatus OrtGetDimensionsCount( @Const OrtTensorTypeAndShapeInfo info, @Cast("size_t*") SizeTPointer out);
public static native OrtStatus OrtGetDimensions( @Const OrtTensorTypeAndShapeInfo info, @Cast("int64_t*") LongPointer dim_values, @Cast("size_t") long dim_values_length);
public static native OrtStatus OrtGetDimensions( @Const OrtTensorTypeAndShapeInfo info, @Cast("int64_t*") LongBuffer dim_values, @Cast("size_t") long dim_values_length);
public static native OrtStatus OrtGetDimensions( @Const OrtTensorTypeAndShapeInfo info, @Cast("int64_t*") long[] dim_values, @Cast("size_t") long dim_values_length);

/**
 * Return the number of elements specified by the tensor shape.
 * Return a negative value if unknown (i.e., any dimension is negative.)
 * e.g.
 * [] -> 1
 * [1,3,4] -> 12
 * [2,0,4] -> 0
 * [-1,3,4] -> -1
 */
public static native OrtStatus OrtGetTensorShapeElementCount( @Const OrtTensorTypeAndShapeInfo info, @Cast("size_t*") SizeTPointer out);

/**
 * @param out Should be freed by OrtReleaseTensorTypeAndShapeInfo after use
 */
public static native OrtStatus OrtGetTensorTypeAndShape( @Const OrtValue value, @Cast("OrtTensorTypeAndShapeInfo**") PointerPointer out);
public static native OrtStatus OrtGetTensorTypeAndShape( @Const OrtValue value, @ByPtrPtr OrtTensorTypeAndShapeInfo out);

/**
 * Get the type information of an OrtValue
 * @param value
 * @param out The returned value should be freed by OrtReleaseTypeInfo after use
 */
public static native OrtStatus OrtGetTypeInfo( @Const OrtValue value, @Cast("OrtTypeInfo**") PointerPointer out);
public static native OrtStatus OrtGetTypeInfo( @Const OrtValue value, @ByPtrPtr OrtTypeInfo out);

public static native OrtStatus OrtGetValueType( @Const OrtValue value, @Cast("ONNXType*") IntPointer out);
public static native OrtStatus OrtGetValueType( @Const OrtValue value, @Cast("ONNXType*") IntBuffer out);
public static native OrtStatus OrtGetValueType( @Const OrtValue value, @Cast("ONNXType*") int[] out);

/** enum OrtAllocatorType */
public static final int
  OrtDeviceAllocator = 0,
  OrtArenaAllocator = 1;

/**
 * memory types for allocator, exec provider specific types should be extended in each provider
 * Whenever this struct is updated, please also update the MakeKey function in onnxruntime/core/framework/execution_provider.cc
*/
/** enum OrtMemType */
public static final int
  OrtMemTypeCPUInput = -2,              // Any CPU memory used by non-CPU execution provider
  OrtMemTypeCPUOutput = -1,             // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeCPU = OrtMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  OrtMemTypeDefault = 0;                // the default allocator for execution provider

public static native OrtStatus OrtCreateAllocatorInfo( @Cast("const char*") BytePointer name1, @Cast("OrtAllocatorType") int type, int id1, @Cast("OrtMemType") int mem_type1, @Cast("OrtAllocatorInfo**") PointerPointer out);
public static native OrtStatus OrtCreateAllocatorInfo( @Cast("const char*") BytePointer name1, @Cast("OrtAllocatorType") int type, int id1, @Cast("OrtMemType") int mem_type1, @ByPtrPtr OrtAllocatorInfo out);
public static native OrtStatus OrtCreateAllocatorInfo( String name1, @Cast("OrtAllocatorType") int type, int id1, @Cast("OrtMemType") int mem_type1, @ByPtrPtr OrtAllocatorInfo out);

/**
 * Convenience function for special case of OrtCreateAllocatorInfo, for the CPU allocator. Uses name = "Cpu" and id = 0.
 */
public static native OrtStatus OrtCreateCpuAllocatorInfo( @Cast("OrtAllocatorType") int type, @Cast("OrtMemType") int mem_type1, @Cast("OrtAllocatorInfo**") PointerPointer out);
public static native OrtStatus OrtCreateCpuAllocatorInfo( @Cast("OrtAllocatorType") int type, @Cast("OrtMemType") int mem_type1, @ByPtrPtr OrtAllocatorInfo out);

/**
 * Test if two allocation info are equal
 * \Sets 'out' to 0 if equal, -1 if not equal
 */
public static native OrtStatus OrtCompareAllocatorInfo( @Const OrtAllocatorInfo info1, @Const OrtAllocatorInfo info2, IntPointer out);
public static native OrtStatus OrtCompareAllocatorInfo( @Const OrtAllocatorInfo info1, @Const OrtAllocatorInfo info2, IntBuffer out);
public static native OrtStatus OrtCompareAllocatorInfo( @Const OrtAllocatorInfo info1, @Const OrtAllocatorInfo info2, int[] out);

/**
 * Do not free the returned value
 */
public static native OrtStatus OrtAllocatorInfoGetName( @Const OrtAllocatorInfo ptr, @Cast("const char**") PointerPointer out);
public static native OrtStatus OrtAllocatorInfoGetName( @Const OrtAllocatorInfo ptr, @Cast("const char**") @ByPtrPtr BytePointer out);
public static native OrtStatus OrtAllocatorInfoGetName( @Const OrtAllocatorInfo ptr, @Cast("const char**") @ByPtrPtr ByteBuffer out);
public static native OrtStatus OrtAllocatorInfoGetName( @Const OrtAllocatorInfo ptr, @Cast("const char**") @ByPtrPtr byte[] out);
public static native OrtStatus OrtAllocatorInfoGetId( @Const OrtAllocatorInfo ptr, IntPointer out);
public static native OrtStatus OrtAllocatorInfoGetId( @Const OrtAllocatorInfo ptr, IntBuffer out);
public static native OrtStatus OrtAllocatorInfoGetId( @Const OrtAllocatorInfo ptr, int[] out);
public static native OrtStatus OrtAllocatorInfoGetMemType( @Const OrtAllocatorInfo ptr, @Cast("OrtMemType*") IntPointer out);
public static native OrtStatus OrtAllocatorInfoGetMemType( @Const OrtAllocatorInfo ptr, @Cast("OrtMemType*") IntBuffer out);
public static native OrtStatus OrtAllocatorInfoGetMemType( @Const OrtAllocatorInfo ptr, @Cast("OrtMemType*") int[] out);
public static native OrtStatus OrtAllocatorInfoGetType( @Const OrtAllocatorInfo ptr, @Cast("OrtAllocatorType*") IntPointer out);
public static native OrtStatus OrtAllocatorInfoGetType( @Const OrtAllocatorInfo ptr, @Cast("OrtAllocatorType*") IntBuffer out);
public static native OrtStatus OrtAllocatorInfoGetType( @Const OrtAllocatorInfo ptr, @Cast("OrtAllocatorType*") int[] out);

public static native OrtStatus OrtAllocatorAlloc( OrtAllocator ptr, @Cast("size_t") long size, @Cast("void**") PointerPointer out);
public static native OrtStatus OrtAllocatorAlloc( OrtAllocator ptr, @Cast("size_t") long size, @Cast("void**") @ByPtrPtr Pointer out);
public static native OrtStatus OrtAllocatorFree( OrtAllocator ptr, Pointer p);
public static native OrtStatus OrtAllocatorGetInfo( @Const OrtAllocator ptr, @Cast("const OrtAllocatorInfo**") PointerPointer out);
public static native OrtStatus OrtAllocatorGetInfo( @Const OrtAllocator ptr, @Const @ByPtrPtr OrtAllocatorInfo out);

public static native OrtStatus OrtCreateDefaultAllocator( @Cast("OrtAllocator**") PointerPointer out);
public static native OrtStatus OrtCreateDefaultAllocator( @ByPtrPtr OrtAllocator out);

public static native @Cast("const char*") BytePointer OrtGetVersionString();
/**
 * @param msg A null-terminated string. Its content will be copied into the newly created OrtStatus
 */
public static native OrtStatus OrtCreateStatus( @Cast("OrtErrorCode") int code, @Cast("const char*") BytePointer msg);
public static native OrtStatus OrtCreateStatus( @Cast("OrtErrorCode") int code, String msg);

public static native @Cast("OrtErrorCode") int OrtGetErrorCode( @Const OrtStatus status);
/**
 * @param status must not be NULL
 * @return The error message inside the {@code status}. Do not free the returned value.
 */
public static native @Cast("const char*") BytePointer OrtGetErrorMessage( @Const OrtStatus status);

/**
   * APIs to support non-tensor types - map and sequence.
   * Currently only the following types are supported
   * Note: the following types should be kept in sync with data_types.h
   * Map types
   * =========
   * std::map<std::string, std::string>
   * std::map<std::string, int64_t>
   * std::map<std::string, float>
   * std::map<std::string, double>
   * std::map<int64_t, std::string>
   * std::map<int64_t, int64_t>
   * std::map<int64_t, float>
   * std::map<int64_t, double>
   * 
   * Sequence types
   * ==============
   * std::vector<std::string>
   * std::vector<int64_t>
   * std::vector<float>
   * std::vector<double>
   * std::vector<std::map<std::string, float>>
   * std::vector<std::map<int64_t, float>
   */

/**
   * If input OrtValue represents a map, you need to retrieve the keys and values
   * separately. Use index=0 to retrieve keys and index=1 to retrieve values.
   * If input OrtValue represents a sequence, use index to retrieve the index'th element
   * of the sequence.
   */
public static native OrtStatus OrtGetValue( @Const OrtValue value, int index, OrtAllocator allocator, @Cast("OrtValue**") PointerPointer out);
public static native OrtStatus OrtGetValue( @Const OrtValue value, int index, OrtAllocator allocator, @ByPtrPtr OrtValue out);

/**
   * Returns 2 for type map and N for sequence where N is the number of elements
   * in the sequence.
   */
public static native OrtStatus OrtGetValueCount( @Const OrtValue value, @Cast("size_t*") SizeTPointer out);

/**
   * To construct a map, use num_values = 2 and 'in' should be an arrary of 2 OrtValues
   * representing keys and values.
   * To construct a sequence, use num_values = N where N is the number of the elements in the
   * sequence. 'in' should be an arrary of N OrtValues.
   * \value_type should be either map or sequence.
   */
public static native OrtStatus OrtCreateValue( @Cast("OrtValue**") PointerPointer in, @Cast("size_t") long num_values, @Cast("ONNXType") int value_type,
               @Cast("OrtValue**") PointerPointer out);
public static native OrtStatus OrtCreateValue( @ByPtrPtr OrtValue in, @Cast("size_t") long num_values, @Cast("ONNXType") int value_type,
               @ByPtrPtr OrtValue out);
// Targeting ../OrtKernelInfo.java


// Targeting ../OrtKernelContext.java


// Targeting ../OrtCustomOpApi.java


// Targeting ../OrtCustomOp.java



/*
* Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
*/
public static native OrtStatus OrtCreateCustomOpDomain( @Cast("const char*") BytePointer domain, @Cast("OrtCustomOpDomain**") PointerPointer out);
public static native OrtStatus OrtCreateCustomOpDomain( @Cast("const char*") BytePointer domain, @ByPtrPtr OrtCustomOpDomain out);
public static native OrtStatus OrtCreateCustomOpDomain( String domain, @ByPtrPtr OrtCustomOpDomain out);

/*
 * Add custom ops to the OrtCustomOpDomain
 *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
*/
public static native OrtStatus OrtCustomOpDomain_Add( OrtCustomOpDomain custom_op_domain, OrtCustomOp op);

/*
 * Add a custom op domain to the OrtSessionOptions
 *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
*/
public static native OrtStatus OrtAddCustomOpDomain( OrtSessionOptions options, OrtCustomOpDomain custom_op_domain);
/*
 * END EXPERIMENTAL
*/

// #ifdef __cplusplus
// #endif


}