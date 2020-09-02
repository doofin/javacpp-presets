/*
 * Copyright (C) 2020 Samuel Audet, Eduardo Gonzalez
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.libtorch.presets;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

@Properties(inherit = javacpp.class, target = "org.bytedeco.libtorch", global = "org.bytedeco.libtorch.global.libtorch", value = {
        @Platform(
                value = {"linux", "macosx", "windows"},
                //define = {},
                compiler = "cpp14",
                define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std", "LEGACY_CONTIGUOUS_MEMORY_FORMAT c10::get_contiguous_memory_format()"},
                include = {
                        //"c10/util/UniqueVoidPtr.h",
                        "c10/core/DeviceType.h",
                        "c10/core/Device.h",
                        "c10/core/Allocator.h",
                        "c10/core/QEngine.h",
                        "c10/core/Backend.h",
                        "c10/core/ScalarType.h",
                        "c10/core/DispatchKey.h",
                        "c10/core/DispatchKeySet.h",
                        "c10/core/DefaultDtype.h",
                        "c10/core/Stream.h",
                        "c10/core/impl/DeviceGuardImplInterface.h",
                        "c10/core/GeneratorImpl.h",
                        "c10/core/MemoryFormat.h",
                        "c10/core/Storage.h",
                        "c10/core/Layout.h",
                        //"c10/core/Scalar.h",
                        "c10/core/ScalarType.h",
                        //"c10/util/Half.h", // Parse Error
                        //"c10/util/BFloat16.h", // Parse Error
                        "c10/core/QScheme.h",
                        "c10/core/TensorImpl.h",
                        "ATen/detail/HIPHooksInterface.h",
                        "ATen/detail/CUDAHooksInterface.h",
                        "ATen/detail/CPUGuardImpl.h",
                        //"c10/util/ArrayRef.h",
                        //"ATen/core/interned_strings.h", // Crazy Macro
                        "ATen/core/Dimname.h",

                        "ATen/core/DeprecatedTypePropertiesRegistry.h",
                        "ATen/core/Generator.h",
                        "ATen/Context.h",
                        "ATen/ATen.h",
                        "ATen/Tensor.h",
                        "ATen/core/NamedTensor.h",
                        "ATen/core/TensorBody.h",
                        "torch/script.h",
                },
                link = {"torch"},
                resource = {"include", "lib"}
        )
})
public class libtorch implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libtorch"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::hash<c10::Device>").javaNames("DeviceMap"));
        infoMap.put(new Info("std::hash<c10::DeviceType>").javaNames("DeviceTypeMap"));
        infoMap.put(new Info("std::function<void(void*)>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("C10_API", "C10_NODISCARD", "CAFFE2_API").skip());
        infoMap.put(new Info("std::size_t").cast().javaNames("long"));
        infoMap.put(new Info("c10::DebugInfoBase").javaNames("Pointer"));
        infoMap.put(new Info("c10::DeleterFnPtr").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("DeleterFnPtr").cast().pointerTypes("Pointer"));
        //infoMap.put(new Info("c10::detail::UniqueVoidPtr").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::unique_ptr<void, c10::DeleterFnPtr>&&").javaText("@StdMove @Cast({\"\", \"std::unique_ptr<void, DeleterFnPtr>*\"} Pointer"));
        infoMap.put(new Info("c10::DataPtr").skip());
        infoMap.put(new Info("c10::InefficientStdFunctionContext").skip());
        infoMap.put(new Info("c10::Allocator").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::Allocator").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("c10::kNoQEngine").skip());
        infoMap.put(new Info("c10::kFBGEMM").skip());
        infoMap.put(new Info("c10::kQNNPACK").skip());

        infoMap.put(new Info("c10::optional<c10::ScalarType>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<int64_t>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::ScalarType").enumerate());
        infoMap.put(new Info("c10::QEngine").enumerate());
        infoMap.put(new Info("c10::Backend").enumerate());
        infoMap.put(new Info("c10::DispatchKey").skip());
        infoMap.put(new Info("c10::DispatchKeySet").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::DispatchKeySet").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("at::DeprecatedTypePropertiesRegistry").skip());


        infoMap.put(new Info("c10::intrusive_ptr<c10::GeneratorImpl>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("PyObject").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("std::hash<c10::DispatchKey>").javaNames("DispatchKeyMap"));
        infoMap.put(new Info("std::initializer_list<c10::DispatchKey>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::DeviceIndex").cast().valueTypes("short"));
        infoMap.put(new Info("at::DeviceIndex").cast().valueTypes("short"));

        infoMap.put(new Info("std::hash<c10::Stream>").javaNames("StreamMap"));

        infoMap.put(new Info("c10::EventFlag").enumerate());
        infoMap.put(new Info("std::vector<at::QEngine>").pointerTypes("QEngineVector").define());
        infoMap.put(new Info("c10::ToVectorint64_t").skip());

        infoMap.put(new Info("at::Generator::key_set").skip());
        infoMap.put(new Info("at::Generator::mutex").skip());

        infoMap.put(new Info("c10::GeneratorImpl::mutex_").skip());

        infoMap.put(new Info("c10::intrusive_ptr_target").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::string_view").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("at::CUDAHooksInterface::initCUDA").skip());
        infoMap.put(new Info("at::CUDAHooksInterface::getCUDADeviceAllocator").skip());
        infoMap.put(new Info("at::CUDAHooksInterface::getPinnedMemoryAllocator").skip());

        infoMap.put(new Info("at::HIPHooksInterface::initHIP").skip());
        infoMap.put(new Info("at::HIPHooksInterface::getPinnedMemoryAllocator").skip());

        infoMap.put(new Info("c10::intrusive_ptr<TensorImpl,UndefinedTensorImpl>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::intrusive_ptr<TensorImpl,at::UndefinedTensorImpl>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::intrusive_ptr<at::TensorImpl,UndefinedTensorImpl>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::intrusive_ptr<at::TensorImpl,at::UndefinedTensorImpl>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("at::Tensor").skipDefaults());
        infoMap.put(new Info("at::TensorImpl").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::IntArrayRef").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::IntArrayRef").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::Tensor::toString").javaNames("toBytePointer"));
        infoMap.put(new Info("at::Tensor::data").skip());
        infoMap.put(new Info("at::Tensor::packed_accessor").skip());
        infoMap.put(new Info("at::Tensor::index").skip());
        infoMap.put(new Info("at::Tensor::index_put_").skip());
        infoMap.put(new Info("at::Tensor::resize_").skip());

        infoMap.put(new Info("optional<at::DimnameList>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::optional<at::DimnameList>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("optional<at::Dimname>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::optional<at::Dimname>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::hash<c10::Symbol>").javaNames("SymbolMap"));

        infoMap.put(new Info("c10::ArrayRef<at::Dimname>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::MemoryFormat").enumerate());
        infoMap.put(new Info("at::MemoryFormat").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::_keys").enumerate());

        infoMap.put(new Info("c10::Layout").enumerate());
        infoMap.put(new Info("c10::QScheme").enumerate());


        infoMap.put(new Info("c10::optional<bool>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<c10::MemoryFormat>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<MemoryFormat>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::MemoryFormat>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::Scalar>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<Scalar>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::optional<c10::Device>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::intrusive_ptr<c10::TensorImpl>").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("c10::optional<at::DimnameList>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::Generator>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::ScalarType>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<ScalarType>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("ArrayRef<at::Tensor>").cast().pointerTypes("Pointer"));
        //infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor>").javaNames("TensorTensorTuple"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::tuple<at::Tensor,at::Tensor,at::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::intrusive_ptr<StorageImpl>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::intrusive_ptr<at::StorageImpl>").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("c10::Storage").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::Storage").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::TensorImpl::set_storage_keep_dtype").skip());
        infoMap.put(new Info("c10::TensorImpl::set_storage_and_dtype").skip());
        infoMap.put(new Info("at::Tensor").purify());

        infoMap.put(new Info("at::DataPtr").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::PlacementDeleteContext").skip());

        infoMap.put(new Info("c10::NamedTensorMetaInterface").skip());
        infoMap.put(new Info("at::NamedTensorMeta").skip());

        infoMap.put(new Info("c10::kPerTensorAffine").skip());
        infoMap.put(new Info("c10::kPerChannelAffine").skip());
        infoMap.put(new Info("c10::kPerTensorSymmetric").skip());
        infoMap.put(new Info("c10::kPerChannelSymmetric").skip());
        infoMap.put(new Info("c10::kStrided").skip());
        infoMap.put(new Info("c10::kSparse").skip());
        infoMap.put(new Info("c10::kMkldnn").skip());

        infoMap.put(new Info("ArrayRef<int>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("LEGACY_CONTIGUOUS_MEMORY_FORMAT").skip());
    }

    @Namespace("c10") public enum DispatchKey {

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // This is not a "real" tensor id, but it exists to give us a "nullopt"
        // element we can return for cases when a DispatchKeySet contains no elements.
        // You can think a more semantically accurate definition of DispatchKey is:
        //
        //    using DispatchKey = optional<RealDispatchKey>
        //
        // and Undefined == nullopt.  We didn't actually represent
        // it this way because optional<RealDispatchKey> would take two
        // words, when DispatchKey fits in eight bits.

        Undefined((byte)0),

        // Define an alias for Undefined to represent CatchAll (long term
        // this will get eliminated, but for now it's convenient)
        CatchAll((byte)0),

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // A "backend" is colloquially used to refer to handlers for dispatch
        // which actually implement the numerics of an operation in question.
        //
        // Due to the nature of the enum, these backends are specified in
        // an ordered way, but for most backends this order is not semantically
        // meaningful (e.g., it's valid to reorder these backends without changing
        // semantics).  The only situation when backend ordering is meaningful
        // is when the backend participates in multiple dispatch with another
        // backend; e.g., CPU and SparseCPU (sparse must have
        // higher priority).

        // Here are backends which you think of as traditionally specifying
        // how to implement operations on some device.
        CPU((byte)1), // registered at build/aten/src/ATen/CPUType.cpp
        CUDA((byte)2), // registered at build/aten/src/ATen/CUDAType.cpp
        HIP((byte)3), // NB: I think this is not actually used, due to Note [Masquerading as
        // CUDA]
        FPGA((byte)4), // Xilinx support lives out of tree at https://gitlab.com/pytorch-complex/vitis_kernels
        MSNPU((byte)5), // unused externally, but tested at
        // test/cpp_extensions/msnpu_extension.cpp
        XLA((byte)6), // lives out of tree at https://github.com/pytorch/xla
        Vulkan((byte)7),

        // These are Caffe2 device types which we grandfathered into
        // DispatchKey.
        // TODO: Caffe2-only DispatchKeys actually should be removed from this enum
        // and just simply be undispatchable.
        MKLDNN((byte)8), // (MKLDNN is treated as another "device" in Caffe2)
        OpenGL((byte)9),
        OpenCL((byte)10),
        IDEEP((byte)11),

        // Here are backends which specify more specialized operators
        // based on the dtype of the tensor.
        QuantizedCPU((byte)12), // registered at build/aten/src/ATen/QuantizedCPUType.cpp
        QuantizedCUDA((byte)13), // registered at build/aten/src/ATen/QuantizedCUDAType.cpp
        ComplexCPU((byte)14), // lives out of tree at
        // https://gitlab.com/pytorch-complex/pytorch-cpu-strided-complex
        ComplexCUDA((byte)15), // and
        // https://gitlab.com/pytorch-complex/pytorch-cuda-strided-complex
        // tested at test/cpp_extensions/complex_registration_extension.cpp
        // TODO: Remove Complex dispatch keys when Complex is moved in tree

        // This backend is to support custom RNGs; it lets you go
        // to a different kernel if you pass in a generator that is not a
        // traditional CPUGeneratorImpl/CUDAGeneratorImpl.  To make use of this
        // key:
        //  1) set it as a second parameter of at::Generator constructor call in
        //     the user-defined PRNG class.
        //  2) use it as a dispatch key while registering custom kernels
        //     (templatized kernels specialized for user-defined PRNG class)
        // intended for out of tree use; tested by aten/src/ATen/test/rng_test.cpp
        CustomRNGKeyId((byte)16),

        // Here are backends which specify more specialized operators
        // based on the layout of the tensor.  Note that the sparse backends
        // are one case where ordering matters: sparse multi-dispatches with
        // the corresponding dense tensors, and must be handled before them.
        MkldnnCPU((byte)17), // registered at build/aten/src/ATen/MkldnnCPUType.cpp
        // NB: not to be confused with MKLDNN, which is Caffe2 only
        SparseCPU((byte)18), // registered at build/aten/src/ATen/SparseCPUType.cpp
        SparseCUDA((byte)19), // registered at build/aten/src/ATen/SparseCUDAType.cpp
        SparseHIP((byte)20), // TODO: I think this is not actually used, due to Note
        // [Masquerading as CUDA]

        // Here are reserved backends for user-defined backends, see Note [Private use
        // DispatchKey]
        // To see some example about how to use this, check out MSNPU
        PrivateUse1((byte)21),
        PrivateUse2((byte)22),
        PrivateUse3((byte)23),

        // The meta function characterizes how an operation affects the metadata of a
        // tensor (shape, dtype) without doing any of the actual computation.  A
        // meta tensor can be used to dry run operators without actually doing
        // any computation, e.g., add on two meta tensors would give you another
        // meta tensor with the output shape and dtype, but wouldn't actually
        // add anything.  A meta implementation typically would look something like:
        //
        //  Tensor meta::add(const Tensor& self, const Tensor& other) {
        //    TORCH_CHECK(self.size().equals(other.size()));
        //    return at::empty_like(self, self.size());
        //  }
        //
        // The meta function would get invoked if you ran an operator passing
        // in meta tensors.  The call stack in such a case would look something like
        // this:
        //
        //  at::add(x: Meta, y: Meta) {
        //    return [dispatch] meta::add(x: Meta, y: Meta) {
        //      output_shape = ...
        //      [dispatch] meta::empty(output_shape) {
        //        return ... meta tensor with output_shape but no data allocated ...
        //      }
        //    }
        //  }
        //
        // Meta functions have an important secondary function, which is they can
        // be used as tensor "allocators".  A typical backend implementation should
        // be implemented in this way:
        //
        //  Tensor cpu::add(const Tensor& self, const Tensor& other) {
        //    Tensor result = meta::add(self, other);
        //    // ... do the actual computation into result ...
        //    return result;
        //  }
        //
        // In this case, the internal at::empty_like invocation would dispatch to the
        // CPU factory function, not the meta factory function.  The call stack in
        // this case looks like:
        //
        //  at::add(x: CPU, y: CPU) {
        //    return [dispatch] cpu::add(x: CPU, y: CPU) {
        //      output = [direct] meta::add(x: CPU, y: CPU) {
        //        output_shape = ...
        //        [dispatch] cpu::empty(output_shape)
        //      }
        //      ... compute on output ...
        //      return output;
        //    }
        //  }
        //
        Meta((byte)24),

        // In some situations, it is not immediately obvious what the correct
        // backend for function is, because the function in question doesn't
        // have any "tensor" arguments.  In this case, a BackendSelect function
        // can be registered to implement the custom determination of the
        // correct backend.
        BackendSelect((byte)25),

        // The named dispatch key is set for any tensors with named dimensions.
        // Although we have a dispatch key for named tensors, for historical reasons,
        // this dispatch key doesn't do any of the substantive functionality for named
        // tensor (though, hypothetically, it could!)  At the moment, it's just
        // responsible for letting us give good error messages when operations
        // don't support named tensors.
        //
        // NB: If you ever consider moving named tensor functionality into
        // this dispatch key, note that it might be necessary add another dispatch
        // key that triggers before composite operators, in case a composite operator
        // has named dimension propagation that doesn't match that of its
        // constituent parts.
        Named((byte)26),

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AUTOGRAD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // All backends are oblivious to autograd; autograd is handled as a
        // layer which happens on top of all backends.  It inspects the autograd
        // metadata of all inputs, determines what autograd metadata should be
        // constructed by the output, and otherwise defers to the backend to
        // actually do the numeric computation.  Autograd contains
        // the bulk of this logic.
        Autograd((byte)27),

        Profiler((byte)28),

        Tracer((byte)29),

        // Pre-autograd dispatch keys allow backends to override the autograd behavior
        // (aka Autograd) for operators which have a Variable kernel
        // already registered.  For example, XLA wants to define autograd for
        // einsum directly.  Registering a custom autograd implementation at the
        // XLA key won't work because we process Autograd
        // before XLA.  This key has higher priority and gets processed
        // first.  You generally should NOT redispatch after handling autograd
        // here (since that would result in execution of the Autograd
        // operator, which you're trying to skip).  In PreAutograd implementations,
        // you are responsible for handling autograd yourself, or deferring to other
        // operators which support autograd.
        XLAPreAutograd((byte)30),

        // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
        // and inputs are saved for backward in the post-autocast type.
        Autocast((byte)31),

        // Here are some reserved pre-autograd keys for user-defined backends, see
        // Note [Private use DispatchKey]
        PrivateUse1_PreAutograd((byte)32),
        PrivateUse2_PreAutograd((byte)33),
        PrivateUse3_PreAutograd((byte)34),

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // There are a number of alternative modes which may want to handle before
        // autograd; for example, error checking, tracing, profiling or vmap.  They
        // go here.

        // This is the dispatch key for BatchedTensorImpl, which is used to implement
        // batching rules for vmap.
        Batched((byte)35),

        // TESTING: This is intended to be a generic testing tensor type id.
        // Don't use it for anything real; its only acceptable use is within a single
        // process test.  Use it by creating a TensorImpl with this DispatchKey, and
        // then registering operators to operate on this type id.  See
        // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
        TESTING_ONLY_GenericWrapper((byte)36),

        // TESTING: This is intended to be a generic testing tensor type id.
        // Don't use it for anything real; its only acceptable use is within a ingle
        // process test.  Use it by toggling the mode on and off via
        // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
        // to operate on this type id.  See
        // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
        // for a usage example
        TESTING_ONLY_GenericMode((byte)37),

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        NumDispatchKeys((byte)38), // Sentinel

        // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
        // The aliases exist for backwards compatibility reasons, they shouldn't
        // be used
        CPUTensorId((byte)1),
        CUDATensorId((byte)2);

        public final byte value;
        private DispatchKey(byte v) { this.value = v; }
        private DispatchKey(DispatchKey e) { this.value = e.value; }
        public DispatchKey intern() { for (DispatchKey e : values()) if (e.value == value) return e; return this; }
        @Override public String toString() { return intern().name(); }
    }

    // A Symbol is like an interned string, but with a little extra
    // structure; it is namespaced via SymbolNamespace and the resulting
    // intern pointers support efficient namespace testing.
    @Namespace("c10") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
    static public class Symbol extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Symbol(Pointer p) { super(p); }

        public Symbol() { super((Pointer)null); allocate(); }
        private native void allocate();
        public Symbol(@Cast("c10::unique_t") int uniq) { super((Pointer)null); allocate(uniq); }
        private native void allocate(@Cast("c10::unique_t") int uniq);

        // Get a Symbol for a qualified string like "attr::bar"
        public static native @ByVal
        Symbol fromQualString(@StdString BytePointer s);
        public static native @ByVal
        Symbol fromQualString(@StdString String s);

        // Get a Symbol from a domain and an unqualified string like "org.pytorch.attr" and "bar"
        public static native @ByVal
        Symbol fromDomainAndUnqualString(@StdString BytePointer d, @StdString BytePointer s);
        public static native @ByVal
        Symbol fromDomainAndUnqualString(@StdString String d, @StdString String s);

        // Constructors for our various namespaced strings.  This will construct
        // the appropriate namespaced string, e.g., "attr::foo" for the
        // argument "foo", and then attempt to intern it.  DO NOT USE THIS
        // with a string literal; attr::foo should be available in that case
        // (and if it's not, you should add it to the built-ins list above.)
        public static native @ByVal
        Symbol attr(@StdString BytePointer s);
        public static native @ByVal
        Symbol attr(@StdString String s);
        public static native @ByVal
        Symbol aten(@StdString BytePointer s);
        public static native @ByVal
        Symbol aten(@StdString String s);
        public static native @ByVal
        Symbol onnx(@StdString BytePointer s);
        public static native @ByVal
        Symbol onnx(@StdString String s);
        public static native @ByVal
        Symbol prim(@StdString BytePointer s);
        public static native @ByVal
        Symbol prim(@StdString String s);
        public static native @ByVal
        Symbol user(@StdString BytePointer s);
        public static native @ByVal
        Symbol user(@StdString String s);
        public static native @ByVal
        Symbol caffe2(@StdString BytePointer s);
        public static native @ByVal
        Symbol caffe2(@StdString String s);
        public static native @ByVal
        Symbol dimname(@StdString BytePointer s);
        public static native @ByVal
        Symbol dimname(@StdString String s);
        // TODO: eliminate me
        public static native @ByVal
        Symbol scope(@StdString BytePointer s);
        public static native @ByVal
        Symbol scope(@StdString String s);

        public native @Cast("bool") boolean is_attr();
        public native @Cast("bool") boolean is_aten();
        public native @Cast("bool") boolean is_prim();
        public native @Cast("bool") boolean is_onnx();
        public native @Cast("bool") boolean is_user();
        public native @Cast("bool") boolean is_caffe2();
        public native @Cast("bool") boolean is_dimname();

        // So we can switch on this
        public native @Cast("c10::unique_t") @Name("operator c10::unique_t") int asInt();

        public native @ByVal
        Symbol ns();

        // Give a string corresponding to the unqualified version of this name, e.g.,
        // "mm". Use this in a context where the intended namespace of the string is
        // obvious; this is a *lossy* conversion.
        public native @Cast("const char*") BytePointer toUnqualString();

        // Give a string corresponding to the qualified version of this name,
        // e.g., "aten::mm".  This string format is made available to Python bindings
        // (so we know how to parse it.)
        public native @Cast("const char*") BytePointer toQualString();

        // This describes a symbol in a case where humans read it.  At the moment it's
        // the same as toQualString.  This has to be a const char* returned because
        // a lot of printf style macros use it.
        public native @Cast("const char*") BytePointer toDisplayString();

        // Give a string corresponding to the domain name for the symbol,
        // e.g., "org.pytorch.aten".
        public native @StdString BytePointer domainString();
    }

    @Namespace("at") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
    static public class AtScalar extends Scalar {}

    @Namespace("c10") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
    static public class Scalar extends Pointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public Scalar(Pointer p) { super(p); }

        public Scalar() { super((Pointer)null); allocate(); }
        private native void allocate();

// #define DEFINE_IMPLICIT_CTOR(type, name)
//   Scalar(type vv) : Scalar(vv, true) { }

        public Scalar(@Cast("uint8_t") byte vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@Cast("uint8_t") byte vv);
        public Scalar(short vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(short vv);
        public Scalar(int vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(int vv);
        public Scalar(@Cast("int64_t") long vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@Cast("int64_t") long vv);
        public Scalar(float vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(float vv);
        public Scalar(double vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(double vv);
        // TODO: remove the std::complex below
        public Scalar(@ByVal @Cast("std::complex<float>*") FloatPointer vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<float>*") FloatPointer vv);
        public Scalar(@ByVal @Cast("std::complex<float>*") FloatBuffer vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<float>*") FloatBuffer vv);
        public Scalar(@ByVal @Cast("std::complex<float>*") float[] vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<float>*") float[] vv);
        public Scalar(@ByVal @Cast("std::complex<double>*") DoublePointer vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<double>*") DoublePointer vv);
        public Scalar(@ByVal @Cast("std::complex<double>*") DoubleBuffer vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<double>*") DoubleBuffer vv);
        public Scalar(@ByVal @Cast("std::complex<double>*") double[] vv) { super((Pointer)null); allocate(vv); }
        private native void allocate(@ByVal @Cast("std::complex<double>*") double[] vv);

// #undef DEFINE_IMPLICIT_CTOR

        // Value* is both implicitly convertible to SymbolicVariable and bool which
        // causes ambiguosity error. Specialized constructor for bool resolves this
        // problem.

// #define DEFINE_ACCESSOR(type, name)
//   type to##name() const {
//     if (Tag::HAS_d == tag) {
//       return checked_convert<type, double>(v.d, #type);
//     } else if (Tag::HAS_z == tag) {
//       return checked_convert<type, c10::complex<double>>(
//           v.z, #type);
//     } if (Tag::HAS_b == tag) {
//       return checked_convert<type, bool>(v.i, #type);
//     } else {
//       return checked_convert<type, int64_t>(v.i, #type);
//     }
//   }

        // TODO: Support ComplexHalf accessor
        public native @Cast("uint8_t") byte toByte();
        public native byte toChar();
        public native short toShort();
        public native int toInt();
        public native @Cast("int64_t") long toLong();

        public native float toFloat();
        public native double toDouble();
        public native @Cast("bool") boolean toBool();

        // also support scalar.to<int64_t>();

        // #undef DEFINE_ACCESSOR
        public native @Cast("bool") boolean isFloatingPoint();

        public native @Cast("bool") boolean isIntegral();
        public native @Cast("bool") boolean isIntegral(@Cast("bool") boolean includeBool);

        public native @Cast("bool") boolean isComplex();
        public native @Cast("bool") boolean isBoolean();

        public native @ByVal @Name("operator -")
        Scalar subtract();

        //public native @ByVal Half toHalf();
        //public native @ByVal BFloat16 toBFloat16();
        //public native ScalarType type();
    }
}
