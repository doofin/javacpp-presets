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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

import java.io.*;
import java.nio.Buffer;
import java.nio.LongBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;


@Properties(inherit = javacpp.class, target = "org.bytedeco.libtorch", global = "org.bytedeco.libtorch.global.libtorch", value = {
        @Platform(
                value = {"linux", "macosx", "windows"},
                //define = {},
                compiler = "cpp14",
                define = {"SHARED_PTR_NAMESPACE std", "UNIQUE_PTR_NAMESPACE std" },
                include = {
                    "c10/core/QEngine.h",
                    "c10/core/DeviceType.h",
                    "c10/core/Device.h",
                    "c10/core/ScalarType.h",
                    "c10/core/Scalar.h",
                    "c10/core/Stream.h",

                    //"c10/util/BFloat16.h", // Parse error EOF
                    //"ATen/core/TensorBody.h", // <- Absolute craziness.  So implemented manually below.
                    "ATen/TensorGeometry.h",
                    //"c10/core/Scalar.h",
                    //"c10/util/ArrayRef.h", // Parse error

                    //"torch/csrc/api/include/torch/ordered_dict.h", // Parse error
                    "torch/csrc/api/include/torch/expanding_array.h", // Does nothing?

                    "torch/csrc/api/include/torch/cuda.h",
                    "torch/csrc/api/include/torch/data.h",
                    "torch/csrc/api/include/torch/enum.h",
                    "torch/csrc/api/include/torch/jit.h",

                    "torch/csrc/api/include/torch/nn.h",
                        "torch/csrc/api/include/torch/nn/cloneable.h",
                        "torch/csrc/api/include/torch/nn/functional.h",
                            "torch/csrc/api/include/torch/nn/functional/batchnorm.h",
                                "torch/csrc/api/include/torch/nn/options/batchnorm.h",

                            "torch/csrc/api/include/torch/nn/functional/conv.h",
                                "torch/csrc/api/include/torch/nn/options/conv.h",

                            "torch/csrc/api/include/torch/nn/functional/distance.h",
                                "torch/csrc/api/include/torch/nn/options/distance.h",

                            "torch/csrc/api/include/torch/nn/functional/dropout.h",
                                "torch/csrc/api/include/torch/nn/options/dropout.h",

                            "torch/csrc/api/include/torch/nn/functional/embedding.h",
                                "torch/csrc/api/include/torch/nn/options/embedding.h",

                            "torch/csrc/api/include/torch/nn/functional/fold.h",
                                "torch/csrc/api/include/torch/nn/options/fold.h",

                            "torch/csrc/api/include/torch/nn/functional/linear.h",
                                "torch/csrc/api/include/torch/nn/options/linear.h",

                            "torch/csrc/api/include/torch/nn/functional/loss.h",
                                "torch/csrc/api/include/torch/nn/options/loss.h",

                            "torch/csrc/api/include/torch/nn/functional/normalization.h",
                                "torch/csrc/api/include/torch/nn/options/normalization.h",

                            "torch/csrc/api/include/torch/nn/functional/padding.h",
                                "torch/csrc/api/include/torch/nn/options/padding.h",

                            "torch/csrc/api/include/torch/nn/functional/pixelshuffle.h",
                                "torch/csrc/api/include/torch/nn/options/pixelshuffle.h",

                            "torch/csrc/api/include/torch/nn/functional/pooling.h",
                                "torch/csrc/api/include/torch/nn/options/pooling.h",

                            "torch/csrc/api/include/torch/nn/functional/upsampling.h",
                                "torch/csrc/api/include/torch/nn/options/upsampling.h",

                            "torch/csrc/api/include/torch/nn/functional/vision.h",
                                "torch/csrc/api/include/torch/nn/options/vision.h",

                            "torch/csrc/api/include/torch/nn/functional/instancenorm.h",
                                "torch/csrc/api/include/torch/nn/options/instancenorm.h",

                        "torch/csrc/api/include/torch/nn/init.h",
                        "torch/csrc/api/include/torch/nn/module.h",
                        "torch/csrc/api/include/torch/nn/modules.h",
                            "torch/csrc/api/include/torch/nn/modules/linear.h",
                            "torch/csrc/api/include/torch/nn/options/linear.h",

                        "torch/csrc/api/include/torch/nn/options.h",
                        "torch/csrc/api/include/torch/nn/pimpl.h",
                        "torch/csrc/api/include/torch/nn/utils.h",
                    "torch/csrc/api/include/torch/optim.h",
                        "torch/csrc/api/include/torch/optim/adagrad.h",
                        "torch/csrc/api/include/torch/optim/adam.h",
                        "torch/csrc/api/include/torch/optim/adamw.h",
                        "torch/csrc/api/include/torch/optim/lbfgs.h",
                        "torch/csrc/api/include/torch/optim/optimizer.h",
                        "torch/csrc/api/include/torch/optim/rmsprop.h",
                        "torch/csrc/api/include/torch/optim/sgd.h",

                    "torch/csrc/api/include/torch/serialize.h",
                    "torch/csrc/api/include/torch/types.h",
                    "torch/csrc/api/include/torch/utils.h",
                    "torch/csrc/api/include/torch/autograd.h",
                        "torch/csrc/autograd/generated/Functions.h",
                        "torch/csrc/autograd/generated/VariableType.h",
                        "torch/csrc/autograd/variable.h",
                        "torch/csrc/autograd/generated/variable_factories.h",
                            "torch/csrc/api/include/torch/detail/TensorDataContainer.h",
                        "torch/csrc/autograd/function.h",
                        "torch/csrc/autograd/input_metadata.h",
                        "torch/csrc/autograd/anomaly_mode.h"
                },
                link = {"torch"},
                resource = {"include", "lib"}
        )
})
public class libtorch implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "libtorch"); }

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("std::size_t").cast().javaNames("long"));

        infoMap.put(new Info("TORCH_API").skip());
        infoMap.put(new Info("TORCH_ARG").skip());
        infoMap.put(new Info("C10_API").skip());


        infoMap.put(new Info("torch::kUInt8").skip());
        infoMap.put(new Info("torch::kInt8").skip());
        infoMap.put(new Info("torch::kInt16").skip());
        infoMap.put(new Info("torch::kInt32").skip());
        infoMap.put(new Info("torch::kInt64").skip());
        infoMap.put(new Info("torch::kFloat16").skip());
        infoMap.put(new Info("torch::kFloat32").skip());
        infoMap.put(new Info("torch::kFloat64").skip());

        infoMap.put(new Info("torch::kU8").skip());
        infoMap.put(new Info("torch::kI8").skip());
        infoMap.put(new Info("torch::kI16").skip());
        infoMap.put(new Info("torch::kI32").skip());
        infoMap.put(new Info("torch::kI64").skip());
        infoMap.put(new Info("torch::kF16").skip());
        infoMap.put(new Info("torch::kF32").skip());
        infoMap.put(new Info("torch::kF64").skip());

        //infoMap.put(new Info("torch::Tensor").cast().pointerTypes("Pointer"));
        //infoMap.put(new Info("std::tuple<Tensor,Tensor>").skip());
        infoMap.put(new Info("c10::optional<torch::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<c10::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<Tensor>").skip());

        infoMap.put(new Info("torch::OrderedDict<std::string,Tensor>").skip());//.javaNames("OrderedTensorMap"));

        infoMap.put(new Info("c10::kNoQEngine").skip());
        infoMap.put(new Info("c10::kFBGEMM").skip());
        infoMap.put(new Info("c10::kQNNPACK").skip());


        infoMap.put(new Info("LEGACY_CONTIGUOUS_MEMORY_FORMAT").skip());

        infoMap.put(new Info("torch::optim::Optimizer").purify());

        //infoMap.put(new Info("torch::optim::SGD").purify());
        //infoMap.put(new Info("torch::optim::RMSprop").purify());

        infoMap.put(new Info("torch::optim::OptimizerParamGroup").skip());//.cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<AdamWOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<SGDOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<LGBFSOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::enable_shared_from_this<Module>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::optional<Device>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::function<void(Module &)>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::OrderedDict<std::string,torch::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::shared_ptr<torch::nn::Module>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::OrderedDict<std::string,std::shared_ptr<torch::nn::Module> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("ska::flat_hash_map<std::string,std::unique_ptr<torch::optim::OptimizerParamState> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<SGDParamState>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<AdamOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<RMSpropParamState>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<AdagradOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<LBFGSOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<AdamWParamState>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableOptions<RMSpropOptions>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<LBFGSParamState>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<AdagradParamState>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::optim::OptimizerCloneableParamState<AdamParamState>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("LossClosure").skip());
        infoMap.put(new Info("torch::optim::Adagrad").skipDefaults());
        infoMap.put(new Info("torch::optim::Adam").skipDefaults());
        infoMap.put(new Info("torch::optim::AdamW").skipDefaults());
        infoMap.put(new Info("torch::optim::SGD").skipDefaults());
        infoMap.put(new Info("torch::optim::LBFGS").skipDefaults());


        //infoMap.put(new Info("torch::Device").enumerate());

        infoMap.put(new Info("c10::optional<torch::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::optional<torch::Device>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::Dtype").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("std::hash<c10::Symbol>").javaNames("SymbolMap"));
        infoMap.put(new Info("std::hash<c10::DeviceType>").javaNames("DeviceTypeMap"));
        infoMap.put(new Info("std::hash<c10::Device>").javaNames("DeviceMap"));
        infoMap.put(new Info("std::hash<c10::Stream>").javaNames("StreamMap"));

        infoMap.put(new Info("c10::optional<c10::ScalarType>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("caffe2::TypeMeta").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::ScalarType").enumerate());
        infoMap.put(new Info("at::ScalarType").enumerate());
        //infoMap.put(new Info("torch::Dtype").enumerate());

        infoMap.put(new Info("PyObject").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("std::hash<c10::DispatchKey>").javaNames("DispatchKeyMap"));

        infoMap.put(new Info("torch::nn::Module::clone").skip());
        infoMap.put(new Info("torch::nn::Cloneable<LinearImpl>::clone").skip());
        infoMap.put(new Info("torch::nn::Cloneable<FlattenImpl>::clone").skip());
        infoMap.put(new Info("torch::nn::Cloneable<IdentityImpl>::clone").skip());
        infoMap.put(new Info("torch::nn::Cloneable<BilinearImpl>::clone").skip());



        infoMap.put(new Info("torch::nn::Cloneable<LinearImpl>")
                .cast()
                .pointerTypes("CloneableLinearImpl")
                .javaText("@Name(\"torch::nn::Cloneable<torch::nn::LinearImpl>\") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)\n" +
                        "public class CloneableLinearImpl extends Module {\n" +
                        "    static { Loader.load(); }\n" +
                        "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
                        "    public CloneableLinearImpl(Pointer p) { super(p); }\n" +
                        "\n" +
                        "\n" +
                        "  /** {@code reset()} must perform initialization of all members with reference\n" +
                        "   *  semantics, most importantly parameters, buffers and submodules. */\n" +
                        "  public native void reset();\n" +
                        "\n" +
                        "  /** Performs a recursive \"deep copy\" of the {@code Module}, such that all parameters\n" +
                        "   *  and submodules in the cloned module are different from those in the\n" +
                        "   *  original module. */\n" +
                        "  \n" +
                        "}\n"));

        infoMap.put(new Info("torch::nn::Cloneable<IdentityImpl>")
                .cast()
                .pointerTypes("CloneableIdentityImpl")
                .javaText("@Name(\"torch::nn::Cloneable<torch::nn::IdentityImpl>\") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)\n" +
                        "public class CloneableIdentityImpl extends Module {\n" +
                        "    static { Loader.load(); }\n" +
                        "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
                        "    public CloneableIdentityImpl(Pointer p) { super(p); }\n" +
                        "\n" +
                        "\n" +
                        "  /** {@code reset()} must perform initialization of all members with reference\n" +
                        "   *  semantics, most importantly parameters, buffers and submodules. */\n" +
                        "  public native void reset();\n" +
                        "\n" +
                        "  /** Performs a recursive \"deep copy\" of the {@code Module}, such that all parameters\n" +
                        "   *  and submodules in the cloned module are different from those in the\n" +
                        "   *  original module. */\n" +
                        "  \n" +
                        "}"));
        infoMap.put(new Info("torch::nn::Cloneable<BilinearImpl>")
                .cast()
                .pointerTypes("CloneableBilinearImpl")
                .javaText("@Name(\"torch::nn::Cloneable<torch::nn::BilinearImpl>\") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)\n" +
                        "public class CloneableBilinearImpl extends Module {\n" +
                        "    static { Loader.load(); }\n" +
                        "    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */\n" +
                        "    public CloneableBilinearImpl(Pointer p) { super(p); }\n" +
                        "\n" +
                        "\n" +
                        "  /** {@code reset()} must perform initialization of all members with reference\n" +
                        "   *  semantics, most importantly parameters, buffers and submodules. */\n" +
                        "  public native void reset();\n" +
                        "\n" +
                        "  /** Performs a recursive \"deep copy\" of the {@code Module}, such that all parameters\n" +
                        "   *  and submodules in the cloned module are different from those in the\n" +
                        "   *  original module. */\n" +
                        "  \n" +
                        "}\n"));
        infoMap.put(new Info("torch::autograd::TraceableFunction").purify());

        infoMap.put(new Info("torch::nn::FlattenImpl").skip());

        infoMap.put(new Info("std::vector<int64_t>").pointerTypes("LongVector").define());
        infoMap.put(new Info("std::vector<torch::autograd::Edge>").pointerTypes("EdgeVector").define());
        infoMap.put(new Info("std::vector<char>").pointerTypes("CharVector").define());

        //infoMap.put(new Info("std::vector<std::unique_ptr<torch::autograd::FunctionPostHook> >").pointerTypes("UniqueFunctionPostHookVector").define());
        //infoMap.put(new Info("std::vector<std::unique_ptr<torch::autograd::FunctionPreHook> >").pointerTypes("UniqueFunctionPreHookVector").define());


        infoMap.put(new Info("torch::autograd::Variable>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::vector<torch::autograd::Variable>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::vector<std::function<Variable(const Variable&)> >", "torch::autograd::hooks_list").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("std::function<at::Tensor(const at::Tensor&)>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("torch::autograd::FunctionPostHook").purify());
        infoMap.put(new Info("torch::autograd::FunctionPreHook").purify());
        infoMap.put(new Info("std::unique_ptr<torch::autograd::FunctionPostHook>").pointerTypes("FunctionPostHook").define());
        infoMap.put(new Info("std::shared_ptr<torch::autograd::FunctionPostHook>").pointerTypes("SharedFunctionPostHook").define());
        infoMap.put(new Info("std::unique_ptr<torch::autograd::FunctionPreHook>").pointerTypes("FunctionPreHook").define());
        infoMap.put(new Info("std::shared_ptr<torch::autograd::FunctionPreHook>", "std::shared_ptr< FunctionPreHook >").cast().pointerTypes("SharedFunctionPreHook").define());
        infoMap.put(new Info("std::weak_ptr<torch::autograd::Node>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::shared_ptr<torch::autograd::Node>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::optional<double>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<bool>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<std::vector<double> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::tuple<torch::Tensor,c10::optional<std::vector<int64_t> >,c10::optional<std::vector<double> >,c10::optional<bool> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::tuple<torch::Tensor,torch::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2,double> >").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("torch::ExpandingArray<1>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<2>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<3>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<4>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<5>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<1>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<2>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<3>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<4>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<5>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::optional<torch::ExpandingArray<1> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<1,double> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2,double> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<3> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<3,double> >").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("torch::ExpandingArray<1,c10::optional<T> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<2,c10::optional<T> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<3,c10::optional<T> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<4,c10::optional<T> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::ExpandingArray<5,c10::optional<T> >").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("at::ArrayRef<T>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("std::array<T,1>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::array<T,2>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::array<T,3>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::array<T,4>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::array<T,5>").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("c10::optional<std::vector<int64_t> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<int64_t>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::Generator>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<at::DimnameList>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<c10::Stream>").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("torch::nn::functional::BatchNormFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Conv1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Conv2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Conv3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::ConvTranspose1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::ConvTranspose2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::ConvTranspose3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::EmbeddingBagMode").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::CTCLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::TripletMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MultilabelSoftMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::SoftMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MultilabelMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::SmoothL1LossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveMaxPool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveMaxPool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::CosineSimilarityFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::PairwiseDistanceFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Dropout1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Dropout2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::Dropout3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::L1LossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::KLDivFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MSELossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::BinaryCrossEntropyFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::HingeEmbeddingLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MultiMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::CosineEmbeddingLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::NLLLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::CrossEntropyFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::LocalResponseNormFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveAvgPool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveAvgPool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveAvgPool3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MaxUnpool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MaxUnpool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::MaxUnpool3dFuncOptions").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("torch::nn::functional::PixelShuffleFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::nn::functional::AdaptiveMaxPool3dFuncOptions").cast().pointerTypes("Pointer"));

        // Skips
        Arrays.asList(
                "torch::nn::functional::L1LossFuncOptions::reduction_t",
                "torch::nn::functional::KLDivFuncOptions::reduction_t",
                "torch::nn::functional::MSELossFuncOptions::reduction_t",
                "torch::nn::functional::SELossFuncOptions::reduction_t",
                "torch::nn::functional::BinaryCrossEntropyFuncOptions::reduction_t",
                "torch::nn::functional::HingeEmbeddingLossFuncOptions::reduction_t",
                "torch::nn::functional::MultiMarginLossFuncOptions::reduction_t",
                "torch::nn::functional::CosineEmbeddingLossFuncOptions::reduction_t",
                "torch::nn::functional::SmoothL1LossFuncOptions::reduction_t",
                "torch::nn::functional::MultilabelMarginLossFuncOptions::reduction_t",
                "torch::nn::functional::SoftMarginLossFuncOptions::reduction_t",
                "torch::nn::functional::MultilabelSoftMarginLossFuncOptions::reduction_t",
                "torch::nn::functional::TripletMarginLossFuncOptions::reduction_t",
                "torch::nn::functional::CTCLossFuncOptions::reduction_t",
                "torch::nn::functional::PoissonNLLLossFuncOptions::reduction_t",
                "torch::nn::functional::MarginRankingLossFuncOptions::reduction_t",
                "torch::nn::functional::NLLLossFuncOptions::reduction_t",
                "torch::nn::functional::CrossEntropyFuncOptions::reduction_t",
                "torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions::reduction_t",
                "torch::nn::functional::PadFuncOptions::mode_t",
                "torch::nn::functional::InterpolateFuncOptions::mode_t",
                "torch::nn::functional::GridSampleFuncOptions::mode_t",
                "torch::nn::functional::GridSampleFuncOptions::padding_mode_t",

                "torch::nn::functional::KLDivFuncOptions",
                "torch::nn::functional::L1LossFuncOptions",
                "torch::nn::functional::MSELossFuncOptions",
                "torch::nn::functional::SELossFuncOptions",
                "torch::nn::functional::BinaryCrossEntropyFuncOptions",
                "torch::nn::functional::HingeEmbeddingLossFuncOptions",
                "torch::nn::functional::MultiMarginLossFuncOptions",
                "torch::nn::functional::CosineEmbeddingLossFuncOptions",
                "torch::nn::functional::SmoothL1LossFuncOptions",
                "torch::nn::functional::MultilabelMarginLossFuncOptions",
                "torch::nn::functional::SoftMarginLossFuncOptions",
                "torch::nn::functional::MultilabelSoftMarginLossFuncOptions",
                "torch::nn::functional::TripletMarginLossFuncOptions",
                "torch::nn::functional::CTCLossFuncOptions",
                "torch::nn::functional::PoissonNLLLossFuncOptions",
                "torch::nn::functional::MarginRankingLossFuncOptions",
                "torch::nn::functional::NLLLossFuncOptions",
                "torch::nn::functional::CrossEntropyFuncOptions",
                "torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions",
                "torch::nn::functional::PadFuncOptions",
                "torch::nn::functional::InterpolateFuncOptions",
                "torch::nn::functional::GridSampleFuncOptions",
                "torch::nn::functional::GridSampleFuncOptions::paddin",

                "torch::nn::functional::FoldFuncOptions",
                "torch::nn::functional::UnfoldFuncOptions",
                "torch::nn::functional::PoissonNLLLossFuncOptions",
                "torch::nn::functional::MarginRankingLossFuncOptions",
                "torch::nn::functional::AvgPool1dFuncOptions",
                "torch::nn::functional::AvgPool2dFuncOptions",
                "torch::nn::functional::AvgPool3dFuncOptions",
                "torch::nn::functional::MaxPool1dFuncOptions",
                "torch::nn::functional::MaxPool1dFuncOptions",
                "torch::nn::functional::MaxPool2dFuncOptions",
                "torch::nn::functional::MaxPool2dFuncOptions",
                "torch::nn::functional::MaxPool3dFuncOptions",
                "torch::nn::functional::MaxPool3dFuncOptions",
                "torch::nn::functional::FractionalMaxPool2dFuncOptions",
                "torch::nn::functional::FractionalMaxPool2dFuncOptions",
                "torch::nn::functional::FractionalMaxPool3dFuncOptions",
                "torch::nn::functional::FractionalMaxPool3dFuncOptions",
                "torch::nn::functional::LPPool1dFuncOptions",
                "torch::nn::functional::LPPool2dFuncOptions",

                "torch::nn::functional::InstanceNormFuncOptions",
                "torch::nn::functional::EmbeddingBagMode",
                "torch::nn::functional::EmbeddingBagFuncOptions",
                "torch::nn::functional::EmbeddingFuncOptions",
                "torch::nn::functional::FeatureAlphaDropoutFuncOptions",
                "torch::nn::functional::AlphaDropoutFuncOptions",
                "torch::nn::functional::DropoutFuncOptions",
                "torch::nn::functional::NormalizeFuncOptions",

                "c10::optional<at::MemoryFormat>",
                "c10::optional<Scalar>",
                "c10::optional<std::function<at::Tensor(const at::Tensor&)> >",

                "std::enable_shared_from_this<torch::autograd::Node>",

                "at::TensorList",
                "BFloat16",
                "Half",
                "c10::complex<float>",
                "c10::complex<double>",

                "c10::AutogradMetaInterface",
                "at::TensorImpl"


        ).forEach(t -> {
            infoMap.put(new Info(t).cast().pointerTypes("Pointer"));
        });

        infoMap.put(new Info("torch::autograd::Variable").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::autograd::SavedVariable").cast().pointerTypes("Pointer"));

        Arrays.asList(
                "torch::autograd::detail::MakeNextFunctionList",
                "torch::nn::functional::detail::grid_sample",
                "torch::nn::functional::detail::pad",
                "torch::nn::functional::detail::embedding_bag",
                "torch::autograd::generated::unpack_list",
                "torch::detail::operator<<",
                "torch::autograd::AutogradMeta.hooks_",
                "torch::autograd::AutogradMeta.mutex_",
                "torch::autograd::impl::version_counter",
                "torch::autograd::impl::hooks",

                "SavedVariable self_;",

                "c10::Scalar::toHalf",
                "c10::Scalar::toBFloat16",

                "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::Half>::t)",
                "decltype(::c10::impl::ScalarTypeToCPPType<::c10::ScalarType::BFloat16>::t)",

                "std::tuple<torch::Tensor,c10::optional<std::vector<int64_t> >,c10::optional<std::vector<double> >,c10::optional<bool> >"
                ).forEach(t -> {
            infoMap.put(new Info(t).skip());
        });

        infoMap.put(new Info("at::Tensor"));//.javaText(getTemplate("src/main/resources/Tensor.java.tpl")));
        infoMap.put(new Info("std::tuple<Tensor,Tensor>").skip());

        infoMap.put(new Info("torch::autograd::AutogradMeta").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("std::initializer_list").skip());
        infoMap.put(new Info("torch::nn::functional::linear")
                .javaText("@Namespace(\"torch::nn::functional\") public static native @ByVal @Cast(\"torch::Tensor*\") Pointer linear(@Cast(\"const torch::Tensor*\") @ByRef Pointer input, @Cast(\"const torch::Tensor*\") @ByRef Pointer weight,\n" +
                "                     @Cast(\"const torch::Tensor&\") Pointer bias);\n"));

        infoMap.put(new Info("torch::detail::TensorDataContainerType").enumerate());
        infoMap.put(new Info("torch::detail::TensorDataContainer").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("at::TensorOptions").skip());
        infoMap.put(new Info("const at::TensorOptions").skip());

        infoMap.put(new Info("at::ArrayRef<torch::autograd::SavedVariable>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("Functions.h").linePatterns("variable_list apply(variable_list&& grads) override;").javaText("  public native @ByVal VariableVector apply(@ByVal VariableVector grads);\n"));

        infoMap.put(new Info("torch::autograd::Node").purify());

        infoMap.put(new Info("torch::autograd::Node::operator ()").skip());
        infoMap.put(new Info("torch::autograd::Node::apply").skip());

        Arrays.asList(
                "torch::autograd::Node::add_post_hook",
                "torch::autograd::Node::del_post_hook",
                "torch::autograd::Node::post_hooks",
                "torch::autograd::Node::add_pre_hook",
                "torch::autograd::Node::pre_hooks",
                "torch::autograd::Node::set_next_edges",
                "torch::nn::functional::_interp_output_size"
        ).forEach(m -> {
            infoMap.put(new Info(m).skip());
        });

        infoMap.put(new Info("torch::range").skip());
        infoMap.put(new Info("torch::nn::functional::bilinear").javaText("@Namespace(\"torch::nn::functional\") public static native @ByVal Tensor bilinear(@Const @ByRef Tensor input1, @Const @ByRef Tensor input2, @Const @ByRef Tensor weight, @Const @ByRef Tensor bias);\n"));
    }

    private String getTemplate(String path) {
        StringBuilder textBuilder;
        try (InputStream stream = new FileInputStream(new File(path))) {
            textBuilder = new StringBuilder();
            try (Reader reader = new BufferedReader(new InputStreamReader
                    (stream, Charset.forName(StandardCharsets.UTF_8.name())))) {
                int c;
                while ((c = reader.read()) != -1) {
                    textBuilder.append((char) c);
                }
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        return textBuilder.toString();
    }

    public static class TensorPairBase extends Pointer {
        public TensorPairBase(Pointer malloc) {
            super(malloc);
        }
    }

    // using IntArrayRef = ArrayRef<int64_t>; // in c10/util/ArrayRef.h
    @Name("at::IntArrayRef") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
    public static class IntArrayRef extends Pointer {
        private IntArrayRef(Buffer b) {
            super(b);
        }

        public static IntArrayRef of(long... args) {
            LongPointer arr = new LongPointer(args);
            LongBuffer buffer = LongBuffer.allocate(2);
            buffer.put(0, arr.address());
            buffer.put(1, args.length);
            return new IntArrayRef(buffer);
        }
    }


    /*public static class Linear extends org.bytedeco.libtorch.LinearImpl {
        public Linear(long in_features, long out_features) {
            super(in_features, out_features);
        }
    }*/




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

    }
}
