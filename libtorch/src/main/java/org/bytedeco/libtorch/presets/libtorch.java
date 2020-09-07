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
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

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

        infoMap.put(new Info("torch::Tensor").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("c10::kNoQEngine").skip());
        infoMap.put(new Info("c10::kFBGEMM").skip());
        infoMap.put(new Info("c10::kQNNPACK").skip());


        infoMap.put(new Info("LEGACY_CONTIGUOUS_MEMORY_FORMAT").skip());


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
        infoMap.put(new Info("at::optional<torch::Device>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("torch::Dtype").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("std::hash<c10::Symbol>").javaNames("SymbolMap"));
        infoMap.put(new Info("std::hash<c10::DeviceType>").javaNames("DeviceTypeMap"));
        infoMap.put(new Info("std::hash<c10::Device>").javaNames("DeviceMap"));

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

        infoMap.put(new Info("torch::nn::FlattenImpl").skip());

        infoMap.put(new Info("c10::optional<double>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<bool>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<std::vector<double> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::tuple<torch::Tensor,c10::optional<std::vector<int64_t> >,c10::optional<std::vector<double> >,c10::optional<bool> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("std::tuple<torch::Tensor,torch::Tensor>").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<torch::ExpandingArray<2,double> >").cast().pointerTypes("Pointer"));


        infoMap.put(new Info("torch::reduction_t").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::variant<enumtype::kNone, enumtype::kMean, enumtype::kSum>").cast().pointerTypes("Pointer"));




        infoMap.put(new Info("mode_t").skip());

        infoMap.put(new Info("torch::ExpandingArray<1>").javaNames("ExpandingArray1"));
        infoMap.put(new Info("torch::ExpandingArray<2>").javaNames("ExpandingArray2"));
        infoMap.put(new Info("torch::ExpandingArray<3>").javaNames("ExpandingArray3"));
        infoMap.put(new Info("torch::ExpandingArray<4>").javaNames("ExpandingArray4"));
        infoMap.put(new Info("torch::ExpandingArray<5>").javaNames("ExpandingArray5"));

        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<1>").javaNames("ExpandingArrayWithOptionalElem1"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<2>").javaNames("ExpandingArrayWithOptionalElem2"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<3>").javaNames("ExpandingArrayWithOptionalElem3"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<4>").javaNames("ExpandingArrayWithOptionalElem4"));
        infoMap.put(new Info("torch::ExpandingArrayWithOptionalElem<5>").javaNames("ExpandingArrayWithOptionalElem5"));

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
//
//std::tuple<torch::Tensor,torch::Tensor>


        infoMap.put(new Info("c10::optional<std::vector<int64_t> >").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("c10::optional<int64_t>").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("BatchNormFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Conv1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Conv2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Conv3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("ConvTranspose1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("IntArrayRef").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("ConvTranspose2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("ConvTranspose3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("EmbeddingBagMode").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("CTCLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("TripletMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MultilabelSoftMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("SoftMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MultilabelMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("SmoothL1LossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("AdaptiveMaxPool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("AdaptiveMaxPool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("CosineSimilarityFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("PairwiseDistanceFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Dropout1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Dropout2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Dropout3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("L1LossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("KLDivFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MSELossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("BinaryCrossEntropyFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("HingeEmbeddingLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MultiMarginLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("CosineEmbeddingLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("NLLLossFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("CrossEntropyFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("BinaryCrossEntropyWithLogitsFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("LocalResponseNormFuncOptions").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("AdaptiveAvgPool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("AdaptiveAvgPool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("AdaptiveAvgPool3dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MaxUnpool1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MaxUnpool2dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("MaxUnpool3dFuncOptions").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("Conv1dFuncOptions").cast().pointerTypes("Pointer"));
        infoMap.put(new Info("Conv1dFuncOptions").cast().pointerTypes("Pointer"));

        infoMap.put(new Info("L1LossFuncOptions::reduction_t").skip());

        infoMap.put(new Info("std::initializer_list").skip());


    }

    public static class Linear extends org.bytedeco.libtorch.LinearImpl {
        public Linear(long in_features, long out_features) {
            super(in_features, out_features);
        }
    }
}
