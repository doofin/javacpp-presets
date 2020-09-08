@Name("std::tuple<torch::Tensor,torch::Tensor>") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public static class TensorPair extends Pointer {
    public TensorPair() {
        super(Pointer.malloc(Loader.sizeof(Tensor.class) * 2));
    }

    Tensor first() {
        return getPointer(Tensor.class, 0);
    }

    Tensor second() {
        return getPointer(Tensor.class, 1);
    }
}
