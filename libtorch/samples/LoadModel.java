import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.libtorch.*;
import org.bytedeco.libtorch.Tensor;

import static org.bytedeco.libtorch.global.libtorch.*;

public class LoadModel {
    public static void main(String[] args) throws Exception {
        LinearImpl model = new LinearImpl(5, 1);
        SGD optimizer = new SGD(model.parameters(), /*lr=*/new SGDOptions(0.1));
        Tensor prediction = model.forward(randn(IntArrayRef.of(3, 5)));
        mse_loss(prediction, ones(IntArrayRef.of(3, 1)));//.backward();
        optimizer.step();
    }
}
