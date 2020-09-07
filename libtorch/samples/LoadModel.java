import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.libtorch.*;
import static org.bytedeco.libtorch.global.libtorch.*;

public class LoadModel {
    public static void main(String[] args) throws Exception {
        Linear model = new Linear(5, 1);
        SGD optimizer = new SGD(model.parameters(), /*lr=*/new SGDOptions(0.1));
        Tensor prediction = model.forward(torch.randn(new long[]{3, 5}));
        torch.mse_loss(prediction, torch.ones(new long[]{3, 1})).backward();
        optimizer.step();
    }
}
