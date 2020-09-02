import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.libtorch.*;
import static org.bytedeco.libtorch.global.libtorch.*;

public class LoadModel {
    public static void main(String[] args) throws Exception {
        torch.nn.Linear model = new torch.nn.Linear(5, 1);
        torch.optim.SGD optimizer = new torch.optim.SGD(model.parameters(), /*lr=*/0.1);
        torch.Tensor prediction = model.forward(torch.randn(new long[]{3, 5}));
        torch.mse_loss(prediction, torch.ones(new long[]{3, 1})).backward();
        optimizer.step();
    }
}
