package hat.optools;

import java.lang.reflect.code.Body;
import java.util.function.Consumer;

public class BodyWrapper extends CodeElementWrapper<Body> {
    public Body body() {
        return codeElement;
    }

    BodyWrapper(Body body) {
        super(body);
    }

    public static BodyWrapper of(Body body) {
        return new BodyWrapper(body);
    }

    public static void onlyBlock(Body body, Consumer<BlockWrapper> blockWrapperConsumer) {
        blockWrapperConsumer.accept(new BlockWrapper(body.blocks().getFirst()));
    }

    public void onlyBlock(Consumer<BlockWrapper> blockWrapperConsumer) {
        onlyBlock(body(), blockWrapperConsumer);
    }


}
