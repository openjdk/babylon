package hat.optools;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;

public class BlockWrapper extends CodeElementWrapper<Block> {
    public Block block() {
        return codeElement;
    }

    public BlockWrapper(Block block) {
        super(block);
    }

    public int opCount() {
        return block().ops().size();
    }

    public <O extends Op> O op(int delta) {
        O op = null;
        if (delta >= 0) {
            op = (O) block().children().get(delta);
        } else {
            op = (O) block().children().get(opCount() + delta);
        }
        return op;
    }

    public BodyWrapper parentBody() {
        return new BodyWrapper(block().parentBody());
    }
}
