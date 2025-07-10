package jdk.incubator.code.analysis;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

import static jdk.incubator.code.dialect.core.CoreOp.branch;
import static jdk.incubator.code.dialect.core.CoreOp.return_;

/**
 * Functionality for inlining code.
 */
public final class Inliner {

    private Inliner() {}

    /**
     * An inline consumer that inserts a return operation with a value, if non-null.
     */
    public static final BiConsumer<Block.Builder, Value> INLINE_RETURN = (block, value) -> {
        block.op(value != null ? return_(value) : CoreOp.return_());
    };

    /**
     * Inlines the invokable operation into the given block builder and returns the block builder from which to
     * continue building.
     * <p>
     * This method {@link Block.Builder#transformBody(Body, List, CopyContext, OpTransformer) transforms} the
     * body of the invokable operation with the given arguments, a new context, and an operation transformer that
     * replaces return operations by applying the given consumer to a block builder and a return value.
     * <p>
     * The operation transformer copies all operations except return operations whose nearest invokable operation
     * ancestor is the given the invokable operation. When such a return operation is encountered, then on
     * first encounter of its grandparent body a return block builder is computed and used for this return operation
     * and encounters of subsequent return operations with the same grandparent body.
     * <p>
     * If the grandparent body has only one block then operation transformer's block builder is the return
     * block builder. Otherwise, if the grandparent body has one or more blocks then the return block builder is
     * created from the operation transformer's block builder. The created return block builder will have a block
     * parameter whose type corresponds to the return type, or will have no parameter for void return.
     * The computation finishes by applying the return block builder and a return value to the inlining consumer.
     * If the grandparent body has only one block then the return value is the value mapped from the return
     * operation's operand, or is null for void return. Otherwise, if the grandparent body has one or more blocks
     * then the value is the block parameter of the created return block builder, or is null for void return.
     * <p>
     * For every encounter of a return operation the associated return block builder is compared against the
     * operation transformer's block builder. If they are not equal then a branch operation is added to the
     * operation transformer's block builder whose successor is the return block builder with a block argument
     * that is the value mapped from the return operation's operand, or with no block argument for void return.
     * @apiNote
     * It is easier to inline an invokable op if its body is in lowered form (there are no operations in the blocks
     * of the body that are lowerable). This ensures a single exit point can be created (paired with the single
     * entry point). If there are one or more nested return operations, then there is unlikely to be a single exit.
     * Transforming the model to create a single exit point while preserving nested structure is in general
     * non-trivial and outside the scope of this method. In such cases the invokable operation can be transformed
     * with a lowering transformation after which it can then be inlined.
     *
     * @param _this the block builder
     * @param invokableOp the invokable operation
     * @param args the arguments to map to the invokable operation's parameters
     * @param inlineConsumer the consumer applied to process the return from the invokable operation.
     *                       This is called once for each grandparent body of a return operation, with a block to
     *                       build replacement operations and the return value, or null for void return.
     * @return the block builder to continue building from
     * @param <O> The invokable type
     */
    public static <O extends Op & Op.Invokable>
    Block.Builder inline(Block.Builder _this, O invokableOp, List<? extends Value> args,
                         BiConsumer<Block.Builder, Value> inlineConsumer) {
        Map<Body, Block.Builder> returnBlocks = new HashMap<>();
        // Create new context, ensuring inlining is isolated
        _this.transformBody(invokableOp.body(), args, CopyContext.create(), (block, op) -> {
            // If the return operation is associated with the invokable operation
            if (op instanceof CoreOp.ReturnOp rop && getNearestInvokeableAncestorOp(op) == invokableOp) {
                // Compute the return block
                Block.Builder returnBlock = returnBlocks.computeIfAbsent(rop.ancestorBody(), _body -> {
                    Block.Builder rb;
                    // If the body has one block we know there is just one return op declared, otherwise there may
                    // one or more. If so, create a new block that joins all the returns.
                    // Note: we could count all return op in a body to avoid creating a new block for a body
                    // with two or more blocks with only one returnOp is declared.
                    Value r;
                    if (rop.ancestorBody().blocks().size() != 1) {
                        List<TypeElement> param = rop.returnValue() != null
                                ? List.of(invokableOp.invokableType().returnType())
                                : List.of();
                        rb = block.block(param);
                        r = !param.isEmpty()
                                ? rb.parameters().get(0)
                                : null;
                    } else {
                        r = rop.returnValue() != null
                                ? block.context().getValue(rop.returnValue())
                                : null;
                        rb = block;
                    }

                    // Inline the return
                    inlineConsumer.accept(rb, r);

                    return rb;
                });

                // Replace the return op with a branch to the return block, if needed
                if (!returnBlock.equals(block)) {
                    // Replace return op with branch to return block, with given return value
                    List<Value> arg = rop.returnValue() != null
                            ? List.of(block.context().getValue(rop.returnValue()))
                            : List.of();
                    block.op(branch(returnBlock.successor(arg)));
                }

                return block;
            }

            block.op(op);
            return block;
        });


        Block.Builder builder = returnBlocks.get(invokableOp.body());
        return builder != null ? builder : _this;
    }

    private static Op getNearestInvokeableAncestorOp(Op op) {
        do {
            op = op.ancestorOp();
        } while (!(op instanceof Op.Invokable));
        return op;
    }
}
