package jdk.incubator.code;

import java.util.List;

/**
 * The abstract implementation of a terminating operation. All concrete terminating operations extend this
 * class and implement {@link Op.Terminating}.
 *
 * <h2>Operation implementation requirements</h2>
 * <p>
 * A concrete terminating operation class must satisfy the implementation requirements of a concrete non-terminating
 * operation specified by {@link AbstractOp} in addition to the following requirements:
 * <ul>
 * <li>
 * override {@link #successors()} if instances may have successors;
 * </ul>
 * <p>
 * A concrete terminating operation class may additionally:
 * <ul>
 * <li>
 * override {@link #externalizeOpName()} and {@link #externalize()} to define an external form;
 * <li>
 * implement {@link Op.Lowerable} to define a lowering; and
 * <li>
 * provide operation-specific accessors for operation-specific state.
 * </ul>
 */
public non-sealed abstract class AbstractTerminatingOp extends InternalAbstractOp implements Op.Terminating {

    final List<Block.Reference> successors;

    /**
     * Constructs a terminating operation with a list of operands and list of successors
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @param successors the list of successors, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     * @throws IllegalArgumentException if a successor's referencing block is built or successor's block argument's
     * declaring block is built.
     */
    protected AbstractTerminatingOp(List<? extends Value> operands, List<Block.Reference> successors) {
        super(operands);

        // @@@ Check unbuilt blocks/arguments
        this.successors = List.copyOf(successors);
    }

    /**
     * Constructs a terminating operation with a list of operands and an empty list of successors
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     */
    protected AbstractTerminatingOp(List<? extends Value> operands) {
        this(operands, List.of());
    }

    /**
     * Constructs a terminating operation with operands and successors mapped from, and location copied from, the given operation.
     * <p>
     * The constructed operation's operands are the values mapped, in order, from the given operation's operands using
     * the given code context. The constructed operation's successors are the successors mapped, in order, from the
     * given operation's successors using the given code context and applying
     * {@link CodeContext#getReferenceOrCreate(Block.Reference)} to each successor. The constructed operation's location
     * is the given operation's location, if any.
     *
     * @param that the operation
     * @param cc   the code context
     * @throws IllegalArgumentException if an operation's operand has no context mapping
     * @throws IllegalArgumentException if a mapped value's declaring block is built.
     * @throws IllegalArgumentException if a mapped successor's referencing block is built or mapped successor's block
     * argument's declaring block is built.
     */
    protected AbstractTerminatingOp(AbstractTerminatingOp that, CodeContext cc) {
        super(that, cc);

        this.successors = that.successors().stream().map(s -> cc.getReferenceOrCreate(s)).toList();
    }

    @Override
    public final List<Block.Reference> successors() {
        return successors;
    }
}
