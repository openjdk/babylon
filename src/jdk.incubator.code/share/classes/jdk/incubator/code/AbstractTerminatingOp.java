package jdk.incubator.code;

import java.util.List;

/**
 * X
 */
public non-sealed abstract class AbstractTerminatingOp extends AbstractOp implements Op.Terminating {

    final List<Block.Reference> successors;

    /**
     * X
     * @param operands x
     * @param successors x
     */
    protected AbstractTerminatingOp(List<? extends Value> operands, List<Block.Reference> successors) {
        super(operands);

        // @@@ Check unbuilt blocks/arguments
        this.successors = List.copyOf(successors);
    }

    /**
     * X
     * @param operands x
     */
    protected AbstractTerminatingOp(List<? extends Value> operands) {
        this(operands, List.of());
    }

    /**
     * X
     * @param that x
     * @param cc x
     */
    protected AbstractTerminatingOp(AbstractOp that, CodeContext cc) {
        super(that, cc);

        // @@@ Remove CodeContext::getReferenceOrCreate
        this.successors = that.successors().stream().map(s -> cc.getReferenceOrCreate(s)).toList();
    }

    @Override
    public final List<Block.Reference> successors() {
        return successors;
    }
}
