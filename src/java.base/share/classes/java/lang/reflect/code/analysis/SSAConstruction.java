package java.lang.reflect.code.analysis;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.CodeElement;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SequencedSet;
import java.util.Set;

/**
 * This is an implementation of SSA construction based on
 * <a href="https://doi.org/10.1007/978-3-642-37051-9">
 * Simple end Efficient Construction of Static Single Assignment Form (pp 102-122)
 * </a>.
 * <p>
 * This implementation contains some adaptions, notably:
 * <ul>
 *     <li>Adapt to block parameters rather than phi functions.</li>
 *     <li>Adapt to work with multiple bodies.</li>
 * </ul>
 */
public class SSAConstruction implements OpTransformer {

    private final Map<CoreOp.VarOp, Map<Block, Val>> currentDef = new HashMap<>();
    private final Set<Block> sealedBlocks = new HashSet<>();
    private final Map<Block, Map<CoreOp.VarOp, Phi>> incompletePhis = new HashMap<>();

    // according to the algorithm:
    // "As only filled blocks may have successors, predecessors are always filled."
    // In consequence, this means that only filled predecessors should be considered
    // when recursively searching for a definition
    private final Map<Block, SequencedSet<Block>> predecessors = new HashMap<>();
    // as we can't modify the graph at the same time as we analyze it,
    // we need to store which load op needs to remapped to which value
    private final Map<CoreOp.VarAccessOp.VarLoadOp, Val> loads = new HashMap<>();
    private final Map<Block, List<Phi>> additionalParameters = new HashMap<>();
    // as we look up definitions during the actual transformation again,
    // we might encounter deleted phis.
    // we use this set to be able to correct that during transformation
    private final Set<Phi> deletedPhis = new HashSet<>();

    public static <O extends Op> O transform(O nestedOp) {
        SSAConstruction construction = new SSAConstruction();
        construction.prepare(nestedOp);
        @SuppressWarnings("unchecked")
        O temp = (O) nestedOp.transform(CopyContext.create(), construction);
        return temp;
    }

    private SSAConstruction() {
    }

    private void prepare(Op nestedOp) {
        nestedOp.traverse(null, CodeElement.opVisitor((_, op) -> {
            switch (op) {
                case CoreOp.VarAccessOp.VarLoadOp load -> {
                    Val val = readVariable(load.varOp(), load.parentBlock());
                    registerLoad(load, val);
                }
                case CoreOp.VarAccessOp.VarStoreOp store ->
                        writeVariable(store.varOp(), store.parentBlock(), new Holder(store.storeOperand()));
                case CoreOp.VarOp initialStore ->
                        writeVariable(initialStore, initialStore.parentBlock(), new Holder(initialStore.initOperand()));
                case Op.Terminating _ -> {
                    Block block = op.parentBlock();
                    // handle the sealing, i.e. only now make this block a predecessor of its successors
                    for (Block.Reference successor : block.successors()) {
                        Block successorBlock = successor.targetBlock();
                        Set<Block> blocks = this.predecessors.computeIfAbsent(successorBlock, _ -> new LinkedHashSet<>());
                        blocks.add(block);
                        // if this was the last predecessor added to successorBlock, seal it
                        if (blocks.size() == successorBlock.predecessors().size()) {
                            sealBlock(successorBlock);
                        }
                    }
                }
                default -> {
                }
            }
            return null;
        }));
    }

    private void registerLoad(CoreOp.VarAccessOp.VarLoadOp load, Val val) {
        this.loads.put(load, val);
        if (val instanceof Phi phi) {
            phi.users.add(load);
        }
    }

    private void writeVariable(CoreOp.VarOp variable, Block block, Val value) {
        this.currentDef.computeIfAbsent(variable, _ -> new HashMap<>()).put(block, value);
    }

    private Val readVariable(CoreOp.VarOp variable, Block block) {
        Val value = this.currentDef.getOrDefault(variable, Map.of()).get(block);
        if (value == null
            // deleted Phi, this is an old reference
            // due to our 2-step variant of the original algorithm, we might encounter outdated definitions
            // when we read to prepare block arguments
            || value instanceof Phi phi && this.deletedPhis.contains(phi)) {
            return readVariableRecursive(variable, block);
        }
        return value;
    }

    private Val readVariableRecursive(CoreOp.VarOp variable, Block block) {
        Val value;
        if (!block.isEntryBlock() && !this.sealedBlocks.contains(block)) {
            Phi phi = new Phi(variable, block);
            value = phi;
            this.incompletePhis.computeIfAbsent(block, _ -> new HashMap<>()).put(variable, phi);
            this.additionalParameters.computeIfAbsent(block, _ -> new ArrayList<>()).add(phi);
        } else if (block.isEntryBlock() && variable.ancestorBody() != block.parentBody()) {
            // we are in an entry block but didn't find a definition yet
            Block enclosingBlock = block.parent().parent().parent();
            assert enclosingBlock != null : "def not found in entry block, with no enclosing block";
            value = readVariable(variable, enclosingBlock);
        } else if (this.predecessors.get(block).size() == 1) {
            value = readVariable(variable, this.predecessors.get(block).getFirst());
        } else {
            Phi param = new Phi(variable, block);
            writeVariable(variable, block, param);
            value = addPhiOperands(variable, param);
            // To go from Phis to block parameters, we remember that we produced a Phi here.
            // This means that edges to this block need to pass a value via parameter
            if (value == param) {
                this.additionalParameters.computeIfAbsent(block, _ -> new ArrayList<>()).add(param);
            }
        }
        writeVariable(variable, block, value); // cache value for this variable + block
        return value;
    }

    private Val addPhiOperands(CoreOp.VarOp variable, Phi value) {
        for (Block pred : this.predecessors.getOrDefault(value.block(), Collections.emptySortedSet())) {
            value.appendOperand(readVariable(variable, pred));
        }
        return tryRemoveTrivialPhi(value);
    }

    private Val tryRemoveTrivialPhi(Phi phi) {
        Val same = null;
        for (Val op : phi.operands()) {
            if (op == same || op == phi) {
                continue;
            }
            if (same != null) {
                return phi;
            }
            same = op;
        }
        // we shouldn't have phis without operands (other than itself)
        assert same != null : "phi without different operands";
        List<Phi> phiUsers = phi.replaceBy(same, this);
        List<Phi> phis = this.additionalParameters.get(phi.block());
        if (phis != null) {
            phis.remove(phi);
        }
        for (Phi user : phiUsers) {
            tryRemoveTrivialPhi(user);
        }
        return same;
    }

    private void sealBlock(Block block) {
        this.incompletePhis.getOrDefault(block, Map.of()).forEach(this::addPhiOperands);
        this.sealedBlocks.add(block);
    }

    // only used during transformation

    private Value resolveValue(CopyContext context, Val val) {
        return switch (val) {
            case Holder holder -> context.getValueOrDefault(holder.value(), holder.value());
            case Phi phi -> {
                List<Phi> phis = this.additionalParameters.get(phi.block());
                int additionalParameterIndex = phis.indexOf(phi);
                assert additionalParameterIndex >= 0 : "phi not in parameters " + phi;
                int index = additionalParameterIndex + phi.block().parameters().size();
                Block.Builder b = context.getBlock(phi.block());
                yield b.parameters().get(index);
            }
        };
    }

    @Override
    public Block.Builder apply(Block.Builder block, Op op) {
        Block originalBlock = op.parentBlock();
        CopyContext context = block.context();
        switch (op) {
            case CoreOp.VarAccessOp.VarLoadOp load -> {
                Val val = this.loads.get(load);
                context.mapValue(load.result(), resolveValue(context, val));
            }
            case CoreOp.VarOp _, CoreOp.VarAccessOp.VarStoreOp _ -> {
            }
            case Op.Terminating _ -> {
                // make sure outgoing branches are corrected
                for (Block.Reference successor : originalBlock.successors()) {
                    Block successorBlock = successor.targetBlock();
                    List<Phi> successorParams = this.additionalParameters.getOrDefault(successorBlock, List.of());
                    List<Value> additionalParams = successorParams.stream()
                            .map(phi -> readVariable(phi.variable, originalBlock))
                            .map(val -> resolveValue(context, val))
                            .toList();
                    List<Value> values = context.getValues(successor.arguments());
                    values.addAll(additionalParams);
                    Block.Builder successorBlockBuilder = context.getBlock(successorBlock);
                    context.mapSuccessor(successor, successorBlockBuilder.successor(values));
                }
                block.op(op);
            }
            default -> block.op(op);
        }
        return block;
    }

    @Override
    public void apply(Block.Builder block, Block b) {
        // add the required additional parameters to this block
        boolean isEntry = b.isEntryBlock();
        for (Phi phi : this.additionalParameters.getOrDefault(b, List.of())) {
            if (isEntry) {
                // Phis in entry blocks denote captured values. Do not add as param but make sure
                // the original value is used
                assert phi.operands().size() == 1 : "entry block phi with multiple operands";
                CopyContext context = block.context();
                context.mapValue(resolveValue(context, phi), resolveValue(context, phi.operands().getFirst()));
            } else {
                block.parameter(phi.variable.varValueType());
            }
        }

        // actually visit ops in this block
        OpTransformer.super.apply(block, b);
    }

    sealed interface Val {
    }

    record Holder(Value value) implements Val {
    }

    record Phi(CoreOp.VarOp variable, Block block, List<Val> operands, Set<Object> users) implements Val {
        Phi(CoreOp.VarOp variable, Block block) {
            this(variable, block, new ArrayList<>(), new HashSet<>());
        }

        void appendOperand(Val val) {
            this.operands.add(val);
            if (val instanceof Phi phi) { // load op uses are added separately
                phi.users.add(this);
            }
        }

        @Override
        public boolean equals(Object obj) {
            return this == obj;
        }

        @Override
        public int hashCode() {
            return Objects.hash(this.variable, this.block);
        }

        public List<Phi> replaceBy(Val same, SSAConstruction construction) {
            List<Phi> usingPhis = new ArrayList<>();
            for (Object user : this.users) {
                if (user == this) {
                    continue;
                }
                if (same instanceof Phi samePhi) {
                    samePhi.users.add(user);
                }
                switch (user) {
                    case Phi phi -> {
                        int i = phi.operands.indexOf(this);
                        assert i >= 0 : "use does not have this as operand";
                        phi.operands.set(i, same);
                        usingPhis.add(phi);
                    }
                    case CoreOp.VarAccessOp.VarLoadOp load -> construction.loads.put(load, same);
                    default -> throw new UnsupportedOperationException(user + ":" + user.getClass());
                }
            }
            if (same instanceof Phi samePhi) {
                samePhi.users.remove(this);
            }
            construction.currentDef.get(this.variable).put(this.block, same);
            construction.deletedPhis.add(this); // we might not replace all current defs, so mark this phi as deleted
            this.users.clear();
            return usingPhis;
        }

        @Override
        public String toString() {
            return "Phi[" + variable.varName() + "(" + block.index() + ")," + "operands: " + operands.size() + "}";
        }
    }
}
