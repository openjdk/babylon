package hat.phases;

import hat.codetypes.PtrType;
import hat.dialect.ArithMathOps;
import hat.dialect.TileOps;
import hat.types.Tile;
import hat.TileContext;
import hat.TileOp;
import hat.buffer.TensorF32;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public record HATTilesPhase() implements HATPhase {

    private CoreOp.FuncOp appendAlignment(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = new HashSet<>();

        // Check for static access to the TileContext class
        boolean isTileUsed = OpHelper.isKlassUsed(lookup, funcOp, TileContext.class);

        // Also check the tile dialect was introduced
        isTileUsed |= funcOp.elements()
                .anyMatch(element -> element instanceof TileOps.TOp
                        || element instanceof ArithMathOps.ArithMathOp);

        if (isTileUsed) {
            // We need to transform the code tree to insert a Invoke for the alignment associated with a new var
            // to carry for all loads/store operations after the alignment.
            List<Block.Parameter> parameters = funcOp.body().entryBlock().parameters();
            Op firstOp = funcOp.bodies().getFirst().blocks().getFirst().firstOp();
            // analyze each parameter
            List<CoreOp.VarOp> tileArgs = new ArrayList<>();
            for (Block.Parameter p : parameters) {
                Op.Result paramUsage = p.uses().getFirst();
                if (paramUsage.declaringElement() instanceof CoreOp.VarOp varOp) {
                    var s = varOp.resultType().valueType();
                    if (s instanceof PtrType || s.toString().equals(TensorF32.class.getCanonicalName())) {
                        tileArgs.add(varOp);
                        opsToProcess.add(varOp);
                    }
                    firstOp = varOp;
                }
            }

            if (tileArgs.contains(firstOp)) {
                // If this is the case, we need to move firstOp to next Op
                // Otherwise, the subsequence transform phase will not expand
                // with the new Ops for performing the alignment.
                boolean wasFound = false;
                for (CodeElement<?, ?> codeElement : funcOp.elements().sequential().toList()) {
                    if (wasFound) {
                        firstOp = (Op) codeElement;
                        break;
                    }
                    if (codeElement == firstOp) {
                        wasFound = true;
                    }
                }
            }

            Map<Op, Op.Result> paramMap = new HashMap<>();
            final Op finalFirstOp = firstOp;
            Map<Op, Op.Result> useVarOps = new HashMap<>();
            CoreOp.FuncOp finalFuncOp = funcOp;
            funcOp = funcOp.transform((builder, op) -> {
                if (opsToProcess.contains(op) && op instanceof CoreOp.VarOp) {
                    Op.Result newVarOpResult = builder.add(op);
                    paramMap.put(op, newVarOpResult);
                } else if (op == finalFirstOp) {
                    builder.add(finalFirstOp);
                    // place new invoke ops here
                    // do this for all parameters
                    for (CoreOp.VarOp varTile : tileArgs) {
                        CoreOp.ConstantOp constantOp = CoreOp.constant(JavaType.INT, 16);
                        Op.Result constantValue = builder.add(constantOp);
                        JavaOp.InvokeOp invoke = JavaOp.invoke(TILE_ARRAY_ALIGN, List.of(paramMap.get(varTile), constantValue));
                        Op.Result invokeResult = builder.add(invoke);
                        CoreOp.VarOp varOp = CoreOp.var(varTile.varName().concat("_"), invokeResult);
                        Op.Result varOpResult = builder.add(varOp);
                        for (Op.Result u : varTile.result().uses()) {
                            useVarOps.put(u.op(), varOpResult);
                            opsToProcess.add(u.op());
                        }
                        varTable.addIfNeededOrThrow(finalFuncOp.funcName(), varOp, VarTable.HATOpAttribute.TILE);
                    }
                } else if (opsToProcess.contains(op) && op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    CoreOp.VarAccessOp.VarLoadOp v = CoreOp.varLoad(useVarOps.get(varLoadOp));
                    Op.Result newVarLoad = builder.add(v);
                    builder.context().mapValue(varLoadOp.result(), newVarLoad);

                } else {
                    builder.add(op);
                }
                return builder;
            });
        }
        return funcOp;
    }

    private static final MethodRef TILE_ARRAY_ALIGN = MethodRef.method(TileAlign.class, "align", Tile.class, Object.class, int.class);

    public static class TileAlign {
        public static Tile align(Object inputRef, int alignment) {
            return null;
        }
    }

    private CoreOp.FuncOp classifyTileVarOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        // process Tile-Vars to insert into the VarTable
        // Load operation returns a new Tile (view of the input data in a tile)
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(TileContext.class))
                .filter(invoke -> invoke.name().equals("load"))
                .forEach(invoke ->
                        invoke.op().result().uses().stream()
                                .filter(result -> (result.op() instanceof CoreOp.VarOp))
                                .map(result -> (CoreOp.VarOp) result.op())
                                .forEach(opsToProcess::add));

        // Process nodes after Tile dialect
        funcOp.elements().forEach(element -> {
            switch (element) {
                case TileOps.LoadOp loadOp when loadOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo ->
                        opsToProcess.add(varOo);
                case TileOps.TileFullOp fullOp when fullOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo ->
                        opsToProcess.add(varOo);
                case TileOps.TileSumOp sumOp when sumOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo ->
                        opsToProcess.add(varOo);
                case TileOps.TileZerosOp zerosOp when zerosOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo ->
                        opsToProcess.add(varOo);
                case null, default -> {
                }
            }
        });

        // We have identified the invoke and the varOp associated with it
        return Trxfmr.of(lookup, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                Op.Result opResult = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(funcOp.funcName(), opResult.op(), VarTable.HATOpAttribute.TILE);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private CoreOp.FuncOp classifyArithmeticTileVarOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        // process Tile-Vars to insert into the VarTable
        // we create Tiles when we load
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(TileOp.class))
                .forEach(invoke ->
                        invoke.op().result().uses().stream()
                                .filter(result -> (result.op() instanceof CoreOp.VarOp))
                                .map(result -> (CoreOp.VarOp) result.op())
                                .forEach(opsToProcess::add));

        // Process nodes after Tile dialect
        funcOp.elements().forEach(element -> {
            if (element instanceof ArithMathOps.ArithMathOp arithMathOp && arithMathOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo) {
                opsToProcess.add(varOo);
            }
        });

        // We have identified the invoke and the varOp associated with it
        return Trxfmr.of(lookup, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                Op.Result opResult = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(funcOp.funcName(), opResult.op(), VarTable.HATOpAttribute.TILE);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }


    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        List<ActionTransformer> transformers = List.of(
                this::appendAlignment,
                this::classifyTileVarOp,
                this::classifyArithmeticTileVarOp
        );
        CoreOp.FuncOp[] f = new CoreOp.FuncOp[]{funcOp};
        transformers.forEach(action -> f[0] = action.apply(lookup, f[0], varTable));
        return f[0];
    }

}
