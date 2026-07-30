package hat.phases;

import hat.TileContext;
import hat.TileOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.Set;

public record HATTilesPhase() implements HATPhase {

    private CoreOp.FuncOp classifyTileVarOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        // process Tile-Vars to insert into the VarTable
        // we create Tiles when we load
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(TileContext.class))
                .filter(invoke -> invoke.name().equals("load"))
                .forEach(invoke -> {
                    invoke.op().result().uses().stream()
                            .filter(result -> (result.op() instanceof CoreOp.VarOp))
                            .map(result -> (CoreOp.VarOp) result.op())
                            .forEach(opsToProcess::add);
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
                .filter(invoke -> invoke.named("add", "mma"))
                .forEach(invoke ->
                        invoke.op().result().uses().stream()
                        .filter(result -> (result.op() instanceof CoreOp.VarOp))
                        .map(result -> (CoreOp.VarOp) result.op())
                        .forEach(opsToProcess::add));

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
        funcOp = classifyTileVarOp(lookup, funcOp, varTable);
        funcOp = classifyArithmeticTileVarOp(lookup, funcOp, varTable);
        return funcOp;
    }

}
