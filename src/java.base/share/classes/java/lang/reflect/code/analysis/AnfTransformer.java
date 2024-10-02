/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package java.lang.reflect.code.analysis;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.AnfDialect;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.util.*;
import java.util.function.Function;

public class AnfTransformer {

    final CoreOp.FuncOp sourceOp;
    final Map<Block, Function<Body.Builder, AnfDialect.AnfFuncOp>> fBuilders = new HashMap<>();
    final Body.Builder outerBodyBuilder;
    final ImmediateDominatorMap idomMap;
    final Map<String, Value> funMap = new HashMap<>();
    final Map<String, AnfDialect.AnfApply> appMap = new HashMap<>();

    public AnfTransformer(CoreOp.FuncOp funcOp) {
        sourceOp = funcOp;
        outerBodyBuilder = Body.Builder.of(null, funcOp.invokableType());
        idomMap = new ImmediateDominatorMap(funcOp.body());
    }

    public AnfDialect.AnfFuncOp transform() {
        return transformOuterBody(sourceOp.body());
    }

    //Outer body corresponds to outermost letrec
    //F_p
    public AnfDialect.AnfFuncOp transformOuterBody(Body b) {
        var entry = b.entryBlock();

        var builderEntry = outerBodyBuilder.entryBlock();

        for (Block.Parameter p : entry.parameters()) {
            var newP = builderEntry.parameter(p.type());
            builderEntry.context().mapValue(p,newP);
        }

        var outerLetRecBody = Body.Builder.of(outerBodyBuilder, FunctionType.functionType(b.yieldType(), List.of()), CopyContext.create(builderEntry.context()));

        List<Block> dominatedBlocks = idomMap.idominates(entry);
        List<AnfDialect.AnfFuncOp> funs = dominatedBlocks.stream().map(block -> transformBlock(block, outerLetRecBody)).toList();

        //Remove ApplyStubs
        var res = transformBlock(entry, outerLetRecBody);

        //var transformContext = CopyContext.create();
        //for (Value v : funMap.values()) {
        //    transformContext.mapValue(v,v);
        //}
        //res = res.transform(transformContext, new ApplyStubTransformer());

        return res;

    }

    public AnfDialect.AnfFuncOp transformBlock(Block b, Body.Builder bodyBuilder) {
        if (idomMap.idominates(b).isEmpty()) {
            return transformLeafBlock(b, bodyBuilder);
        }
        return transformDomBlock(b, bodyBuilder);
    }

    //"Leaf" in this case is a leaf of the dominator tree
    public AnfDialect.AnfFuncOp transformLeafBlock(Block b, Body.Builder ancestorBodyBuilder) {
        var blockParamTypes = b.parameters().stream().map(Value::type).toList();
        var blockReturnType = getBlockReturnType(b);
        var blockFType = FunctionType.functionType(blockReturnType, blockParamTypes);

        Body.Builder newBodyBuilder = Body.Builder.of(ancestorBodyBuilder, blockFType, CopyContext.create(ancestorBodyBuilder.entryBlock().context()));
        for (Block.Parameter param : b.parameters()) {
            var p = newBodyBuilder.entryBlock().parameter(param.type());
            newBodyBuilder.entryBlock().context().mapValue(param, p);
        }


        var letBody = Body.Builder.of(newBodyBuilder, FunctionType.functionType(blockReturnType, List.of()), CopyContext.create(newBodyBuilder.entryBlock().context()));

        AnfDialect.AnfLetOp let = transformOps(b, letBody);
        newBodyBuilder.entryBlock().op(let);
        return AnfDialect.func(b.toString(), newBodyBuilder);
    }

    //Non leaf nodes of the dominator tree
    public AnfDialect.AnfFuncOp transformDomBlock(Block b, Body.Builder ancestorBodyBuilder) {
        //This block dominates another block, and will return a function containing a letrec structure
        var blockParamTypes = b.parameters().stream().map(Value::type).toList();
        var blockReturnType = getBlockReturnType(b);
        var blockFType = FunctionType.functionType(blockReturnType, blockParamTypes);

        //Function body contains letrec and its bodies
        Body.Builder funcBodyBuilder = Body.Builder.of(ancestorBodyBuilder, blockFType, CopyContext.create(ancestorBodyBuilder.entryBlock().context()));

        for (Block.Parameter param : b.parameters()) {
            var p = funcBodyBuilder.entryBlock().parameter(param.type());
            funcBodyBuilder.entryBlock().context().mapValue(param, p);
        }

        //letrec inner body
        Body.Builder letrecBody = Body.Builder.of(funcBodyBuilder, FunctionType.functionType(blockReturnType, List.of()), CopyContext.create(funcBodyBuilder.entryBlock().context()));

        List<Block> dominates = idomMap.idominates(b);
        List<AnfDialect.AnfFuncOp> funs = new ArrayList<>();
                //dominates.stream().map((block) -> transformDomBlock(block, letrecBody)).toList();
        for (Block dblock : dominates) {
           var res = transformDomBlock(dblock, letrecBody);
           funs.add(res);
        }

        Block.Builder blockBuilder = letrecBody.entryBlock();

        for (AnfDialect.AnfFuncOp f : funs) {
            var res = blockBuilder.op(f);
            this.funMap.put(f.funcName(), res);
        }

        AnfDialect.AnfLetOp let = transformOps(b, letrecBody);
        funcBodyBuilder.entryBlock().op(let);
        return AnfDialect.func(b.toString(), funcBodyBuilder);

    }

    private TypeElement getBlockReturnType(Block b) {
        var ops = b.ops().iterator();
        while(ops.hasNext()) {
            var op = ops.next();
            if (op instanceof Op.Terminating) {
                List<Block.Reference> destBlocks = new ArrayList<>();
                if (op instanceof CoreOp.ReturnOp || op instanceof CoreOp.YieldOp) {
                    return op.resultType();
                } else if (op instanceof CoreOp.BranchOp bop) {
                    destBlocks.addAll(bop.successors());
                } else if (op instanceof CoreOp.ConditionalBranchOp cbop) {
                    destBlocks.addAll(cbop.successors());
                }
                //Traverse until we find a yield or return type, TODO: not going to try to unify types

                Set<Block> visitedBlocks = new HashSet<>();
                visitedBlocks.add(b);

                while (!destBlocks.isEmpty()) {
                    var block = destBlocks.removeFirst().targetBlock();
                    if (visitedBlocks.contains(block)) {
                        continue;
                    }

                    //Discovered a terminator with a return value, use its type
                    if (block.successors().isEmpty()) {
                        return block.ops().getLast().resultType();
                    } else {
                        visitedBlocks.add(block);
                        var newDests = block.successors().stream().filter((s) -> !visitedBlocks.contains(s.targetBlock())).toList();
                        destBlocks.addAll(newDests);
                    }
                }

            }
        }
        throw new RuntimeException("Encountered Block with no return");
    }

    private Block.Builder transformEndOp(Block.Builder b, Op op) {
        if (op instanceof Op.Terminating t) {
            switch (t) {
                case CoreOp.ConditionalBranchOp c -> {
                    var tbranch_args = c.trueBranch().arguments();
                    tbranch_args = tbranch_args.stream().map(b.context()::getValue).toList();
                    var fbranch_args = c.falseBranch().arguments();
                    fbranch_args = fbranch_args.stream().map(b.context()::getValue).toList();

                    var trueFuncName = CoreOp.constant(JavaType.J_L_STRING, c.trueBranch().targetBlock().toString());
                    var falseFuncName = CoreOp.constant(JavaType.J_L_STRING, c.falseBranch().targetBlock().toString());
                    var trueFuncVal = b.op(trueFuncName);
                    var falseFuncVal = b.op(falseFuncName);

                    List<Value> trueArgs = new ArrayList<>();
                    //trueArgs.add(funMap.get(c.trueBranch().targetBlock().toString()));
                    trueArgs.addAll(tbranch_args);

                    List<Value> falseArgs = new ArrayList<>();
                    //falseArgs.add(get(c.falseBranch().targetBlock().toString()));
                    falseArgs.addAll(fbranch_args);


                    var trueApp = AnfDialect.applyStub(c.trueBranch().targetBlock().toString(), trueArgs, getBlockReturnType(c.trueBranch().targetBlock()));
                    var falseApp = AnfDialect.applyStub(c.falseBranch().targetBlock().toString(), falseArgs, getBlockReturnType(c.falseBranch().targetBlock()));

                    var ifExp = AnfDialect.if_(b.parentBody(),
                            c.trueBranch().targetBlock().terminatingOp().resultType(),
                            b.context().getValue(c.predicate()))
                        .if_((bodyBuilder) -> bodyBuilder.op(trueApp))
                        .else_((bodyBuilder) -> bodyBuilder.op(falseApp));

                    b.op(ifExp);

                    return b;
                }
                case CoreOp.BranchOp br -> {
                    var args = br.branch().arguments().stream().toList();
                    args = args.stream().map(b.context()::getValue).toList();
                    //var targetFuncConst = CoreOp.constant(JavaType.J_L_STRING, br.branch().targetBlock().toString());
                    //var targetFuncVal = b.op(targetFuncConst);

                    List<Value> funcArgs = new ArrayList<>();
                    //funcArgs.add(funMap.get(br.branch().targetBlock().toString()));
                    funcArgs.addAll(args);

                    var funcApp = AnfDialect.applyStub(br.branch().targetBlock().toString(), funcArgs, getBlockReturnType(br.branch().targetBlock()));

                    b.op(funcApp);
                    return b;
                }
                case CoreOp.ReturnOp ro -> {
                    var rval = b.context().getValue(ro.returnValue());
                    b.op(CoreOp._yield(rval));
                    return b;
                }
                case CoreOp.YieldOp y ->  {
                    var rval = b.context().getValue(y.yieldValue());
                    b.op(CoreOp._yield(rval));
                    return b;
                }
                default -> {
                    throw new UnsupportedOperationException("Unsupported terminating op encountered: " + op);
                }
            }
        } else {
            b.op(op);
            return b;
        }
    }

    public AnfDialect.AnfLetOp transformOps(Block b, Body.Builder bodyBuilder) {
        Block.Builder blockb = bodyBuilder.entryBlock();
        return transformOps(b, blockb);
    }

    public AnfDialect.AnfLetOp transformOps(Block b, Block.Builder blockBuilder) {
        for (var op : b.ops()) {
            transformEndOp(blockBuilder, op);
        }
        return AnfDialect.let(blockBuilder.parentBody());
    }

    /*
    private Map<Block, AnfDialect.AnfFuncOp> letRecConstruction(Body b, Body.Builder bodyBuilder) {
        List<Block> workQueue = new LinkedList<>(processedFunctions.keySet().stream().map(Block::immediateDominator).toList());
        Set<Block> processed = new HashSet<>(processedFunctions.keySet());
        processed.add(b.entryBlock());

        while (!workQueue.isEmpty()) {
            Block workBlock = workQueue.removeFirst();

            if (workBlock == null || processed.contains(workBlock)) {
                continue;
            }

            //Ugly slow. Blocks dominated by this one.
            var domBlocks = b.blocks().stream().filter((block) -> block.immediateDominator() != null && block.immediateDominator().equals(workBlock)).toList();

            var unProcessedDomBlocks = domBlocks.stream().filter((block) -> !processedFunctions.containsKey(block)).toList();

            //If all dependencies aren't processed, queue them in front, requeue, and continue
            if (!unProcessedDomBlocks.isEmpty()) {
                unProcessedDomBlocks.forEach(workQueue::addLast);
                workQueue.addLast(workBlock);
                continue;
            }

            List<AnfDialect.AnfFuncOp> funcs = domBlocks.stream().map(processedFunctions::get).toList();


            //var letrecBodyBuilder = Body.Builder.of(bodyBuilder,FunctionType.VOID); //TODO: Solve Void Type
            var letrecBuilder = AnfDialect.letrec(bodyBuilder, FunctionType.VOID); // TODO: Solve Void Type

            var letRec = letrecBuilder.body(block -> {
                //Define the functions
                for (var func : funcs) {
                    block.op(func);
                }
                //LetRec "in" Body here
                transformOps(workBlock, block);
            });


            //var paramTys = workBlock.parameters().stream().map(Block.Parameter::type).toList();
            var funBuilder = AnfDialect.func(bodyBuilder,workBlock.toString(),bodyBuilder.bodyType());
            var fun = funBuilder.body(c -> c.op(letRec));

            processedFunctions.put(workBlock,fun);
        }

        return processedFunctions;
    }

    public AnfDialect.AnfFuncOp funcConstructor(Block b, Body.Builder ancestorBody)

    private void leafFunctions(Body b) {
        List<Block> leafBlocks = leafBlocks(b);
        //HashMap<Block, AnfDialect.AnfFuncOp> functions = new HashMap<>();

        for (Block leafBlock : leafBlocks) {
            Function<Body.Builder, AnfDialect.AnfFuncOp> fBuilder = (Body.Builder bodyBuilder) -> transformBlock(leafBlock, bodyBuilder);
            funcOps.put(leafBlock, fBuilder);
        }
    }

    private static List<Block> leafBlocks(Body b) {
        var idoms = b.immediateDominators();
        HashSet<Block> leafBlocks = new HashSet<>(b.blocks());
        leafBlocks.remove(b.entryBlock());
        b.blocks().forEach((block) -> {
            var dom = idoms.get(block);
            //Remove all blocks that dominate other blocks.
            if (dom != null) {
                leafBlocks.remove(dom);
            }
        });
        //Return blocks that dominate nothing. These are leaves.
        return leafBlocks.stream().toList();
    }
*/
    static class ImmediateDominatorMap {

        private final Map <Block, List<Block>> dominatesMap;
        private final Map <Block, Block> dominatorsMap;

        public ImmediateDominatorMap(Body b) {
            dominatorsMap = b.immediateDominators();
            dominatesMap = new HashMap<>();

            //Reverse the idom relation
            b.immediateDominators().forEach((dominated, dominator) -> {
               if (!dominated.equals(dominator)) {
                   dominatesMap.compute(dominator, (k, v) -> {
                       if (v == null) {
                           var newList = new ArrayList<Block>();
                           newList.add(dominated);
                           return newList;
                       } else {
                           v.add(dominated);
                           return v;
                       }
                   });
               }
            });

        }

        //Looks "down" the dominator tree toward leaves
        public List<Block> idominates(Block b) {
            return dominatesMap.getOrDefault(b, List.of());
        }

        //Looks "up" the dominator tree toward start node
        public Block idominatedBy(Block b) {
            return dominatorsMap.get(b);
        }
    }

    class ApplyStubTransformer implements OpTransformer {

        //public Map<String, Value> fmap = new HashMap<>();

        @Override
        public Block.Builder apply(Block.Builder builder, Op op) {
            switch (op) {

                /*
                case AnfDialect.AnfFuncOp af -> {
                    var name = af.funcName();
                    var val = builder.op(af);
                    fmap.put(name, val);
                }
                 */
                case AnfDialect.AnfApplyStub as -> {
                    var name = as.callSiteName;
                    var args = as.args();
                    List<Value> newArgs = new ArrayList<>();
                    newArgs.add(funMap.get(name));
                    newArgs.addAll(args);
                    builder.op(AnfDialect.apply(newArgs));
                }
                default -> {
                    builder.op(op);
                }
            }
            return builder;
        }
    }
}
