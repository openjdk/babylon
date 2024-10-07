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
    final Map<Block, Value> funMap = new HashMap<>();

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

        var selfRefParam = newBodyBuilder.entryBlock().parameter(blockFType);
        funMap.put(b, selfRefParam);

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

        //Self param
        var selfRefParam = funcBodyBuilder.entryBlock().parameter(blockFType);
        funMap.put(b, selfRefParam);

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

        var letBody = Body.Builder.of(letrecBody, letrecBody.bodyType(), CopyContext.create(letrecBody.entryBlock().context()));
        transformBlockOps(b, letBody.entryBlock());
        var let = AnfDialect.let(letBody);

        letrecBody.entryBlock().op(let);

        var letrec = AnfDialect.letrec(letrecBody);
        funcBodyBuilder.entryBlock().op(letrec);
        return AnfDialect.func(b.toString(), funcBodyBuilder);

    }

    private TypeElement getBlockReturnType(Block b) {
        var op = b.ops().getLast();
        if (op instanceof Op.Terminating) {
            List<Block.Reference> destBlocks = new ArrayList<>();
            if (op instanceof CoreOp.ReturnOp ro) {
                return ro.returnValue().type();
            } else if (op instanceof CoreOp.YieldOp yo) {
                return yo.yieldValue().type();
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
                    var o = block.ops().getLast();
                    if (o instanceof CoreOp.ReturnOp ro) {
                        return ro.returnValue().type();
                    } else if (o instanceof CoreOp.YieldOp yo) {
                        return yo.yieldValue().type();
                    } else {
                        throw new UnsupportedOperationException("Unsupported terminator encountered: " + o.opName());
                    }
                } else {
                    visitedBlocks.add(block);
                    var newDests = block.successors().stream().filter((s) -> !visitedBlocks.contains(s.targetBlock())).toList();
                    destBlocks.addAll(newDests);
                }
            }

        }

        throw new RuntimeException("Encountered Block with no return " + op.opName());
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
                    trueArgs.add(funMap.get(c.trueBranch().targetBlock()));
                    trueArgs.addAll(tbranch_args);

                    List<Value> falseArgs = new ArrayList<>();
                    falseArgs.add(funMap.get(c.falseBranch().targetBlock()));
                    falseArgs.addAll(fbranch_args);

                    var trueApp = AnfDialect.apply(trueArgs);
                    var falseApp = AnfDialect.apply(falseArgs);

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
                    funcArgs.add(funMap.get(br.branch().targetBlock()));
                    funcArgs.addAll(args);

                    var funcApp = AnfDialect.apply(funcArgs);

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
        transformBlockOps(b, blockBuilder);
        return AnfDialect.let(blockBuilder.parentBody());
    }

    public void transformBlockOps(Block b, Block.Builder blockBuilder) {
        for (var op : b.ops()) {
            transformEndOp(blockBuilder, op);
        }
    }

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
