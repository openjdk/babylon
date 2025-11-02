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
package experiments;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.stream.Stream;

public class TransformState {

    @CodeReflection
    static int threeSum(int a, int b, int c) {
        return a + b * c;
    }

    //@Test
    public static void testOpToOp() {
        CoreOp.FuncOp threeSumFuncOp = getFuncOp("threeSum");
        Map<Op, Op> oldOpToNewOpMap = new HashMap<>();
        OpTransformer opTracker = (block, op) -> {
            if (op instanceof JavaOp.AddOp) {
                CopyContext cc = block.context();
                var newSubOp = JavaOp.sub(cc.getValue(op.operands().get(0)), cc.getValue(op.operands().get(1)));
                Op.Result result = block.op(newSubOp);
                cc.mapValue(op.result(), result);
                oldOpToNewOpMap.put(op,newSubOp);// <-- this maps op -> new subOp
            } else {
                var result = block.op(op);
                oldOpToNewOpMap.put(op,result.op()); //<-- this maps op ->  op'
            }
            return block;
        };

        System.out.println(threeSumFuncOp.toText());
        CoreOp.FuncOp threeSumFuncOp1 = threeSumFuncOp.transform(opTracker);
        System.out.println(threeSumFuncOp1.toText());
    }

    //@Test
    public static void testDelegate() {
        CoreOp.FuncOp threeSumFuncOp = getFuncOp("threeSum");
        JavaOp.AddOp addOp = (JavaOp.AddOp) threeSumFuncOp.elements().filter(e -> e instanceof JavaOp.AddOp).findFirst().orElseThrow();
        JavaOp.MulOp mulOp = (JavaOp.MulOp) threeSumFuncOp.elements().filter(e -> e instanceof JavaOp.MulOp).findFirst().orElseThrow();
        Map<Value, String> mapState = new HashMap<>();
        mapState.put(addOp.result(), "STATE 1");
        mapState.put(mulOp.result(), "STATE 2");

        Map<Value, String> transformedMapState = new HashMap<>();
        Map<Op, Op> transformedOpMapState = new HashMap<>();

        OpTransformer opTracker = (block, op) -> {
            if (op instanceof JavaOp.AddOp) {
                CopyContext cc = block.context();
                var newSubOp = JavaOp.sub(cc.getValue(op.operands().get(0)), cc.getValue(op.operands().get(1)));
                Op.Result result = block.op(newSubOp);
                cc.mapValue(op.result(), result);
                transformedOpMapState.put(op,newSubOp);// <-- this maps op -> new subOp
            } else {
                var result = block.op(op);
                transformedOpMapState.put(op,result.op()); //<-- this maps op ->  op'
            }
            return block;
        };
        OpTransformer t = trackingValueDelegatingTransformer(
                (block, op) -> {
                    if (op instanceof JavaOp.AddOp) {
                        CopyContext cc = block.context();
                        var newSubOp = JavaOp.sub(cc.getValue(op.operands().get(0)), cc.getValue(op.operands().get(1)));
                        Op.Result r = block.op(newSubOp);
                        cc.mapValue(op.result(), r);
                        transformedOpMapState.put(op,newSubOp);
                    } else {
                        var r = block.op(op);
                        transformedOpMapState.put(op,r.op());
                    }
                    return block;
                },
                (vIn, vOut) -> {
                    if (mapState.containsKey(vIn) && vOut != null) {
                        transformedMapState.put(vOut, mapState.get(vIn));
                    }
                });

        System.out.println(threeSumFuncOp.toText());
        print(mapState);
        CoreOp.FuncOp threeSumFuncOp1 = threeSumFuncOp.transform(opTracker);
        System.out.println(threeSumFuncOp1.toText());
        print(transformedMapState);
    }

    static void print(Map<Value, String> mappedStated) {
        mappedStated.forEach((v, s) -> {
            System.out.println(v + "[" + ((v instanceof Op.Result r ? r.op() : "") + "] -> " + s));
        });
    }

    static OpTransformer trackingValueDelegatingTransformer(
            BiFunction<Block.Builder, Op, Block.Builder> t,
            BiConsumer<Value, Value> mapAction) {
        return (block, op) -> {
            try {
                return t.apply(block, op);
            } finally {
                Value in = op.result();
                Value out = block.context().getValueOrDefault(in, null);
                mapAction.accept(in, out);
            }
        };
    }


    //@Test
    static public void testAndThen() {
        CoreOp.FuncOp threeSumFuncOp = getFuncOp("threeSum");
        JavaOp.AddOp addOp = (JavaOp.AddOp) threeSumFuncOp.elements().filter(e -> e instanceof JavaOp.AddOp).findFirst().orElseThrow();
        JavaOp.MulOp mulOp = (JavaOp.MulOp) threeSumFuncOp.elements().filter(e -> e instanceof JavaOp.MulOp).findFirst().orElseThrow();
        Map<Value, String> mapState = new HashMap<>();
        mapState.put(addOp.result(), "STATE 1");
        mapState.put(mulOp.result(), "STATE 2");

        Map<Value, String> transformedMapState = new HashMap<>();

        OpTransformer t = trackingValueAndThenTransformer(
                (block, op) -> {
                    if (op instanceof JavaOp.AddOp) {
                        CopyContext cc = block.context();
                        Op.Result r = block.op(JavaOp.sub(
                                cc.getValue(op.operands().get(0)),
                                cc.getValue(op.operands().get(1))));
                        cc.mapValue(op.result(), r);
                    } else {
                        block.op(op);
                    }
                    return block;
                },
                (vIn, vOut) -> {
                    if (mapState.containsKey(vIn) && vOut != null) {
                        transformedMapState.put(vOut, mapState.get(vIn));
                    }
                });

        System.out.println(threeSumFuncOp.toText());
        print(mapState);
        CoreOp.FuncOp threeSumFuncOp1 = threeSumFuncOp.transform(t);
        System.out.println(threeSumFuncOp1.toText());
        print(transformedMapState);
    }

    static OpTransformer trackingValueAndThenTransformer(
            OpTransformer t,
            BiConsumer<Value, Value> mapAction) {
        return OpTransformer.andThen(t, (block, op) -> {
            Value in = op.result();
            Value out = block.context().getValueOrDefault(in, null);
            mapAction.accept(in, out);
            return block;
        });
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TransformState.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }

    static public  void main(String[] args) {
        testOpToOp();
        //testDelegate();
        //testAndThen();
    }
}