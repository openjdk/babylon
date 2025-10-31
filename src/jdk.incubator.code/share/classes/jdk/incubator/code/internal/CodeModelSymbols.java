/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.internal;

import java.util.ArrayDeque;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Queue;

import com.sun.tools.javac.code.Attribute;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.code.Type.MethodType;
import com.sun.tools.javac.code.Types;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Names;
import com.sun.tools.javac.util.Pair;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.OpWriter.AttributeMapper;


/**
 * This class (lazily) initialized the symbols in the jdk.incubator.code module,
 * whose symbol is not yet available when Symtab is first constructed.
 */
public final class CodeModelSymbols {

    final class Indexer<T> {
        final Map<T, Integer> map = new IdentityHashMap<>();

        int indexOf(T t) {
            return map.computeIfAbsent(t, _ -> map.size());
        }
    }

    final Symtab syms;

    final Type intArrayType,
               stringArrayType,
               bodyArrayType,
               blockArrayType,
               opArrayType,
               codeModelType,
               bodyType,
               blockType,
               blockReferenceType,
               blockReferenceArrayType,
               opType;

    final MethodSymbol modelFuncOp,
                       modelBodies,
                       bodyYieldType,
                       bodyBlocks,
                       blockParamTypes,
                       blockOps,
                       blockReferenceBlock,
                       blockReferenceArguments,
                       opName,
                       opLocation,
                       opOperands,
                       opSuccessors,
                       opResultType,
                       opAttributes,
                       opBodyDefinitions;

    CodeModelSymbols(Context context) {
        syms = Symtab.instance(context);
        Names names = Names.instance(context);
        Types types = Types.instance(context);
        ModuleSymbol jdk_incubator_code = syms.enterModule(names.jdk_incubator_code);
        intArrayType = types.makeArrayType(syms.intType);
        stringArrayType = types.makeArrayType(syms.stringType);
        codeModelType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeModel");
        bodyType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeModel$Body");
        bodyArrayType = types.makeArrayType(bodyType);
        blockType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeModel$Block");
        blockArrayType = types.makeArrayType(blockType);
        blockReferenceType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeModel$BlockReference");
        blockReferenceArrayType = types.makeArrayType(blockReferenceType);
        opType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.CodeModel$Op");
        opArrayType = types.makeArrayType(opType);
        var atypes = new Object() {
            MethodSymbol methodType(String name, Type restype, Type owner) {
                return new MethodSymbol(
                        Flags.PUBLIC | Flags.ABSTRACT,
                        names.fromString(name),
                        new MethodType(List.nil(), restype, List.nil(), syms.methodClass),
                        owner.tsym);
            }
        };
        modelFuncOp = atypes.methodType("funcOp", opType, codeModelType);
        modelBodies = atypes.methodType("bodies", bodyArrayType, codeModelType);
        bodyYieldType = atypes.methodType("yieldType", syms.stringType, bodyType);
        bodyBlocks = atypes.methodType("blocks", blockArrayType, bodyType);
        blockParamTypes = atypes.methodType("paramTypes", stringArrayType, blockType);
        blockOps = atypes.methodType("ops", opArrayType, blockType);
        blockReferenceBlock = atypes.methodType("block", syms.intType, blockReferenceType);
        blockReferenceArguments = atypes.methodType("arguments", intArrayType, blockReferenceType);
        opName = atypes.methodType("name", syms.stringType, opType);
        opLocation = atypes.methodType("location", syms.stringType, opType);
        opOperands = atypes.methodType("operands", intArrayType, opType);
        opSuccessors = atypes.methodType("successors", blockReferenceArrayType, opType);
        opResultType = atypes.methodType("resultType", syms.stringType, opType);
        opAttributes = atypes.methodType("attributes", stringArrayType, opType);
        opBodyDefinitions = atypes.methodType("bodyDefinitions", intArrayType, opType);
    }

    Attribute.Constant stringConstant(String s) {
        return new Attribute.Constant(syms.stringType, s);
    }

    Attribute.Array intArray(List<Integer> ints) {
        return new Attribute.Array(intArrayType, ints.map(i -> new Attribute.Constant(syms.intType, i)));
    }

    Attribute.Compound op(Op op, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, ListBuffer<Attribute> bodyAttributes) {
        var lb = new ListBuffer<Pair<MethodSymbol, Attribute>>();
        lb.add(Pair.of(opName, stringConstant(op.externalizeOpName())));
        if (op.location() != null) {
            lb.add(Pair.of(opLocation, stringConstant(op.location().toString())));
        }
        lb.add(Pair.of(opOperands, intArray(List.from(op.operands()).map(valueIndexer::indexOf))));
        if (!op.successors().isEmpty()) {
            lb.add(Pair.of(opSuccessors, new Attribute.Array(blockReferenceArrayType, successors(List.from(op.successors()), valueIndexer, blockIndexer))));
        }
        if (op.resultType() != JavaType.VOID) {
            valueIndexer.indexOf(op.result());
            lb.add(Pair.of(opResultType, stringConstant(op.resultType().externalize().toString())));
        }
        if (!op.externalize().isEmpty()) {
            lb.add(Pair.of(opAttributes, new Attribute.Array(stringArrayType,
                    List.from(op.externalize().entrySet().stream().<String>mapMulti((e, t) -> {
                        t.accept(e.getKey());
                        t.accept(AttributeMapper.toString(e.getValue()));
                    }).map(this::stringConstant).toList()))));
        }
        if (!op.bodies().isEmpty()) {
            var bodies = List.from(op.bodies());
            bodies(bodies, valueIndexer, blockIndexer, bodyIndexer, bodyAttributes);
            lb.add(Pair.of(opBodyDefinitions, intArray(bodies.map(bodyIndexer::indexOf))));
        }
        return new Attribute.Compound(opType, lb.toList());
    }

    void bodies(List<Body> bodies, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, ListBuffer<Attribute> bodyAttributes) {
        for (Body body : bodies) {
            bodyIndexer.indexOf(body);
            var nested = new ListBuffer<Attribute>();
            bodyAttributes.add(new Attribute.Compound(bodyType, List.of(Pair.of(bodyYieldType, stringConstant(body.yieldType().externalize().toString())),
                Pair.of(bodyBlocks, new Attribute.Array(blockArrayType, blocks(List.from(body.blocks()), valueIndexer, blockIndexer, bodyIndexer, nested))))));
            bodyAttributes.appendList(nested);
        }
    }

    List<Attribute> blocks(List<Block> blocks, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, ListBuffer<Attribute> bodyAttributes) {
        var lb = new ListBuffer<Attribute>();
        for (Block block : blocks) {
            blockIndexer.indexOf(block);
            block.parameters().forEach(valueIndexer::indexOf);
            lb.add(new Attribute.Compound(blockType,
                    List.of(Pair.of(blockParamTypes, new Attribute.Array(stringArrayType, List.from(block.parameterTypes()).map(pt -> stringConstant(pt.externalize().toString())))),
                    Pair.of(blockOps, new Attribute.Array(opArrayType, List.from(block.ops()).map(op -> op(op, valueIndexer, blockIndexer, bodyIndexer, bodyAttributes)))))));
        }
        return lb.toList();
    }

    List<Attribute> successors(List<Block.Reference> successors, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer) {
        var lb = new ListBuffer<Attribute>();
        for (Block.Reference succ : successors) {
            lb.add(new Attribute.Compound(blockReferenceType, List.of(
                    Pair.of(blockReferenceBlock, new Attribute.Constant(syms.intType, blockIndexer.indexOf(succ.targetBlock()))),
                    Pair.of(blockReferenceArguments, intArray(List.from(succ.arguments()).map(valueIndexer::indexOf))))));
        }
        return lb.toList();
    }

    Attribute.Compound toCodeModelAnnotation(CoreOp.FuncOp funcOp) {
        System.out.println(funcOp.toText());
        Indexer<Value> valueIndexer = new Indexer<>();
        Indexer<Block> blockIndexer = new Indexer<>();
        Indexer<Body> bodyIndexer = new Indexer<>();
        ListBuffer<Attribute> bodies = new ListBuffer<>();
        return new Attribute.Compound(codeModelType, List.of(
                Pair.of(modelFuncOp, op(funcOp, valueIndexer, blockIndexer, bodyIndexer, bodies)),
                Pair.of(modelBodies, new Attribute.Array(bodyArrayType, bodies.toList()))));
    }
}
