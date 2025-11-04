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

import java.util.Map;

import com.sun.tools.javac.code.Attribute;
import com.sun.tools.javac.code.Flags;
import com.sun.tools.javac.code.Symbol.DynamicMethodSymbol;
import com.sun.tools.javac.code.Symbol.MethodSymbol;
import com.sun.tools.javac.code.Symbol.ModuleSymbol;
import com.sun.tools.javac.code.Symtab;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.code.Type.MethodType;
import com.sun.tools.javac.code.Types;
import com.sun.tools.javac.jvm.PoolConstant.LoadableConstant;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Name;
import com.sun.tools.javac.util.Names;
import com.sun.tools.javac.util.Pair;
import java.util.HashMap;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedTypeElement;
import jdk.incubator.code.extern.impl.AttributeMapper;


 /**
  * Initializes and provides javac symbols and helper builders to encode a
  * jdk.incubator.code model as a javac attribute compatible with the
  * {@link CodeModel} annotation format.
  * <p>
  * Instances of this class set up the necessary symbol and type handles and
  * offer utilities to transform high-level code model elements (ops, bodies,
  * blocks, successors) into {@link Attribute} representations.
  */
public final class CodeModelSymbols {

    final class Indexer<T> {
        final Map<T, Integer> map = new HashMap<>();

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
               opType,
               typeType,
               typeArrayType,
               opParserType,
               fromAnnotation;

    final MethodSymbol modelFuncOp,
                       modelBodies,
                       modelTypes,
                       bodyYieldType,
                       bodyBlocks,
                       blockParamTypes,
                       blockOps,
                       blockReferenceTargetBlock,
                       blockReferenceArguments,
                       opName,
                       opOperands,
                       opSuccessors,
                       opResultType,
                       opDefaultAttribute,
                       opSourceRef,
                       opLocation,
                       opAttributes,
                       opBodyDefinitions,
                       typeIdentifier,
                       typeArguments,
                       bsmFromAnnotation;

    /**
     * Constructs and initializes symbol/type handles used to build CodeModel attributes. This performs module/class
     * symbol lookups and synthesizes method signatures to align with the CodeModel annotation schema.
     *
     * @param context the javac compilation context
     */
    public CodeModelSymbols(Context context) {
        syms = Symtab.instance(context);
        Names names = Names.instance(context);
        Types types = Types.instance(context);
        ModuleSymbol jdk_incubator_code = syms.enterModule(names.jdk_incubator_code);
        intArrayType = types.makeArrayType(syms.intType);
        stringArrayType = types.makeArrayType(syms.stringType);
        codeModelType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel");
        bodyType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel$Body");
        bodyArrayType = types.makeArrayType(bodyType);
        blockType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel$Block");
        blockArrayType = types.makeArrayType(blockType);
        blockReferenceType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel$BlockReference");
        blockReferenceArrayType = types.makeArrayType(blockReferenceType);
        opType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel$Op");
        opArrayType = types.makeArrayType(opType);
        typeType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.internal.CodeModel$Type");
        typeArrayType = types.makeArrayType(typeType);
        opParserType = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.extern.OpParser");
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
        modelTypes = atypes.methodType("types", typeArrayType, codeModelType);
        bodyYieldType = atypes.methodType("yieldType", syms.intType, bodyType);
        bodyBlocks = atypes.methodType("blocks", blockArrayType, bodyType);
        blockParamTypes = atypes.methodType("paramTypes", intArrayType, blockType);
        blockOps = atypes.methodType("ops", opArrayType, blockType);
        blockReferenceTargetBlock = atypes.methodType("targetBlock", syms.intType, blockReferenceType);
        blockReferenceArguments = atypes.methodType("arguments", intArrayType, blockReferenceType);
        opName = atypes.methodType("name", syms.stringType, opType);
        opOperands = atypes.methodType("operands", intArrayType, opType);
        opSuccessors = atypes.methodType("successors", blockReferenceArrayType, opType);
        opResultType = atypes.methodType("resultType", syms.intType, opType);
        opDefaultAttribute = atypes.methodType("defaultAttribute", syms.stringType, opType);
        opSourceRef = atypes.methodType("sourceRef", syms.stringType, opType);
        opLocation = atypes.methodType("location", intArrayType, opType);
        opAttributes = atypes.methodType("attributes", stringArrayType, opType);
        opBodyDefinitions = atypes.methodType("bodyDefinitions", intArrayType, opType);
        typeIdentifier = atypes.methodType("identifier", syms.stringType, typeType);
        typeArguments = atypes.methodType("arguments", intArrayType, typeType);
        var opT = syms.enterClass(jdk_incubator_code, "jdk.incubator.code.Op");
        fromAnnotation = new MethodType(List.nil(), opT, List.nil(), syms.methodClass);
        bsmFromAnnotation = new MethodSymbol(Flags.PUBLIC | Flags.STATIC,
                names.fromString("fromAnnotation"),
                new MethodType(List.of(syms.methodHandleLookupType, syms.stringType, syms.methodTypeType),
                        syms.enterClass(syms.java_base, "java.lang.invoke.CallSite"), List.nil(), syms.methodClass),
                opParserType.tsym);
    }

    /**
     * Produces the top-level CodeModel attribute for a function operation, including the root op and the flattened set
     * of referenced bodies.
     *
     * @param funcOp the function operation to encode
     * @return the compound attribute suitable for storage in the CodeModel annotation
     */
    public Attribute.Compound toCodeModelAttribute(CoreOp.FuncOp funcOp) {
        Indexer<Value> valueIndexer = new Indexer<>();
        Indexer<Block> blockIndexer = new Indexer<>();
        Indexer<Body> bodyIndexer = new Indexer<>();
        Indexer<ExternalizedTypeElement> typeIndexer = new Indexer<>();
        ListBuffer<Attribute> allBodies = new ListBuffer<>();
        ListBuffer<Attribute> allTypes = new ListBuffer<>();
        return new Attribute.Compound(codeModelType, List.of(
                Pair.of(modelFuncOp, op(funcOp, valueIndexer, blockIndexer, bodyIndexer, typeIndexer, allBodies, allTypes)),
                Pair.of(modelBodies, new Attribute.Array(bodyArrayType, allBodies.toList())),
                Pair.of(modelTypes, new Attribute.Array(typeArrayType, allTypes.toList()))));
    }

    /**
     * Creates an invokedynamic bootstrap-bound dynamic method symbol for a given name, using the {@code fromAnnotation}
     * bootstrap.
     *
     * @param methodName the dynamic method name
     * @return a {@link DynamicMethodSymbol} configured with the bootstrap and signature
     */
    public DynamicMethodSymbol indyType(Name methodName) {
        return new DynamicMethodSymbol(
                methodName,
                syms.noSymbol,
                bsmFromAnnotation.asHandle(),
                fromAnnotation,
                new LoadableConstant[0]);
    }

    Attribute.Constant stringConstant(String s) {
        return new Attribute.Constant(syms.stringType, s);
    }

    Attribute.Array intArray(List<Integer> ints) {
        return new Attribute.Array(intArrayType, ints.map(i -> new Attribute.Constant(syms.intType, i)));
    }

    Attribute.Compound op(Op op,
                          Indexer<Value> valueIndexer,
                          Indexer<Block> blockIndexer,
                          Indexer<Body> bodyIndexer,
                          Indexer<ExternalizedTypeElement> typeIndexer,
                          ListBuffer<Attribute> allBodies,
                          ListBuffer<Attribute> allTypes) {
        var lb = new ListBuffer<Pair<MethodSymbol, Attribute>>();
        lb.add(Pair.of(opName, stringConstant(op.externalizeOpName())));
        if (!op.operands().isEmpty()) {
            lb.add(Pair.of(opOperands, intArray(List.from(op.operands()).map(valueIndexer::indexOf))));
        }
        if (!op.successors().isEmpty()) {
            lb.add(Pair.of(opSuccessors, new Attribute.Array(blockReferenceArrayType, successors(List.from(op.successors()), valueIndexer, blockIndexer))));
        }
        if (op.resultType() != JavaType.VOID) {
            valueIndexer.indexOf(op.result());
            lb.add(Pair.of(opResultType, type(op.resultType().externalize(), typeIndexer, allTypes)));
        }
        if (op.location() instanceof Location loc) {
            if (loc.sourceRef() != null) {
                lb.add(Pair.of(opSourceRef, stringConstant(loc.sourceRef())));
            }
            lb.add(Pair.of(opLocation, intArray(List.of(loc.line(), loc.column()))));
        }
        ListBuffer<Attribute> attrs = new ListBuffer<>();
        op.externalize().entrySet().forEach(me -> {
            if (me.getKey().isEmpty()) {
                lb.add(Pair.of(opDefaultAttribute, stringConstant(AttributeMapper.toString(me.getValue()))));
            } else {
                attrs.add(stringConstant(me.getKey()));
                attrs.add(stringConstant(AttributeMapper.toString(me.getValue())));
            }
        });
        if (!attrs.isEmpty()) {
            lb.add(Pair.of(opAttributes, new Attribute.Array(stringArrayType, attrs.toList())));
        }
        if (!op.bodies().isEmpty()) {
            var bodies = List.from(op.bodies());
            bodies(bodies, valueIndexer, blockIndexer, bodyIndexer, typeIndexer, allBodies, allTypes);
            lb.add(Pair.of(opBodyDefinitions, intArray(bodies.map(bodyIndexer::indexOf))));
        }
        return new Attribute.Compound(opType, lb.toList());
    }

    void bodies(List<Body> opBodies,
                Indexer<Value> valueIndexer,
                Indexer<Block> blockIndexer,
                Indexer<Body> bodyIndexer,
                Indexer<ExternalizedTypeElement> typeIndexer,
                ListBuffer<Attribute> allBodies,
                ListBuffer<Attribute> allTypes) {
        for (Body body : opBodies) {
            bodyIndexer.indexOf(body);
            var nested = new ListBuffer<Attribute>();
            allBodies.add(new Attribute.Compound(bodyType, List.of(Pair.of(bodyYieldType, type(body.yieldType().externalize(), typeIndexer, allTypes)),
                Pair.of(bodyBlocks, new Attribute.Array(blockArrayType, blocks(List.from(body.blocks()), valueIndexer, blockIndexer, bodyIndexer, typeIndexer, nested, allTypes))))));
            allBodies.appendList(nested);
        }
    }

    List<Attribute> blocks(List<Block> blocks,
                           Indexer<Value> valueIndexer,
                           Indexer<Block> blockIndexer,
                           Indexer<Body> bodyIndexer,
                           Indexer<ExternalizedTypeElement> typeIndexer,
                           ListBuffer<Attribute> allBodies,
                           ListBuffer<Attribute> allTypes) {
        var lb = new ListBuffer<Attribute>();
        for (Block block : blocks) {
            blockIndexer.indexOf(block);
            block.parameters().forEach(valueIndexer::indexOf);
            var args = new ListBuffer<Pair<MethodSymbol, Attribute>>();
            args.add(Pair.of(blockOps, new Attribute.Array(opArrayType, List.from(block.ops()).map(op -> op(op, valueIndexer, blockIndexer, bodyIndexer, typeIndexer, allBodies, allTypes)))));
            if (!block.parameterTypes().isEmpty()) {
                args.add(Pair.of(blockParamTypes, new Attribute.Array(stringArrayType, List.from(block.parameterTypes()).map(pt -> type(pt.externalize(), typeIndexer, allTypes)))));
            }
            lb.add(new Attribute.Compound(blockType, args.toList()));
        }
        return lb.toList();
    }

    Attribute.Constant type(ExternalizedTypeElement type,
                            Indexer<ExternalizedTypeElement> typeIndexer,
                            ListBuffer<Attribute> allTypes) {
        var args = type.arguments().stream().map(et -> type(et, typeIndexer, allTypes)).toArray(Attribute[]::new);
        int index = typeIndexer.indexOf(type);
        if (index == allTypes.size()) {
            allTypes.add(new Attribute.Compound(typeType, List.of(
                    Pair.of(typeIdentifier, stringConstant(type.identifier())),
                    Pair.of(typeArguments, new Attribute.Array(intArrayType, args)))));
        }
        return new Attribute.Constant(syms.intType, index);
    }

    List<Attribute> successors(List<Block.Reference> successors,
                               Indexer<Value> valueIndexer,
                               Indexer<Block> blockIndexer) {
        var lb = new ListBuffer<Attribute>();
        for (Block.Reference succ : successors) {
            var args = new ListBuffer<Pair<MethodSymbol, Attribute>>();
            args.add(Pair.of(blockReferenceTargetBlock, new Attribute.Constant(syms.intType, blockIndexer.indexOf(succ.targetBlock()))));
            if (!succ.arguments().isEmpty()) {
                args.add(Pair.of(blockReferenceArguments, intArray(List.from(succ.arguments()).map(valueIndexer::indexOf))));
            }
            lb.add(new Attribute.Compound(blockReferenceType, args.toList()));
        }
        return lb.toList();
    }
}
