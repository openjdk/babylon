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

import java.lang.reflect.Array;
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
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.impl.JavaTypeUtils;
import jdk.incubator.code.extern.ExternalizedOp;


/**
 * This class (lazily) initialized the symbols in the jdk.incubator.code module,
 * whose symbol is not yet available when Symtab is first constructed.
 */
public class CodeModelSymbols {

    final class Indexer<T> {
        final Map<T, Integer> map = new IdentityHashMap<>();

        int indexOf(T t) {
            return map.computeIfAbsent(t, _ -> map.size());
        }
    }

    final Symtab syms;

    final Type intArrayType;
    final Type stringArrayType;
    final Type bodyArrayType;
    final Type blockArrayType;
    final Type opArrayType;
    final Type codeModelType;
    final Type bodyType;
    final Type blockType;
    final Type opType;

    final MethodSymbol modelFuncOp;
    final MethodSymbol modelBodies;
    final MethodSymbol bodyYieldType;
    final MethodSymbol bodyBlocks;
    final MethodSymbol blockParamTypes;
    final MethodSymbol blockOps;
    final MethodSymbol opName;
    final MethodSymbol opLocation;
    final MethodSymbol opOperands;
    final MethodSymbol opSuccessors;
    final MethodSymbol opResultType;
    final MethodSymbol opAttributes;
    final MethodSymbol opBodyDefinitions;

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
        opName = atypes.methodType("name", syms.stringType, opType);
        opLocation = atypes.methodType("location", syms.stringType, opType);
        opOperands = atypes.methodType("operands", intArrayType, opType);
        opSuccessors = atypes.methodType("successors", intArrayType, opType);
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

    Attribute.Compound op(Op op, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, Queue<Body> backlog) {
        var lb = new ListBuffer<Pair<MethodSymbol, Attribute>>();
        lb.add(Pair.of(opName, stringConstant(op.externalizeOpName())));
        if (op.location() instanceof Location l) {
            lb.add(Pair.of(opLocation, stringConstant(l.toString())));
        }
        lb.add(Pair.of(opOperands, intArray(List.from(op.operands()).map(valueIndexer::indexOf))));
        lb.add(Pair.of(opSuccessors, intArray(List.from(op.successors()).map(bl -> blockIndexer.indexOf(bl.targetBlock())))));
        lb.add(Pair.of(opResultType, stringConstant(op.resultType().externalize().toString())));
        lb.add(Pair.of(opAttributes, new Attribute.Array(stringArrayType,
                List.from(op.externalize().entrySet().stream().<String>mapMulti((e, t) -> {
                    t.accept(e.getKey());
                    t.accept(AttributeMapper.toString(e.getValue()));
                }).map(this::stringConstant).toList()))));
        lb.add(Pair.of(opBodyDefinitions, intArray(List.from(op.bodies()).map(bodyIndexer::indexOf))));
        backlog.addAll(op.bodies());
        return new Attribute.Compound(opType, lb.toList());
    }

    static final class AttributeMapper {
        static String toString(Object value) {
            if (value == ExternalizedOp.NULL_ATTRIBUTE_VALUE) {
                return "null";
            }

            StringBuilder sb = new StringBuilder();
            toString(value, sb);
            return sb.toString();
        }

        static void toString(Object o, StringBuilder sb) {
            if (o.getClass().isArray()) {
                // note, while we can't parse back the array representation, this might be useful
                // for non-externalizable ops that want better string representation of array attribute values (e.g. ONNX)
                arrayToString(o, sb);
            } else {
                switch (o) {
                    case Integer i -> sb.append(i);
                    case Long l -> sb.append(l).append('L');
                    case Float f -> sb.append(f).append('f');
                    case Double d -> sb.append(d).append('d');
                    case Character c -> sb.append('\'').append(c).append('\'');
                    case Boolean b -> sb.append(b);
                    case TypeElement te -> sb.append(JavaTypeUtils.flatten(te.externalize()));
                    default -> {  // fallback to a string
                        sb.append('"');
                        quote(o.toString(), sb);
                        sb.append('"');
                    }
                }
            }
        }

        static void arrayToString(Object a, StringBuilder sb) {
            boolean first = true;
            sb.append("[");
            for (int i = 0; i < Array.getLength(a); i++) {
                if (!first) {
                    sb.append(", ");
                }

                toString(Array.get(a, i), sb);
                first = false;
            }
            sb.append("]");
        }

        static void quote(String s, StringBuilder sb) {
            for (int i = 0; i < s.length(); i++) {
                sb.append(quote(s.charAt(i)));
            }
        }

        /**
         * Escapes a character if it has an escape sequence or is
         * non-printable ASCII.  Leaves non-ASCII characters alone.
         */
        // Copied from com.sun.tools.javac.util.Convert
        static String quote(char ch) {
            return switch (ch) {
                case '\b' -> "\\b";
                case '\f' -> "\\f";
                case '\n' -> "\\n";
                case '\r' -> "\\r";
                case '\t' -> "\\t";
                case '\'' -> "\\'";
                case '\"' -> "\\\"";
                case '\\' -> "\\\\";
                default -> (isPrintableAscii(ch))
                        ? String.valueOf(ch)
                        : String.format("\\u%04x", (int) ch);
            };
        }

        /**
         * Is a character printable ASCII?
         */
        static boolean isPrintableAscii(char ch) {
            return ch >= ' ' && ch <= '~';
        }
    }

    List<Attribute> bodies(Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, Queue<Body> backlog) {
        var lb = new ListBuffer<Attribute>();
        while (!backlog.isEmpty()) {
            Body body = backlog.poll();
            body.blocks().forEach(blockIndexer::indexOf);
            lb.add(new Attribute.Compound(bodyType, List.of(Pair.of(bodyYieldType, stringConstant(body.yieldType().externalize().toString())),
                Pair.of(bodyBlocks, new Attribute.Array(blockArrayType, blocks(List.from(body.blocks()), valueIndexer, blockIndexer, bodyIndexer, backlog))))));

        }
        return lb.toList();
    }

    List<Attribute> blocks(List<Block> blocks, Indexer<Value> valueIndexer, Indexer<Block> blockIndexer, Indexer<Body> bodyIndexer, Queue<Body> backlog) {
        var lb = new ListBuffer<Attribute>();
        for (Block block : blocks) {
            lb.add(new Attribute.Compound(blockType, List.of(Pair.of(blockParamTypes, new Attribute.Array(stringArrayType, List.from(block.parameterTypes()).map(pt -> stringConstant(pt.externalize().toString())))),
                Pair.of(blockOps, new Attribute.Array(opArrayType, List.from(block.ops()).map(op -> op(op, valueIndexer, blockIndexer, bodyIndexer, backlog)))))));
        }
        return lb.toList();
    }

    Attribute.Compound toCodeModelAttribute(CoreOp.FuncOp funcOp) {
        Indexer<Value> valueIndexer = new Indexer<>();
        Indexer<Block> blockIndexer = new Indexer<>();
        Indexer<Body> bodyIndexer = new Indexer<>();
        Queue<Body> backlog = new ArrayDeque<>();
        return new Attribute.Compound(codeModelType, List.of(
                Pair.of(modelFuncOp, op(funcOp, valueIndexer, blockIndexer, bodyIndexer, backlog)),
                Pair.of(modelBodies, new Attribute.Array(bodyArrayType, bodies(valueIndexer, blockIndexer, bodyIndexer, backlog)))));
    }
}
