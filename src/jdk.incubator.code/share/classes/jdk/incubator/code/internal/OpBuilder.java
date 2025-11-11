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

package jdk.incubator.code.internal;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.extern.*;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.core.CoreType.functionType;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

/**
 * A transformer of code models to models that build them.
 * <p>
 * A building code model when executed will construct the same code model it was transformed from.
 * Such a building code model could be transformed to bytecode and stored in class files.
 */
public class OpBuilder {

    static final JavaType J_C_E_EXTERNALIZED_OP = type(ExternalizedOp.class);

    static final MethodRef DIALECT_FACTORY_OP_FACTORY = MethodRef.method(DialectFactory.class, "opFactory",
            OpFactory.class);

    static final MethodRef DIALECT_FACTORY_TYPE_ELEMENT_FACTORY = MethodRef.method(DialectFactory.class, "typeElementFactory",
            TypeElementFactory.class);

    static final MethodRef OP_FACTORY_CONSTRUCT = MethodRef.method(OpFactory.class, "constructOp",
            Op.class, ExternalizedOp.class);

    static final MethodRef TYPE_ELEMENT_FACTORY_CONSTRUCT = MethodRef.method(TypeElementFactory.class, "constructType",
            TypeElement.class, ExternalizedTypeElement.class);

    static final MethodRef BODY_BUILDER_OF = MethodRef.method(Body.Builder.class, "of",
            Body.Builder.class, Body.Builder.class, FunctionType.class);

    static final MethodRef BODY_BUILDER_ENTRY_BLOCK = MethodRef.method(Body.Builder.class, "entryBlock",
            Block.Builder.class);

    // instance varargs
    static final MethodRef BLOCK_BUILDER_SUCCESSOR = MethodRef.method(Block.Builder.class, "successor",
            Block.Reference.class, Value[].class);

    static final MethodRef BLOCK_BUILDER_OP = MethodRef.method(Block.Builder.class, "op",
            Op.Result.class, Op.class);

    // instance varargs
    static final MethodRef BLOCK_BUILDER_BLOCK = MethodRef.method(Block.Builder.class, "block",
            Block.Builder.class, TypeElement[].class);

    static final MethodRef BLOCK_BUILDER_PARAMETER = MethodRef.method(Block.Builder.class, "parameter",
            Block.Parameter.class, TypeElement.class);

    // static varargs
    static final MethodRef FUNCTION_TYPE_FUNCTION_TYPE = MethodRef.method(CoreType.class, "functionType",
            FunctionType.class, TypeElement.class, TypeElement[].class);


    static final JavaType J_U_LIST = type(List.class);

    static final MethodRef LIST_OF_ARRAY = MethodRef.method(J_U_LIST, "of",
            J_U_LIST, array(J_L_OBJECT, 1));

    static final JavaType J_U_MAP = type(Map.class);

    static final JavaType J_U_MAP_ENTRY = type(Map.Entry.class);

    static final MethodRef MAP_ENTRY = MethodRef.method(J_U_MAP, "entry",
            J_U_MAP, J_L_OBJECT, J_L_OBJECT);

    static final MethodRef MAP_OF_ARRAY = MethodRef.method(J_U_MAP, "of",
            J_U_MAP, array(J_U_MAP_ENTRY, 1));


    static final JavaType EX_TYPE_ELEM = type(ExternalizedTypeElement.class);

    static final MethodRef EX_TYPE_ELEM_OF_LIST = MethodRef.method(EX_TYPE_ELEM, "of",
            EX_TYPE_ELEM, J_L_STRING, J_U_LIST);


    static final JavaType J_C_LOCATION = type(Location.class);

    static final MethodRef LOCATION_FROM_STRING = MethodRef.method(J_C_LOCATION, "fromString",
            J_C_LOCATION, J_L_STRING);

    static final FunctionType EXTERNALIZED_OP_F_TYPE = functionType(
            J_C_E_EXTERNALIZED_OP,
            J_L_STRING,
            J_C_LOCATION,
            J_U_LIST,
            J_U_LIST,
            type(TypeElement.class),
            J_U_MAP,
            J_U_LIST);

    static final FunctionType BUILDER_F_TYPE = functionType(type(Op.class));

    static final MethodRef LIST_EMPTY = MethodRef.method(List.class, "of", List.class);
    static final MethodRef LIST_OF_OBJECT = MethodRef.method(List.class, "of", List.class, Object.class);
    static final MethodRef MAP_EMPTY = MethodRef.method(Map.class, "of", Map.class);
    static final MethodRef MAP_OF_OBJECT_OBJECT = MethodRef.method(Map.class, "of", Map.class, Object.class, Object.class);

    static final String LIST_BUILDER_F_NAME = "::list";
    static final String MAP_BUILDER_F_NAME = "::map";
    static final String OP_BUILDER_F_NAME = "::op";

    static final JavaType J_U_LIST_VALUE = JavaType.parameterized(J_U_LIST, type(Value.class));
    static final JavaType J_U_LIST_BLOCK_REFERENCE = JavaType.parameterized(J_U_LIST, type(Block.Reference.class));
    static final JavaType J_U_LIST_BODY_BUILDER = JavaType.parameterized(J_U_LIST, type(Body.Builder.class));

    static final FunctionType LIST_BUILDER_F_TYPE = functionType(
            J_U_LIST,
            J_L_OBJECT);

    static final FunctionType MAP_BUILDER_F_TYPE = functionType(
            J_U_MAP,
            J_L_OBJECT);

    static final FunctionType OP_BUILDER_F_OVERRIDE_1 = functionType(
            type(Op.Result.class),
            type(Block.Builder.class),
            J_L_STRING, // op name
            J_C_LOCATION, // location: Location or null
            J_L_OBJECT, // operand(s): Value, List<Value> or null
            J_L_OBJECT, // successor(s): Block.Reference, List<Block.Reference> or null
            J_L_OBJECT, // result type: TypeElement or null for void
            J_L_OBJECT, // attribute(s): Map<String, Object>, Object or null
            J_L_OBJECT); // body definition(s): Body.Builder, List<Body.Builder> or null

    static final FunctionType OP_BUILDER_F_OVERRIDE_2 = functionType(
            type(Op.Result.class),
            type(Block.Builder.class),
            J_L_STRING, // op name
            INT, // location.line
            INT, // location.columnt
            J_L_OBJECT, // operand(s): Value, List<Value> or null
            J_L_OBJECT, // successor(s): Block.Reference, List<Block.Reference> or null
            J_L_OBJECT, // result type: TypeElement or null for void
            J_L_OBJECT, // attribute(s): Map<String, Object>, Object or null
            J_L_OBJECT); // body definition(s): Body.Builder, List<Body.Builder> or null

    final JavaType currentClass;

    final Map<Value, Value> valueMap;

    final Map<Block, Value> blockMap;

    final Map<ExternalizedTypeElement, Value> exTypeElementMap;

    final Map<TypeElement, Value> typeElementMap;

    final Block.Builder builder;

    final Value dialectFactory;

    final Value opFactory;

    final Value typeElementFactory;

    /**
     * Transform the given code model to one that builds it.
     * <p>
     * This method initially applies the function {@code dialectFactoryF} to
     * the block builder that is used to build resulting code model. The result
     * is a dialect factory value which is subsequently used to build operations
     * that construct type elements and operations present in the given code model.
     *
     * @param currentClass type of the current class
     * @param op the code model.
     * @param dialectFactoryF a function that builds code items to produce a dialect factory value.
     * @return the building code model.
     */
    public static FuncOp createBuilderFunction(JavaType currentClass, String name, Op op, Function<Block.Builder, Value> dialectFactoryF) {
        return new OpBuilder(currentClass, dialectFactoryF).build(name, op);
    }

    public static List<FuncOp> createSupportFunctions(JavaType currentClass) {
        MethodRef listMethodRef = MethodRef.method(currentClass, LIST_BUILDER_F_NAME, LIST_BUILDER_F_TYPE);
        return List.of(
                func(LIST_BUILDER_F_NAME, LIST_BUILDER_F_TYPE).body(bb -> {
                    Block.Builder b0 = bb.entryBlock(), b1 = b0.block(), b2 = b0.block(), b3 = b0.block(), b4 = b0.block();
                    Value arg = b0.parameters().get(0);
                    // %1 : java.type:"java.lang.Object" = constant @null;
                    // %2 : java.type:"boolean" = eq %0 %1;
                    // cbranch %2 ^block_1 ^block_2;
                    b0.op(conditionalBranch(b0.op(eq(arg, b0.op(constant(J_L_OBJECT, null)))), b1.successor(), b2.successor()));
                    // %3 : java.type:"java.util.List<java.lang.Object>" = invoke @java.ref:"java.util.List::of():java.util.List";
                    // return %3;
                    b1.op(return_(b1.op(invoke(LIST_EMPTY))));
                    // %4 : java.type:"boolean" = instanceof %0 @java.type:"java.util.List";
                    // cbranch %4 ^block_3 ^block_4;
                    b2.op(conditionalBranch(b2.op(instanceOf(J_U_LIST, arg)), b3.successor(), b4.successor()));
                    // %5 : java.type:"java.util.List" = cast %0 @java.type:"java.util.List";
                    // return %5;
                    b3.op(return_(b3.op(cast(J_U_LIST, arg))));
                    // %6 : java.type:"java.util.List<java.lang.Object>" = invoke %0 @java.ref:"java.util.List::of(java.lang.Object):java.util.List";
                    // return %6;
                    b4.op(return_(b4.op(invoke(LIST_OF_OBJECT, arg))));
                }),
                func(MAP_BUILDER_F_NAME, MAP_BUILDER_F_TYPE).body(bb -> {
                    Block.Builder b0 = bb.entryBlock(), b1 = b0.block(), b2 = b0.block(), b3 = b0.block(), b4 = b0.block();
                    Value arg = b0.parameters().get(0);
                    // %1 : java.type:"java.lang.Object" = constant @null;
                    // %2 : java.type:"boolean" = eq %0 %1;
                    // cbranch %2 ^block_1 ^block_2;
                    b0.op(conditionalBranch(b0.op(eq(arg, b0.op(constant(J_L_OBJECT, null)))), b1.successor(), b2.successor()));
                    // %3 : java.type:"java.util.Map<java.lang.Object, java.lang.Object>" = invoke @java.ref:"java.util.Map::of():java.util.Map";
                    // return %3;
                    b1.op(return_(b1.op(invoke(MAP_EMPTY))));
                    // %4 : java.type:"boolean" = instanceof %0 @java.type:"java.util.Map";
                    // cbranch %4 ^block_3 ^block_4;
                    b2.op(conditionalBranch(b2.op(instanceOf(J_U_MAP, arg)), b3.successor(), b4.successor()));
                    // %5 : java.type:"java.util.Map" = cast %0 @java.type:"java.util.Map";
                    // return %5;
                    b3.op(return_(b3.op(cast(J_U_MAP, arg))));
                    // %6 : java.type:"java.lang.String" = constant @"";
                    // %7 : java.type:"java.util.Map<java.lang.String, java.lang.Object>" = invoke %6 %0 @java.ref:"java.util.Map::of(java.lang.Object, java.lang.Object):java.util.Map";
                    // return %7;
                    b4.op(return_(b4.op(invoke(MAP_OF_OBJECT_OBJECT, b4.op(constant(J_L_STRING, "")), arg))));
                }),
                func(OP_BUILDER_F_NAME, OP_BUILDER_F_OVERRIDE_1).body(bb -> {
                    Block.Builder b = bb.entryBlock();
                    List<Block.Parameter> args = b.parameters();
                    // %15 : java.type:"jdk.incubator.code.Op$Result" = invoke %0 %14 @java.ref:"jdk.incubator.code.Block::Builder::op(jdk.incubator.code.Op):jdk.incubator.code.Op$Result";
                    b.op(return_(b.op(invoke(BLOCK_BUILDER_OP,
                            args.get(0),
                            // %14 : java.type:"jdk.incubator.code.Op" = invoke %8 %13 @java.ref:"jdk.incubator.code.extern.OpFactory::constructOp(jdk.incubator.code.extern.ExternalizedOp):jdk.incubator.code.Op";
                            b.op(invoke(OP_FACTORY_CONSTRUCT,
                                    // %8 : java.type:"jdk.incubator.code.extern.OpFactory" = field.load @java.ref:"jdk.incubator.code.dialect.java.JavaOp::JAVA_OP_FACTORY:jdk.incubator.code.extern.OpFactory";
                                    b.op(fieldLoad(FieldRef.field(JavaOp.class, "JAVA_OP_FACTORY", OpFactory.class))),
                                    // %13 : java.type:"jdk.incubator.code.extern.ExternalizedOp" = new %1 %2 %9 %10 %5 %11 %12 @java.ref:"jdk.incubator.code.extern.ExternalizedOp::(java.lang.String, jdk.incubator.code.Location, java.util.List, java.util.List, jdk.incubator.code.TypeElement, java.util.Map, java.util.List)";
                                    b.op(new_(ConstructorRef.constructor(EXTERNALIZED_OP_F_TYPE),
                                            args.get(1),
                                            args.get(2),
                                            // %9 : java.type:"java.util.List<jdk.incubator.code.Value>" = invoke %3 @java.ref:"javaapplication28.JavaApplication28::list(java.lang.Object):java.util.List";
                                            b.op(invoke(J_U_LIST_VALUE, listMethodRef, args.get(3))),
                                            // %10 : java.type:"java.util.List<jdk.incubator.code.Block$Reference>" = invoke %4 @java.ref:"javaapplication28.JavaApplication28::list(java.lang.Object):java.util.List";
                                            b.op(invoke(J_U_LIST_BLOCK_REFERENCE, listMethodRef, args.get(4))),
                                            args.get(5),
                                            // %11 : java.type:"java.util.Map" = invoke %6 @java.ref:"javaapplication28.JavaApplication28::map(java.lang.Object):java.util.Map";
                                            b.op(invoke(J_U_MAP, MethodRef.method(currentClass, MAP_BUILDER_F_NAME, MAP_BUILDER_F_TYPE), args.get(6))),
                                            // %12 : java.type:"java.util.List<jdk.incubator.code.Body::Builder>" = invoke %7 @java.ref:"javaapplication28.JavaApplication28::list(java.lang.Object):java.util.List";
                                            b.op(invoke(J_U_LIST_BODY_BUILDER, listMethodRef, args.get(7)))))))))));
                }),
                func(OP_BUILDER_F_NAME, OP_BUILDER_F_OVERRIDE_2).body(bb -> {
                    Block.Builder b = bb.entryBlock();
                    List<Block.Parameter> args = b.parameters();
                    b.op(return_(b.op(invoke(MethodRef.method(currentClass, OP_BUILDER_F_NAME, OP_BUILDER_F_OVERRIDE_1),
                            args.get(0),
                            args.get(1),
                            b.op(new_(ConstructorRef.constructor(Location.class, int.class, int.class), args.get(2), args.get(3))),
                            args.get(4),
                            args.get(5),
                            args.get(6),
                            args.get(7),
                            args.get(8)))));
                })
        );
    }

    OpBuilder(JavaType currentClass, Function<Block.Builder, Value> dialectFactoryF) {
        this.currentClass = currentClass;
        this.valueMap = new HashMap<>();
        this.blockMap = new HashMap<>();
        this.exTypeElementMap = new HashMap<>();
        this.typeElementMap = new HashMap<>();

        Body.Builder body = Body.Builder.of(null, BUILDER_F_TYPE);
        this.builder = body.entryBlock();
        this.dialectFactory = dialectFactoryF.apply(builder);
        this.opFactory = builder.op(invoke(DIALECT_FACTORY_OP_FACTORY, dialectFactory));
        this.typeElementFactory = builder.op(invoke(DIALECT_FACTORY_TYPE_ELEMENT_FACTORY, dialectFactory));
    }

    FuncOp build(String name, Op op) {
        Value ancestorBody = builder.op(constant(type(Body.Builder.class), null));
        Value result = buildOp(ancestorBody, op);
        // seal op
        builder.op(invoke(MethodRef.method(Op.class, "seal", void.class), result));
        builder.op(return_(result));

        return func(name, builder.parentBody());
    }


    Value buildOp(Value ancestorBody, Op inputOp) {
        List<Value> bodies = new ArrayList<>();
        for (Body inputBody : inputOp.bodies()) {
            Value body = buildBody(ancestorBody, inputBody);
            bodies.add(body);
        }

        List<Value> operands = new ArrayList<>();
        for (Value inputOperand : inputOp.operands()) {
            Value operand = valueMap.get(inputOperand);
            operands.add(operand);
        }

        List<Value> successors = new ArrayList<>();
        for (Block.Reference inputSuccessor : inputOp.successors()) {
            List<Value> successorArgs = new ArrayList<>();
            for (Value inputOperand : inputSuccessor.arguments()) {
                Value operand = valueMap.get(inputOperand);
                successorArgs.add(operand);
            }
            Value referencedBlock = blockMap.get(inputSuccessor.targetBlock());

            List<Value> args = new ArrayList<>();
            args.add(referencedBlock);
            args.addAll(successorArgs);
            Value successor = builder.op(invoke(
                    InvokeOp.InvokeKind.INSTANCE, true,
                    BLOCK_BUILDER_SUCCESSOR.type().returnType(),
                    BLOCK_BUILDER_SUCCESSOR, args));
            successors.add(successor);
        }

        Value opDef = buildOpDefinition(
                inputOp,
                inputOp.externalizeOpName(),
                inputOp.location(),
                operands,
                successors,
                inputOp.resultType(),
                inputOp.externalize(),
                bodies);
        return builder.op(invoke(OP_FACTORY_CONSTRUCT, opFactory, opDef));
    }


    Value buildOpDefinition(Op inputOp,
                            String name,
                            Location location,
                            List<Value> operands,
                            List<Value> successors,
                            TypeElement resultType,
                            Map<String, Object> attributes,
                            List<Value> bodies) {
        List<Value> args = List.of(
                builder.op(constant(J_L_STRING, name)),
                buildLocation(location),
                buildList(type(Value.class), operands),
                buildList(type(Block.Reference.class), successors),
                buildType(resultType),
                buildAttributeMap(inputOp, attributes),
                buildList(type(Body.Builder.class), bodies));
        return builder.op(new_(ConstructorRef.constructor(EXTERNALIZED_OP_F_TYPE), args));
    }

    Value buildLocation(Location l) {
        if (l == null) {
            return builder.op(constant(J_C_LOCATION, null));
        } else {
            return builder.op(invoke(LOCATION_FROM_STRING,
                    builder.op(constant(J_L_STRING, l.toString()))));
        }
    }

    Value buildBody(Value ancestorBodyValue, Body inputBody) {
        Value yieldType = buildType(inputBody.yieldType());
        Value bodyType = builder.op(invoke(
                InvokeOp.InvokeKind.STATIC, true,
                FUNCTION_TYPE_FUNCTION_TYPE.type().returnType(),
                FUNCTION_TYPE_FUNCTION_TYPE, List.of(yieldType)));
        Value body = builder.op(invoke(BODY_BUILDER_OF, ancestorBodyValue, bodyType));

        Value entryBlock = null;
        for (Block inputBlock : inputBody.blocks()) {
            Value block;
            if (inputBlock.isEntryBlock()) {
                block = entryBlock = builder.op(invoke(BODY_BUILDER_ENTRY_BLOCK, body));
            } else {
                assert entryBlock != null;
                block = builder.op(invoke(InvokeOp.InvokeKind.INSTANCE, true,
                        BLOCK_BUILDER_BLOCK.type().returnType(),
                        BLOCK_BUILDER_BLOCK, List.of(entryBlock)));
            }
            blockMap.put(inputBlock, block);

            for (Block.Parameter inputP : inputBlock.parameters()) {
                Value type = buildType(inputP.type());
                Value blockParameter = builder.op(invoke(BLOCK_BUILDER_PARAMETER, block, type));
                valueMap.put(inputP, blockParameter);
            }
        }

        for (Block inputBlock : inputBody.blocks()) {
            Value block = blockMap.get(inputBlock);
            for (Op inputOp : inputBlock.ops()) {
                Value op = buildOp(body, inputOp);
                Value result = builder.op(invoke(BLOCK_BUILDER_OP, block, op));
                valueMap.put(inputOp.result(), result);
            }
        }

        return body;
    }

    Value buildType(TypeElement _t) {
        return typeElementMap.computeIfAbsent(_t, t -> {
            Value exTypeElem = buildExternalizedType(t.externalize());
            return builder.op(invoke(TYPE_ELEMENT_FACTORY_CONSTRUCT, typeElementFactory, exTypeElem));
        });
    }

    Value buildExternalizedType(ExternalizedTypeElement e) {
        // Cannot use computeIfAbsent due to recursion
        if (exTypeElementMap.get(e) instanceof Value v) {
            return v;
        }

        List<Value> arguments = new ArrayList<>();
        for (ExternalizedTypeElement a : e.arguments()) {
            arguments.add(buildExternalizedType(a));
        }

        Value identifier = builder.op(constant(J_L_STRING, e.identifier()));
        Value ve;
        if (e.arguments().size() < 5) {
            MethodRef elemOf = MethodRef.method(EX_TYPE_ELEM, "of",
                    EX_TYPE_ELEM, Stream.concat(Stream.of(J_L_STRING),
                            Collections.nCopies(e.arguments().size(), EX_TYPE_ELEM).stream()).toList());
            arguments.addFirst(identifier);
            ve = builder.op(invoke(elemOf, arguments));
        } else {
            Value list = buildList(EX_TYPE_ELEM, arguments);
            ve = builder.op(invoke(EX_TYPE_ELEM_OF_LIST, identifier, list));
        }

        exTypeElementMap.put(e, ve);
        return ve;
    }

    Value buildAttributeMap(Op inputOp, Map<String, Object> attributes) {
        List<Value> keysAndValues = new ArrayList<>();
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            Value key = builder.op(constant(J_L_STRING, entry.getKey()));
            Value value = buildAttributeValue(entry.getValue());
            keysAndValues.add(key);
            keysAndValues.add(value);
        }
        return buildMap(J_L_STRING, J_L_OBJECT, keysAndValues);
    }

    private Value box(TypeElement to, Value v) {
        return builder.op(invoke(MethodRef.method(to, "valueOf", to, v.type()), v));
    }

    Value buildAttributeValue(Object value) {
        return switch (value) {
            case Boolean v -> {
                yield box(J_L_BOOLEAN, builder.op(constant(BOOLEAN, value)));
            }
            case Byte v -> {
                yield box(J_L_BYTE, builder.op(constant(BYTE, value)));
            }
            case Short v -> {
                yield box(J_L_SHORT, builder.op(constant(SHORT, value)));
            }
            case Character v -> {
                yield box(J_L_CHARACTER, builder.op(constant(CHAR, value)));
            }
            case Integer v -> {
                yield box(J_L_INTEGER, builder.op(constant(INT, value)));
            }
            case Long v -> {
                yield box(J_L_LONG, builder.op(constant(LONG, value)));
            }
            case Float v -> {
                yield box(J_L_FLOAT, builder.op(constant(FLOAT, value)));
            }
            case Double v -> {
                yield box(J_L_DOUBLE, builder.op(constant(DOUBLE, value)));
            }
            case Class<?> v -> {
                yield buildType(JavaType.type(v));
            }
            case String s -> {
                yield builder.op(constant(J_L_STRING, value));
            }
            case TypeElement f -> {
                yield buildType(f);
            }
            case InvokeOp.InvokeKind ik -> {
                FieldRef enumValueRef = FieldRef.field(InvokeOp.InvokeKind.class, ik.name(), InvokeOp.InvokeKind.class);
                yield builder.op(fieldLoad(enumValueRef));
            }
            case Object o when value == ExternalizedOp.NULL_ATTRIBUTE_VALUE -> {
                yield builder.op(fieldLoad(FieldRef.field(ExternalizedOp.class,
                        "NULL_ATTRIBUTE_VALUE", Object.class)));
            }
            default -> {
                // @@@ use the result of value.toString()?
                throw new UnsupportedOperationException("Unsupported attribute value: " + value);
            }
        };
    }


    Value buildMap(JavaType keyType, JavaType valueType, List<Value> keysAndValues) {
        JavaType mapType = parameterized(J_U_MAP, keyType, valueType);
        if (keysAndValues.size() < 21) {
            MethodRef mapOf = MethodRef.method(J_U_MAP, "of",
                    J_U_MAP, Collections.nCopies(keysAndValues.size(), J_L_OBJECT));
            return builder.op(invoke(mapType, mapOf, keysAndValues));
        } else {
            JavaType mapEntryType = parameterized(J_U_MAP_ENTRY, keyType, valueType);
            List<Value> elements = new ArrayList<>(keysAndValues.size() / 2);
            for (int i = 0; i < keysAndValues.size(); i += 2) {
                Value key = keysAndValues.get(i);
                Value value = keysAndValues.get(i + 1);
                Value entry = builder.op(invoke(mapEntryType, MAP_ENTRY, key, value));
                elements.add(entry);
            }
            Value array = buildArray(mapEntryType, elements);
            return builder.op(invoke(mapType, MAP_OF_ARRAY, array));
        }
    }


    Value buildList(JavaType elementType, List<Value> elements) {
        JavaType listType = parameterized(J_U_LIST, elementType);
        if (elements.size() < 11) {
            MethodRef listOf = MethodRef.method(J_U_LIST, "of",
                    J_U_LIST, Collections.nCopies(elements.size(), J_L_OBJECT));
            return builder.op(invoke(listType, listOf, elements));
        } else {
            Value array = buildArray(elementType, elements);
            return builder.op(invoke(listType, LIST_OF_ARRAY, array));
        }
    }


    Value buildArray(JavaType elementType, List<Value> elements) {
        Value array = builder.op(newArray(JavaType.array(elementType),
                builder.op(constant(INT, elements.size()))));
        for (int i = 0; i < elements.size(); i++) {
            builder.op(arrayStoreOp(
                    array,
                    builder.op(constant(INT, i)),
                    elements.get(i)));
        }
        return array;
    }
}
