package java.lang.reflect.code.writer;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.OpDefinition;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.*;
import java.util.*;

import static java.lang.reflect.code.op.CoreOps.*;
import static java.lang.reflect.code.type.FunctionType.functionType;
import static java.lang.reflect.code.type.JavaType.*;

/**
 * A transformer of code models to models that build them.
 * <p>
 * A building code model when executed will construct the same code model it was transformed from.
 * Such a building code model could be transformed to bytecode and stored in class files.
 */
public class OpBuilder {

    static final JavaType J_C_O_OP_DEFINITION = type(OpDefinition.class);

    static final MethodRef OP_FACTORY_CONSTRUCT = MethodRef.method(OpFactory.class, "constructOp",
            Op.class, OpDefinition.class);

    static final MethodRef TYPE_ELEMENT_FACTORY_CONSTRUCT = MethodRef.method(TypeElementFactory.class, "constructType",
            TypeElement.class, TypeDefinition.class);

    static final MethodRef TYPE_DEFINITION_OF_STRING = MethodRef.method(TypeDefinition.class, "ofString",
            TypeDefinition.class, String.class);

    static final MethodRef BODY_BUILDER_OF = MethodRef.method(Body.Builder.class, "of",
            Body.Builder.class, Body.Builder.class, FunctionType.class);

    static final MethodRef BODY_BUILDER_ENTRY_BLOCK = MethodRef.method(Body.Builder.class, "entryBlock",
            Block.Builder.class);

    static final MethodRef BLOCK_BUILDER_SUCCESSOR = MethodRef.method(Block.Builder.class, "successor",
            Block.Reference.class, Value[].class);

    static final MethodRef BLOCK_BUILDER_OP = MethodRef.method(Block.Builder.class, "op",
            Op.Result.class, Op.class);

    static final MethodRef BLOCK_BUILDER_BLOCK = MethodRef.method(Block.Builder.class, "block",
            Block.Builder.class, TypeElement[].class);

    static final MethodRef BLOCK_BUILDER_PARAMETER = MethodRef.method(Block.Builder.class, "parameter",
            Block.Parameter.class, TypeElement.class);

    static final MethodRef FUNCTION_TYPE_FUNCTION_TYPE = MethodRef.method(FunctionType.class, "functionType",
            FunctionType.class, TypeElement.class, TypeElement[].class);


    static final JavaType J_U_LIST = type(List.class);

    static final MethodRef LIST_OF_ARRAY = MethodRef.method(J_U_LIST, "of",
            J_U_LIST, type(J_L_OBJECT, 1));

    static final JavaType J_U_MAP = type(Map.class);

    static final JavaType J_U_HASH_MAP = type(HashMap.class);

    static final JavaType J_U_MAP_ENTRY = type(Map.Entry.class);

    static final MethodRef MAP_ENTRY = MethodRef.method(J_U_MAP, "entry",
            J_U_MAP, J_L_OBJECT, J_L_OBJECT);

    static final MethodRef MAP_OF = MethodRef.method(J_U_MAP, "of",
            J_U_MAP);

    static final MethodRef MAP_OF_ARRAY = MethodRef.method(J_U_MAP, "of",
            J_U_MAP, type(J_U_MAP_ENTRY, 1));

    static final MethodRef MAP_PUT = MethodRef.method(J_U_MAP, "put",
            J_L_OBJECT, J_L_OBJECT, J_L_OBJECT);


    static final FunctionType OP_DEFINITION_F_TYPE = functionType(
            J_C_O_OP_DEFINITION,
            J_L_STRING,
            J_U_LIST,
            J_U_LIST,
            type(TypeElement.class),
            J_U_MAP,
            J_U_LIST);

    static final FunctionType BUILDER_F_TYPE = functionType(type(Op.class),
            type(OpFactory.class),
            type(TypeElementFactory.class));


    Map<Value, Value> valueMap;

    Map<Block, Value> blockMap;

    Block.Builder builder;

    Value opFactory;

    Value typeElementFactory;

    /**
     * Transform the given code model to one that builds it.
     *
     * @param op the code model
     * @return the building code model.
     */
    public static FuncOp createBuilderFunction(Op op) {
        return new OpBuilder().build(op);
    }

    OpBuilder() {
        this.valueMap = new HashMap<>();
        this.blockMap = new HashMap<>();
    }

    FuncOp build(Op op) {
        Body.Builder body = Body.Builder.of(null, BUILDER_F_TYPE);

        builder = body.entryBlock();
        opFactory = builder.parameters().get(0);
        typeElementFactory = builder.parameters().get(1);

        Value ancestorBody = builder.op(constant(type(Body.Builder.class), null));
        Value result = buildOp(ancestorBody, op);
        builder.op(_return(result));

        return func("builder." + op.opName(), body);
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
            Value successor = builder.op(invoke(BLOCK_BUILDER_SUCCESSOR, args));
            successors.add(successor);
        }

        Value opDef = buildOpDefinition(
                inputOp.opName(),
                operands,
                successors,
                inputOp.resultType(),
                inputOp.attributes(),
                bodies);
        return builder.op(invoke(OP_FACTORY_CONSTRUCT, opFactory, opDef));
    }


    Value buildOpDefinition(String name,
                            List<Value> operands,
                            List<Value> successors,
                            TypeElement resultType,
                            Map<String, Object> attributes,
                            List<Value> bodies) {
        List<Value> args = List.of(
                builder.op(constant(J_L_STRING, name)),
                buildList(type(Value.class), operands),
                buildList(type(Block.Reference.class), successors),
                buildType(resultType),
                buildAttributeMap(attributes),
                buildList(type(Body.Builder.class), bodies));
        return builder.op(_new(OP_DEFINITION_F_TYPE, args));
    }

    Value buildBody(Value ancestorBodyValue, Body inputBody) {
        Value yieldType = buildType(inputBody.yieldType());
        Value bodyType = builder.op(invoke(FUNCTION_TYPE_FUNCTION_TYPE, yieldType));
        Value body = builder.op(invoke(BODY_BUILDER_OF, ancestorBodyValue, bodyType));

        Value entryBlock = null;
        for (Block inputBlock : inputBody.blocks()) {
            Value block;
            if (inputBlock.isEntryBlock()) {
                block = entryBlock = builder.op(invoke(BODY_BUILDER_ENTRY_BLOCK, body));
            } else {
                assert entryBlock != null;
                block = builder.op(invoke(BLOCK_BUILDER_BLOCK, entryBlock));
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

    Value buildType(TypeElement t) {
        Value typeString = builder.op(constant(J_L_STRING, t.toString()));
        Value typeDef = builder.op(invoke(TYPE_DEFINITION_OF_STRING, typeString));
        return builder.op(invoke(TYPE_ELEMENT_FACTORY_CONSTRUCT, typeElementFactory, typeDef));
    }

    Value buildAttributeMap(Map<String, Object> attributes) {
        List<Value> keysAndValues = new ArrayList<>();
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            Value key = builder.op(constant(J_L_STRING, entry.getKey()));
            Value value = buildAttributeValue(entry.getValue());
            keysAndValues.add(key);
            keysAndValues.add(value);
        }
        return buildMap(J_L_STRING, J_L_OBJECT, keysAndValues);
    }

    Value buildAttributeValue(Object value) {
        return switch (value) {
            case String s -> {
                yield builder.op(constant(J_L_STRING, value));
            }
            case Integer i -> {
                yield builder.op(constant(INT, value));
            }
            default -> {
                throw new UnsupportedOperationException(value.getClass().toString());
            }
        };
    }


    Value buildMap(JavaType keyType, JavaType valueType, List<Value> keysAndValues) {
        JavaType mapType = type(J_U_MAP, keyType, valueType);
        if (keysAndValues.isEmpty()) {
            return builder.op(invoke(MAP_OF));
        } else {
            Value map = builder.op(_new(mapType, functionType(J_U_HASH_MAP)));
            for (int i = 0; i < keysAndValues.size(); i += 2) {
                Value key = keysAndValues.get(i);
                Value value = keysAndValues.get(i + 1);
                builder.op(invoke(MAP_PUT, map, key, value));
            }
            return map;
        }
    }


    Value buildList(JavaType elementType, List<Value> elements) {
        JavaType listType = type(J_U_LIST, elementType);
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
        Value array = builder.op(newArray(elementType,
                builder.op(constant(INT, elements.size()))));
        for (int i = 0; i < elements.size(); i++) {
            builder.op(arrayStoreOp(array, elements.get(i),
                    builder.op(constant(INT, i))));
        }
        return array;
    }
}
