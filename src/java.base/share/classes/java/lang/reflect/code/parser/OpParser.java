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

package java.lang.reflect.code.parser;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.code.*;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.op.*;
import java.lang.reflect.code.parser.impl.DescParser;
import java.lang.reflect.code.parser.impl.Lexer;
import java.lang.reflect.code.parser.impl.Scanner;
import java.lang.reflect.code.parser.impl.Tokens;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.lang.reflect.code.type.TypeElementFactory;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A parser of serialized code models from their textual form.
 * <p>
 * The syntactic grammar of a code mode is specified in the grammar notation, and is a subset of the grammar,
 * specified by the JLS, see section 2.4. (Except that we cannot express non-terminal symbols in italic type.)
 * <p>
 * {@snippet lang=text :
 * Operation:
 *   [Value =] Name {Operands} {Successors} {Attributes} {Bodies} ;
 *
 * Operands:
 *   ValueIdentifier {ValueIdentifier}
 *
 * Successors:
 *   Successor {Successor}
 *
 * Successor:
 *   BlockIdentifier
 *   BlockIdentifier ()
 *   BlockIdentifier ( ValueIdentifier {, ValueIdentifier} )
 *
 * Attributes:
 *   Attribute {Attribute}
 *
 * Attribute:
 *   @ AttributeValue
 *   @ Name = AttributeValue
 *
 * AttributeValue:
 *   Name
 *   StringLiteral
 *   NullLiteral
 *
 * Bodies:
 *   Body {Body}
 *
 * Body:
 *   BlockIdentifier ( ) Type -> { Operations {Block} }
 *   BlockIdentifier ( Value {, Value} ) Type -> { Operations {Block} }
 *
 * Operations:
 *   Operation {Operation}
 *
 * Block:
 *   BlockIdentifier : Operations
 *   BlockIdentifier ( ) : Operations
 *   BlockIdentifier ( Value {, Value} ) : Operations
 *
 * BlockIdentifier:
 *   ^ Identifier
 *
 * Value:
 *   ValueIdentifier : Type
 *
 * ValueIdentifier:
 *   % JavaLetterOrDigit {JavaLetterOrDigit}
 *
 * Name:
 *   Identifier
 *   Name . Identifier
 *
 * Type:
 *   same as in section 4.1 of JLS but without any annotations
 *
 * Identifier:
 *   same as in section 3 of JLS
 *
 * JavaLetterOrDigit:
 *   same as in section 3 of JLS
 *
 * StringLiteral:
 *   same as in section 3 of JLS
 *
 * NullLiteral:
 *   same as in section 3 of JLS
 * }
 */
public final class OpParser {

    static final TypeElement.ExternalizedTypeElement VOID =
            new TypeElement.ExternalizedTypeElement("void", List.of());

    /**
     * Parse a code model from its serialized textual form obtained from an input stream.
     *
     * @param opFactory the operation factory used to construct operations from their general definition
     * @param in the input stream
     * @return the list of operations
     * @throws IOException if parsing fails
     */
    public static List<Op> fromStream(OpFactory opFactory, InputStream in) throws IOException {
        return fromStream(opFactory, CoreTypeFactory.CORE_TYPE_FACTORY, in);
    }

    public static List<Op> fromStream(OpFactory opFactory, TypeElementFactory typeFactory, InputStream in) throws IOException {
        String s = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        return fromString(opFactory, typeFactory, s);
    }

    /**
     * Parse a code model from its serialized textual form obtained from an input string.
     *
     * @param opFactory the operation factory used to construct operations from their general definition
     * @param in the input string
     * @return the list of operations
     */
    public static List<Op> fromString(OpFactory opFactory, String in) {
        return parse(opFactory, CoreTypeFactory.CORE_TYPE_FACTORY, in);
    }

    public static List<Op> fromString(OpFactory opFactory, TypeElementFactory typeFactory, String in) {
        return parse(opFactory, typeFactory, in);
    }

    /**
     * Parse a code model, modeling a method, from its serialized textual form obtained from an input string.
     *
     * @param in the input string
     * @return the func operation
     */
    public static Op fromStringOfFuncOp(String in) {
        Op op = fromString(ExtendedOps.FACTORY, in).get(0);
        if (!(op instanceof CoreOps.FuncOp)) {
            throw new IllegalArgumentException("Op is not a FuncOp: " + op);
        }
        return op;
    }

    static List<Op> parse(OpFactory opFactory, TypeElementFactory typeFactory, String in) {
        Lexer lexer = Scanner.factory().newScanner(in);
        lexer.nextToken();

        List<OpNode> opNodes = new OpParser(lexer).parseNodes();

        Context c = new Context(opFactory, typeFactory);
        return opNodes.stream().map(n -> nodeToOp(n, VOID, c, null)).toList();
    }


    static final class Context {
        final Context parent;
        final OpFactory opFactory;
        final TypeElementFactory typeFactory;
        final Map<String, Value> valueMap;
        final Map<String, Block.Builder> blockMap;

        Context(Context that, boolean isolated) {
            this.parent = that;
            this.opFactory = that.opFactory;
            this.typeFactory = that.typeFactory;
            this.valueMap = isolated ? new HashMap<>() : new HashMap<>(that.valueMap);
            this.blockMap = new HashMap<>();
        }

        Context(OpFactory opFactory, TypeElementFactory typeFactory) {
            this.parent = null;
            this.opFactory = opFactory;
            this.typeFactory = typeFactory;
            this.valueMap = new HashMap<>();
            this.blockMap = new HashMap<>();
        }

        Context fork(boolean isolated) {
            return new Context(this, isolated);
        }

        void putValue(String name, Value opr) {
            valueMap.put(name, opr);
        }

        Value getValue(String name) {
            Value value = valueMap.get(name);
            if (value == null) {
                // @@@ location
                throw new IllegalArgumentException("Undeclared value referenced: " + name);
            }

            return value;
        }

        void putBlock(String name, Block.Builder bm) {
            blockMap.put(name, bm);
        }

        Block.Builder getBlock(String name) {
            Block.Builder block = blockMap.get(name);
            if (block == null) {
                // @@@ location
                throw new IllegalArgumentException("Undeclared block referenced: " + name);
            }

            return block;
        }
    }

    static Op nodeToOp(OpNode opNode, TypeElement.ExternalizedTypeElement rtype, Context c, Body.Builder ancestorBody) {
        ExternalizableOp.ExternalizedOp opdef = nodeToOpDef(opNode, rtype, c, ancestorBody);
        return c.opFactory.constructOpOrFail(opdef);
    }

    static ExternalizableOp.ExternalizedOp nodeToOpDef(OpNode opNode, TypeElement.ExternalizedTypeElement rtype, Context c, Body.Builder ancestorBody) {
        String operationName = opNode.name;
        List<Value> operands = opNode.operands.stream().map(c::getValue).toList();
        List<Block.Reference> successors = opNode.successors.stream()
                .map(n -> nodeToSuccessor(n, c)).toList();
        List<Body.Builder> bodies = opNode.bodies.stream()
                .map(n -> nodeToBody(n, c.fork(false), ancestorBody)).toList();
        return new ExternalizableOp.ExternalizedOp(operationName,
                operands,
                successors,
                c.typeFactory.constructType(rtype),
                opNode.attributes,
                bodies);
    }

    static Body.Builder nodeToBody(BodyNode n, Context c, Body.Builder ancestorBody) {
        Body.Builder body = Body.Builder.of(ancestorBody,
                // Create function type with just the return type and add parameters afterward
                FunctionType.functionType(c.typeFactory.constructType(n.rtype)));
        Block.Builder eb = body.entryBlock();

        // Create blocks upfront for forward referencing successors
        for (int i = 0; i < n.blocks.size(); i++) {
            BlockNode bn = n.blocks.get(i);
            Block.Builder b;
            if (i == 0) {
                b = body.entryBlock();
            } else {
                b = eb.block();
                c.putBlock(bn.name, b);
            }

            for (ValueNode a : bn.parameters) {
                Block.Parameter v = b.parameter(c.typeFactory.constructType(a.type));
                c.putValue(a.name, v);
            }
        }

        // Create operations
        for (int i = 0; i < n.blocks.size(); i++) {
            BlockNode bn = n.blocks.get(i);
            Block.Builder b;
            if (i == 0) {
                b = body.entryBlock();
            } else {
                b = c.getBlock(n.blocks.get(i).name);
            }

            for (OpNode on : bn.ops) {
                ValueNode r = on.result;
                if (r != null) {
                    Op.Result v = b.op(nodeToOp(on, r.type, c, body));
                    c.putValue(r.name, v);
                } else {
                    b.op(nodeToOp(on, VOID, c, body));
                }
            }
        }

        return body;
    }

    static Block.Reference nodeToSuccessor(SuccessorNode n, Context c) {
        return c.getBlock(n.blockName).successor(n.arguments().stream().map(c::getValue).toList());
    }

    // @@@ Add tokens to access position of nodes on error

    record OpNode(ValueNode result,
                  String name,
                  List<String> operands,
                  List<SuccessorNode> successors,
                  Map<String, Object> attributes,
                  List<BodyNode> bodies) {
    }

    record SuccessorNode(String blockName,
                         List<String> arguments) {
    }

    record BodyNode(TypeElement.ExternalizedTypeElement rtype,
                    List<BlockNode> blocks) {
    }

    record BlockNode(String name,
                     List<ValueNode> parameters,
                     List<OpNode> ops) {
    }

    record ValueNode(String name,
                     TypeElement.ExternalizedTypeElement type) {
    }

    final Lexer lexer;

    OpParser(Lexer lexer) {
        this.lexer = lexer;
    }

    List<OpNode> parseNodes() {
        List<OpNode> ops = new ArrayList<>();
        while (lexer.token().kind != Tokens.TokenKind.EOF) {
            ops.add(parseOperation());
        }
        return ops;
    }

    OpNode parseOperation() {
        ValueNode operationResult;
        if (lexer.is(Tokens.TokenKind.VALUE_IDENTIFIER)) {
            operationResult = parseValueNode();
            lexer.accept(Tokens.TokenKind.EQ);
        } else {
            operationResult = null;
        }

        String operationName = parseName();

        // Operands
        final List<String> operands;
        if (lexer.is(Tokens.TokenKind.VALUE_IDENTIFIER)) {
            operands = parseOperands();
        } else {
            operands = List.of();
        }

        // Successors
        // ^name(%x, %d)
        final List<SuccessorNode> successors;
        if (lexer.is(Tokens.TokenKind.CARET)) {
            successors = parseSuccessors();
        } else {
            successors = List.of();
        }

        // Attributes
        final Map<String, Object> attributes;
        if (lexer.is(Tokens.TokenKind.MONKEYS_AT)) {
            attributes = parseAttributes();
        } else {
            attributes = Map.of();
        }

        // Bodies
        List<BodyNode> bodies;
        if (lexer.is(Tokens.TokenKind.CARET) || lexer.is(Tokens.TokenKind.LPAREN)) {
            bodies = parseBodies();
        } else {
            bodies = List.of();
        }

        lexer.accept(Tokens.TokenKind.SEMI);

        return new OpNode(operationResult, operationName, operands, successors, attributes, bodies);
    }

    Map<String, Object> parseAttributes() {
        Map<String, Object> attributes = new HashMap<>();
        while (lexer.acceptIf(Tokens.TokenKind.MONKEYS_AT)) {
            String attributeName;
            if (lexer.is(Tokens.TokenKind.IDENTIFIER)) {
                attributeName = parseName();
                lexer.accept(Tokens.TokenKind.EQ);
            } else {
                attributeName = "";
            }
            Object attributeValue = parseAttributeValue();
            attributes.put(attributeName, attributeValue);
        }
        return attributes;
    }

    Object parseAttributeValue() {
        if (lexer.is(Tokens.TokenKind.IDENTIFIER)) {
            return parseName();
        }

        Object value = parseLiteral(lexer.token());
        lexer.nextToken();

        return value;
    }

    Object parseLiteral(Tokens.Token t) {
        return switch (t.kind) {
            case STRINGLITERAL -> t.stringVal();
            case NULL -> ExternalizableOp.NULL_ATTRIBUTE_VALUE;
            default -> throw lexer.unexpected();
        };
    }

    List<String> parseOperands() {
        List<String> operands = new ArrayList<>();
        while (lexer.is(Tokens.TokenKind.VALUE_IDENTIFIER)) {
            operands.add(lexer.token().name().substring(1));
            lexer.nextToken();
        }
        return operands;
    }

    List<SuccessorNode> parseSuccessors() {
        List<SuccessorNode> successors = new ArrayList<>();

        while (lexer.is(Tokens.TokenKind.CARET) && !isBody()) {
            lexer.nextToken();
            successors.add(parseSuccessor());
        }

        return successors;
    }

    // Lookahead from "^" to determine if Body
    boolean isBody() {
        assert lexer.is(Tokens.TokenKind.CARET);

        int pos = 1;
        lexer.token(pos++);
        assert lexer.token(1).kind == Tokens.TokenKind.IDENTIFIER;

        if (lexer.token(pos++).kind != Tokens.TokenKind.LPAREN) {
            return false;
        }

        Tokens.Token t;
        while ((t = lexer.token(pos++)).kind != Tokens.TokenKind.RPAREN) {
            if (t.kind == Tokens.TokenKind.EOF) {
                return false;
            } else if (t.kind == Tokens.TokenKind.COLON) {
                // Encountered Value
                return true;
            }
        }

        // Encountered return type
        return lexer.token(pos++).kind == Tokens.TokenKind.IDENTIFIER;
    }

    SuccessorNode parseSuccessor() {
        String blockName = lexer.accept(Tokens.TokenKind.IDENTIFIER).name();

        List<String> arguments = new ArrayList<>();
        if (lexer.acceptIf(Tokens.TokenKind.LPAREN) && !lexer.acceptIf(Tokens.TokenKind.RPAREN)) {
            do {
                arguments.add(lexer.accept(Tokens.TokenKind.VALUE_IDENTIFIER).name().substring(1));
            } while (lexer.acceptIf(Tokens.TokenKind.COMMA));
            lexer.accept(Tokens.TokenKind.RPAREN);
        }

        return new SuccessorNode(blockName, arguments);
    }

    List<BodyNode> parseBodies() {
        List<BodyNode> bodies = new ArrayList<>();
        while (lexer.is(Tokens.TokenKind.CARET) || lexer.is(Tokens.TokenKind.LPAREN)) {
            BodyNode body = parseBody();
            bodies.add(body);
        }
        return bodies;
    }

    BodyNode parseBody() {
        // Body name
        final String bodyName;
        if (lexer.acceptIf(Tokens.TokenKind.CARET)) {
            bodyName = lexer.accept(Tokens.TokenKind.IDENTIFIER).name();
        } else {
            bodyName = null;
        }

        // Entry block header
        List<ValueNode> arguments = parseBlockHeaderArguments(true);
        // Body return type
        TypeElement.ExternalizedTypeElement rtype = parseExTypeElem();

        lexer.accept(Tokens.TokenKind.ARROW);
        lexer.accept(Tokens.TokenKind.LBRACE);

        List<BlockNode> blocks = parseBlocks(bodyName, arguments);

        lexer.accept(Tokens.TokenKind.RBRACE);

        return new BodyNode(rtype, blocks);
    }

    List<ValueNode> parseBlockHeaderArguments(boolean isEntryBlock) {
        boolean parseArguments;
        if (isEntryBlock) {
            lexer.accept(Tokens.TokenKind.LPAREN);
            parseArguments = true;
        } else {
            parseArguments = lexer.acceptIf(Tokens.TokenKind.LPAREN);
        }
        if (!parseArguments || lexer.acceptIf(Tokens.TokenKind.RPAREN)) {
            return new ArrayList<>();
        }

        List<ValueNode> arguments = new ArrayList<>();
        do {
            arguments.add(parseValueNode());
        } while (lexer.acceptIf(Tokens.TokenKind.COMMA));
        lexer.accept(Tokens.TokenKind.RPAREN);

        return arguments;
    }

    ValueNode parseValueNode() {
        String valueName = lexer.accept(Tokens.TokenKind.VALUE_IDENTIFIER).name().substring(1);

        lexer.accept(Tokens.TokenKind.COLON);

        TypeElement.ExternalizedTypeElement type = parseExTypeElem();

        return new ValueNode(valueName, type);
    }

    List<BlockNode> parseBlocks(String entryBlockName, List<ValueNode> entryBlockArguments) {
        List<BlockNode> blocks = new ArrayList<>();

        // Entry block ops
        BlockNode entryBlock = new BlockNode(entryBlockName, entryBlockArguments, parseOperations());
        blocks.add(entryBlock);

        // Subsequent blocks
        while (lexer.acceptIf(Tokens.TokenKind.CARET)) {
            String blockName = lexer.accept(Tokens.TokenKind.IDENTIFIER).name();
            List<ValueNode> blockArguments = parseBlockHeaderArguments(false);

            lexer.accept(Tokens.TokenKind.COLON);

            BlockNode block = new BlockNode(blockName, blockArguments, parseOperations());
            blocks.add(block);
        }

        return blocks;
    }

    List<OpNode> parseOperations() {
        List<OpNode> ops = new ArrayList<>();
        while (lexer.is(Tokens.TokenKind.MONKEYS_AT) || lexer.is(Tokens.TokenKind.VALUE_IDENTIFIER) || lexer.is(Tokens.TokenKind.IDENTIFIER)) {
            OpNode op = parseOperation();
            ops.add(op);
        }
        return ops;
    }

    String parseName() {
        Tokens.Token t = lexer.accept(Tokens.TokenKind.IDENTIFIER);
        StringBuilder name = new StringBuilder();
        name.append(t.name());
        while (lexer.acceptIf(Tokens.TokenKind.DOT)) {
            name.append(Tokens.TokenKind.DOT.name);
            t = lexer.accept(Tokens.TokenKind.IDENTIFIER);
            name.append(t.name());
        }
        return name.toString();
    }

    TypeElement.ExternalizedTypeElement parseExTypeElem() {
        return DescParser.parseExTypeElem(lexer);
    }
}

