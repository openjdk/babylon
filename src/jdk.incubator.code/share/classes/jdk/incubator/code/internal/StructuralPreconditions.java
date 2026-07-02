/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.util.List;
import java.util.Optional;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;

public final class StructuralPreconditions {

    private StructuralPreconditions() {
    }

    public static void requireNoOperands(ExternalizedOp def) {
        requireOperands(def, 0);
    }

    public static Value requireSingleOperand(ExternalizedOp def) {
        return requireOperands(def, 1).getFirst();
    }

    public static List<Value> requireOperands(ExternalizedOp def, int expCount) {
        List<Value> operands = def.operands();
        if (operands.size() != expCount) {
            throw structuralException(def.name(), "requires %d operand%s, found %d".formatted(expCount, expCount == 1 ? "" : "s", operands.size()));
        }
        return operands;
    }

    public static List<Value> requireOperands(ExternalizedOp def, int expCount1, int expCount2) {
        int count = def.operands().size();
        if (count != expCount1 && count != expCount2) {
            throw structuralException(def.name(), "requires %d or %d operands, found %d".formatted(expCount1, expCount2, count));
        }
        return def.operands();
    }

    public static Body.Builder requireSingleBody(ExternalizedOp def) {
        return requireBodies(def, 1).getFirst();
    }

    public static List<Body.Builder> requireBodies(ExternalizedOp def, int expCount) {
        List<Body.Builder> bodies = def.bodyDefinitions();
        if (bodies.size() != expCount) {
            throw structuralException(def.name(), "requires %d bod%s, found %d".formatted(expCount, expCount == 1 ? "y" : "ies", bodies.size()));
        }
        return bodies;
    }

    public static List<Body.Builder> requireMinBodies(ExternalizedOp def, int expCount) {
        List<Body.Builder> bodies = def.bodyDefinitions();
        if (bodies.size() < expCount) {
            throw structuralException(def.name(), "requires at least %d bod%s, found %d".formatted(expCount, expCount == 1 ? "y" : "ies", bodies.size()));
        }
        return bodies;
    }

    public static List<Body.Builder> requireBodyPairs(String opName, List<Body.Builder> bodies) {
        int count = bodies.size();
        if (count < 2 || count % 2 == 1) {
            throw structuralException(opName, "requires one or more body pairs, found %d".formatted(count));
        }
        return bodies;
    }

    public static Block.Reference requireSingleSuccessor(ExternalizedOp def) {
        return requireSuccessors(def, 1).getFirst();
    }

    public static List<Block.Reference> requireSuccessors(ExternalizedOp def, int expCount) {
        List<Block.Reference> succ = def.successors();
        if (succ.size() != expCount) {
            throw structuralException(def.name(), "requires %d successor%s, found %d".formatted(expCount, expCount == 1 ? "" : "s", succ.size()));
        }
        return succ;
    }

    @SuppressWarnings("unchecked")
    public static <T> T requireAttribute(ExternalizedOp def, String attributeName, boolean isDefaultAttribute, Class<T> attributeType) {
        Object attr = requireAttribute(def, attributeName, isDefaultAttribute);
        if (attributeType.isInstance(attr)) {
            return (T)attr;
        }
        throw unsupportedAttributeValueException(def, attributeName, attr);
    }

    public static Object requireAttribute(ExternalizedOp def, String attributeName, boolean isDefaultAttribute) {
        if (isDefaultAttribute && def.attributes().containsKey("")) {
            return def.attributes().get("");
        }
        if (def.attributes().containsKey(attributeName)) {
            return def.attributes().get(attributeName);
        }
        throw structuralException(def.name(), "requires attribute %s".formatted(attributeName));
    }

    public static boolean optionalBooleanAttribute(ExternalizedOp def, String attributeName) {
        return optionalAttribute(def, attributeName, false, Boolean.class).orElse(false);
    }

    @SuppressWarnings("unchecked")
    public static <T> Optional<T> optionalAttribute(ExternalizedOp def, String attributeName, boolean isDefaultAttribute, Class<T> attributeType) {
        Object attr = def.attributes().get(isDefaultAttribute && def.attributes().containsKey("") ? "" : attributeName);
        if (attr == null || attributeType.isInstance(attr)) {
            return Optional.ofNullable((T)attr);
        }
        throw unsupportedAttributeValueException(def, attributeName, attr);
    }

    public static Body.Builder requireVoidBodySignature(String opName, Body.Builder bodyC) {
        return requireBodySignature(opName, bodyC, CoreType.FUNCTION_TYPE_VOID);
    }

    public static Body.Builder requireBodySignature(String opName, Body.Builder bodyC, FunctionType signature) {
        if (!bodyC.bodySignature().equals(signature)) {
            throw structuralException(opName, "requires body signature %s, found %s".formatted(signature, bodyC.bodySignature()));
        }
        return bodyC;
    }

    public static Body.Builder requireNonVoidReturnType(String opName, Body.Builder bodyC, int parameters) {
        if (bodyC.bodySignature().returnType().equals(JavaType.VOID)) {
            throw structuralException(opName, "requires non-void return type");
        }
        if (bodyC.bodySignature().parameterTypes().size() != parameters) {
            throw structuralException(opName, "requires %d parameters, found %d".formatted(parameters, bodyC.bodySignature().parameterTypes().size()));
        }
        return bodyC;
    }

    public static Body.Builder requireVoidReturnType(String opName, Body.Builder bodyC, int parameters) {
        if (!bodyC.bodySignature().returnType().equals(JavaType.VOID)) {
            throw structuralException(opName, "requires void return type, found: %s".formatted(bodyC.bodySignature().returnType()));
        }
        if (bodyC.bodySignature().parameterTypes().size() != parameters) {
            throw structuralException(opName, "requires %d parameters, found %d".formatted(parameters, bodyC.bodySignature().parameterTypes().size()));
        }
        return bodyC;
    }

    public static Body.Builder requireNoParameters(String opName, Body.Builder bodyC) {
        if (!bodyC.bodySignature().parameterTypes().isEmpty()) {
            throw structuralException(opName, "requires no parameters, found %s".formatted(bodyC.bodySignature()));
        }
        return bodyC;
    }

    public static IllegalStateException structuralException(String opName, String msg) {
        return new IllegalStateException("Operation " + opName + " " + msg);
    }

    public static UnsupportedOperationException unsupportedAttributeValueException(ExternalizedOp def, String attributeName, Object value) {
        return new UnsupportedOperationException(
                "Operation %s attribute %s has unsupported value: %s".formatted(def.name(), attributeName, value));
    }
}
