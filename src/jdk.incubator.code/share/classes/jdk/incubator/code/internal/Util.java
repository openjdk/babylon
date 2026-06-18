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

import jdk.incubator.code.Body;
import jdk.incubator.code.extern.ExternalizedOp;

public final class Util {

    private Util() {
    }

    public static void requireNoOperands(ExternalizedOp def) {
        int count = def.operands().size();
        if (count != 0) {
            throw structuralException(def, "requires no operands, found %d".formatted(count));
        }
    }

    public static void requireOperands(ExternalizedOp def, int expCount) {
        int count = def.operands().size();
        if (count != expCount) {
            throw structuralException(def, "requires %d operand%s, found %d".formatted(expCount, expCount == 1 ? "" : "s", count));
        }
    }

    public static void requireOperands(ExternalizedOp def, int expCount1, int expCount2) {
        int count = def.operands().size();
        if (count != expCount1 && count != expCount2) {
            throw structuralException(def, "requires %d or %d operands, found %d".formatted(expCount1, expCount2, count));
        }
    }

    public static Body.Builder requireSingleBody(ExternalizedOp def) {
        var bd = def.bodyDefinitions();
        if (bd.size() != 1) {
            throw structuralException(def, "requires single body, found %d".formatted(bd.size()));
        }
        return bd.getFirst();
    }

    public static void requireBodies(ExternalizedOp def, int expCount) {
        int count = def.bodyDefinitions().size();
        if (count != expCount) {
            throw structuralException(def, "requires %d bod%s, found %d".formatted(expCount, expCount == 1 ? "y" : "ies", count));
        }
    }

    public static void requireMinBodies(ExternalizedOp def, int expCount) {
        int count = def.bodyDefinitions().size();
        if (count < expCount) {
            throw structuralException(def, "requires at least %d bod%s, found %d".formatted(expCount, expCount == 1 ? "y" : "ies", count));
        }
    }

    public static void requireBodies(ExternalizedOp def, int expCount1, int expCount2) {
        int count = def.bodyDefinitions().size();
        if (count != expCount1 && count != expCount2) {
            throw structuralException(def, "requires %d or %d bodies, found %d".formatted(expCount1, expCount2, count));
        }
    }

    public static void requireBodyPairs(ExternalizedOp def) {
        int count = def.bodyDefinitions().size();
        if (count < 2 || count % 2 == 1) {
            throw structuralException(def, "requires one or more body pairs, found %d".formatted(count));
        }
    }

    public static void requireSuccessors(ExternalizedOp def, int expCount) {
        int count = def.successors().size();
        if (count != expCount) {
            throw structuralException(def, "requires %d successor%s, found %d".formatted(expCount, expCount == 1 ? "" : "s", count));
        }
    }

    public static void requireMinSuccessors(ExternalizedOp def, int expCount) {
        int count = def.successors().size();
        if (count < expCount) {
            throw structuralException(def, "requires at least %d successor%s, found %d".formatted(expCount, expCount == 1 ? "" : "s", count));
        }
    }

    public static Object requireAttribute(ExternalizedOp def, String attributeName, boolean isDefaultAttribute) {
        if (isDefaultAttribute && def.attributes().containsKey("")) {
            return def.attributes().get("");
        }
        if (def.attributes().containsKey(attributeName)) {
            return def.attributes().get(attributeName);
        }
        throw structuralException(def, "requires attribute %s".formatted(attributeName));
    }

    public static IllegalStateException structuralException(ExternalizedOp def, String msg) {
        return new IllegalStateException("Operation " + def.name() + " " + msg);
    }

    public static UnsupportedOperationException unsupportedAttributeValueException(ExternalizedOp def, String attributeName, Object value) {
        return new UnsupportedOperationException(
                "Operation %s attribute %s has unsupported value: %s" .formatted(def.name(), attributeName, value));
    }
}
