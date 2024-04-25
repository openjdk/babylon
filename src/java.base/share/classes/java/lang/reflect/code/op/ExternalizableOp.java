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

package java.lang.reflect.code.op;

import java.lang.reflect.code.*;
import java.util.List;
import java.util.Map;

/**
 * An operation that supports externalization of its content and reconstruction
 * via an instance of {@link ExternalOpContents}.
 * <p>
 * The specific state of an externalizable operation can is externalized to a
 * map of {@link #attributes attributes}, and is reconstructed from the
 * attributes component by an instance of {@link ExternalOpContents}.
 * <p>
 * An externalizable operation could be externalized via serialization to
 * a textual representation. That textual representation could then be deserialized,
 * via parsing, into an instance of {@link ExternalOpContents} from which a new
 * externalizable operation can be reconstructed that is identical to the original.
 */
public abstract class ExternalizableOp extends Op {

    /**
     * The attribute name associated with the location attribute.
     */
    public static final String ATTRIBUTE_LOCATION = "loc";

    /**
     * The attribute value that represents the external null value.
     */
    public static final Object NULL_ATTRIBUTE_VALUE = new Object();

    /**
     * Constructs an operation by copying given operation.
     *
     * @param that the operation to copy.
     * @param cc   the copy context.
     * @implSpec The default implementation calls the constructor with the operation's name, result type, and a list
     * values computed, in order, by mapping the operation's operands using the copy context.
     */
    protected ExternalizableOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    /**
     * Constructs an operation with a name, operation result type, and list of operands.
     *
     * @param name     the operation name.
     * @param operands the list of operands, a copy of the list is performed if required.
     */
    protected ExternalizableOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    /**
     * Constructs an operation from its externalized operation definition.
     *
     * @param def the operation definition.
     * @implSpec This implementation invokes the {@link Op#Op(String, List) constructor}
     * accepting the non-optional components of the operation definition, {@code name}, {@code resultType},
     * and {@code operands}:
     * <pre> {@code
     *  this(def.name(), def.resultType(), def.operands());
     * }</pre>
     * If the attributes component of the operation definition is copied as if by {@code Map.copyOf}.
     */
    protected ExternalizableOp(ExternalOpContents def) {
        super(def.name(), def.operands());
        setLocation(extractLocation(def));
    }

    static Location extractLocation(ExternalOpContents def) {
        Object v = def.attributes().get(ATTRIBUTE_LOCATION);
        return switch (v) {
            case String s -> Location.fromString(s);
            case Location loc -> loc;
            case null -> null;
            default -> throw new UnsupportedOperationException("Unsupported location value:" + v);
        };
    }

    /**
     * Returns the operation's specific state as a map of attributes,
     * such that the specific state can be externalized.
     *
     * <p>A null attribute value is represented by the constant
     * value {@link #NULL_ATTRIBUTE_VALUE}.
     *
     * @return the operation's attributes, as an unmodifiable map
     */
    public Map<String, Object> attributes() {
        Location l = location();
        return l == null ? Map.of() : Map.of(ATTRIBUTE_LOCATION, l);
    }
}
