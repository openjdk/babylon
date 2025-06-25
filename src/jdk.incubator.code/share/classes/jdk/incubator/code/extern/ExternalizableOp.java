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

package jdk.incubator.code.extern;

import jdk.incubator.code.*;

import java.util.List;
import java.util.Map;

/**
 * An operation that supports externalization of its content and reconstruction
 * via an instance of {@link ExternalizedOp}.
 * <p>
 * The specific content of an externalizable operation can be externalized to a
 * map of {@link #attributes attributes}, and is reconstructed from the
 * attributes component of an instance of {@link ExternalizedOp}.
 * <p>
 * An externalizable operation could be externalized via serialization to
 * a textual representation. That textual representation could then be deserialized,
 * via parsing, into an instance of {@link ExternalizedOp} from which a new
 * externalizable operation can be reconstructed that is identical to one that
 * was serialized.
 */
public abstract class ExternalizableOp extends Op {

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
     * Constructs an operation from its external content.
     *
     * @param def the operation's external content.
     * @implSpec This implementation invokes the {@link Op#Op(String, List) constructor}
     * accepting the non-optional components of the operation's content, {@code name},
     * and {@code operands}:
     * <pre> {@code
     *  this(def.name(), def.operands());
     * }</pre>
     */
    @SuppressWarnings("this-escape")
    protected ExternalizableOp(ExternalizedOp def) {
        super(def.name(), def.operands());
    }


    /**
     * Externalizes the operation's specific content as a map of attributes.
     *
     * <p>A null attribute value is represented by the constant
     * value {@link Op#NULL_ATTRIBUTE_VALUE}.
     *
     * @return the operation's attributes, as an unmodifiable map
     */
    public Map<String, Object> attributes() {
        return Map.of();
    }
}
