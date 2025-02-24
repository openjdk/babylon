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

package oracle.code.json;

import java.io.Serial;

/**
 * Signals that an error has been detected while parsing the
 * JSON document.
 *
 */
public class JsonParseException extends RuntimeException {

    @Serial
    private static final long serialVersionUID = 7022545379651073390L;

    /**
     * Position of the error row in the document
     */
    private final int row;

    /**
     * Position of the error column in the document
     */
    private final int col;

    /**
     * Constructs a JsonParseException with the specified detail message.
     * @param message the detail message
     * @param row the row of the error on parsing the document
     * @param col the column of the error on parsing the document
     */
    public JsonParseException(String message, int row, int col) {
        super(message);
        this.row = row;
        this.col = col;
    }

    /**
     * {@return the row of the error on parsing the document}
     */
    public int getErrorRow() {
        return row;
    }

    /**
     * {@return the column of the error on parsing the document}
     */
    public int getErrorColumn() {
        return col;
    }
}
