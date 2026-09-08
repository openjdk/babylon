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
package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class IndexType implements TileType {

    private final List<CodeType> indexes;

    public IndexType(CodeType ...indexes) {
        this.indexes = Arrays.stream(indexes).toList();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {return false;}
        IndexType that = (IndexType) obj;
        if (indexes.size() != that.indexes.size()) {
            return false;
        }
        boolean eq = true;
        for (int i = 0; i < indexes.size(); i++) {
            if (!Objects.equals(indexes.get(i), that.indexes.get(i))) {
                eq = false;
            }
        }
        return eq;
    }

    @Override
    public ExternalizedCodeType externalize() {
        List<ExternalizedCodeType> externalizedTypes = new ArrayList<>();
        for (CodeType index : indexes) {
            externalizedTypes.add(new ExternalizedCodeType("i" + index, List.of()));
        }
        return ExternalizedCodeType.of("shape", externalizedTypes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(indexes);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }

}
