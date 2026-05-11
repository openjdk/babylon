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
package optkl;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;

public class VarTable {

    public enum HATOpAttribute {
        UNKNOWN,
        PRIVATE,
        SHARED,
        INIT,
        NARROW,
        VECTOR,
    }

    /**
     * FunctionName within a KernelCallGraph -> { FunctioName -> { Table: Op -> <Attributes> } }
     *
     */
    private final ConcurrentHashMap<String, ConcurrentHashMap<Op, HATOpAttribute>> table;

    public HATOpAttribute getAttributeOrThrow(String functionName, CoreOp.VarOp varOp) {
        if (table.containsKey(functionName)) {
            return table.get(functionName).get(varOp);
        } else {
            throw new IllegalStateException("Function: " + functionName + " not registered");
        }
    }

    public boolean doesVarOpExist(String functionName, CoreOp.VarOp varOp) {
        if (table.containsKey(functionName)) {
            return table.get(functionName).containsKey(varOp);
        } else {
            return false;
        }
    }

    public VarTable() {
        this.table = new ConcurrentHashMap<>();
    }

    public VarTable(String function) {
        this.table = new ConcurrentHashMap<>();
        this.addFunction(function);
    }

    public void addFunction(String funcName) {
        if (!table.containsKey(funcName)) {
            table.put(funcName, new ConcurrentHashMap<>());
        }
    }

    public void addIfNeededOrThrow(String functionName, Op op, HATOpAttribute attribute) {
        if (table.containsKey(functionName)) {
            table.get(functionName).put(op, attribute);
        } else {
            throw new IllegalStateException("Function Name: " + functionName + " not present");
        }
    }

    public void passthrough(String functionName, Op oldOp, Op newOp) {
        if (table.containsKey(functionName)) {
            ConcurrentHashMap<Op, HATOpAttribute> opDeviceRegionHashMap = table.get(functionName);
            if (opDeviceRegionHashMap.containsKey(oldOp)) {
                opDeviceRegionHashMap.put(newOp, opDeviceRegionHashMap.get(oldOp));
            }
        }
    }
}