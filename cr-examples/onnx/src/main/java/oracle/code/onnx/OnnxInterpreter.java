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

package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.util.LinkedHashMap;
import oracle.code.onnx.ir.OnnxOp;

import java.util.List;
import java.util.Optional;

public class OnnxInterpreter {

    public static Object interpret(Class<? extends OnnxOp> opClass,
                                   List<Object> inputs,
                                   List<Object> attributes) {
        try {
            // @@@ assuming tensor inputs and outputs
            var schema = (OnnxOp.OnnxSchema)opClass.getDeclaredField("SCHEMA").get(null);
            var attrSchema = schema.attributes();
            var attributeMap = new LinkedHashMap<String, Object>(attributes.size());
            for (int i = 0; i < attributes.size(); i++) {
                var a = attributes.get(i);
                if (a instanceof Optional o) {
                    if (o.isPresent()) {
                        attributeMap.put(attrSchema.get(i).name(), o.get());
                    }
                } else {
                    attributeMap.put(attrSchema.get(i).name(), a);
                }
            }
            var outTensors = OnnxRuntime.getInstance().runOp(Arena.ofAuto(), schema.name(),
                    inputs.stream().takeWhile(i -> !(i instanceof Optional o && o.isEmpty())) // @@@ assuming gaps in the optional inputs are not allowed
                            .map(i -> i instanceof Optional o ? o.get() : i)
                            .mapMulti((i, ic) -> {
                                if (i instanceof List li) {
                                    li.forEach(ic);
                                } else {
                                    ic.accept(i);
                                }
                            })
                            .map(Tensor.class::cast)
                            .toList(),
                    schema.outputs().size(),
                    attributeMap);
            var outputs = schema.outputs();
            if (outputs.size() == 1) {
                if (outputs.getLast().quantifier() == OnnxOp.OnnxParameter.Quantifier.VARIADIC) {
                    return outTensors; // single variadic
                } else {
                    return outTensors.getFirst(); // single tensor
                }
            } else {
                // @@@ assuming only tail can be variadic
                if (outputs.getLast().quantifier() == OnnxOp.OnnxParameter.Quantifier.VARIADIC) {
                    var outArray = new Object[schema.outputs().size()];
                    for (int i = 0; i < outArray.length - 1; i++) {
                        outArray[i] = outputs.get(i);
                    }
                    outArray[outArray.length - 1] = outputs.subList(outArray.length - 1, outputs.size());
                    return outArray; // multiple tensors with variadic tail
                } else {
                    return outTensors.toArray(); // multiple tensors
                }
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
