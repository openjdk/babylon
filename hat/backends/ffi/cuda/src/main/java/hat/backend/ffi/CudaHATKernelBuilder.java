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
package hat.backend.ffi;

import hat.codebuilders.C99HATKernelBuilder;
import hat.codebuilders.ScopedCodeBuilderContext;
import hat.dialect.HatVSelectLoadOp;
import hat.dialect.HatVSelectStoreOp;
import hat.dialect.HatVectorBinaryOp;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorStoreView;
import hat.dialect.HatVectorVarLoadOp;
import hat.dialect.HatVectorVarOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;

public class CudaHATKernelBuilder extends C99HATKernelBuilder<CudaHATKernelBuilder> {

    private CudaHATKernelBuilder threadDimId(int id) {
        return keyword(switch(id){
            case 0->"x";
            case 1->"y";
            case 2->"z";
            default -> throw new RuntimeException("Thread Dimension not supported");
        });
    }
    @Override
    public CudaHATKernelBuilder defines() {
        return self()
                .hashDefine("HAT_CUDA")
                 // .hashIfdef("HAT_CUDA", _ ->
                   //     indent(_ -> self()
                .hashDefine("HAT_GLOBAL_MEM", _ -> {})
                .hashDefine("HAT_LOCAL_MEM", _ -> keyword("__shared__"))
                .hashDefine("HAT_FUNC", _->externC().space().keyword("__device__").space().keyword("inline"))
                .hashDefine("HAT_KERNEL", _->externC().space().keyword("__global__"))
                .hashDefine("HAT_GIX", _ -> paren(_->blockId(0).asterisk().localSize(0).plus().localId(0)))
                .hashDefine("HAT_GIY", _ -> paren(_->blockId(1).asterisk().localSize(1).plus().localId(1)))
                .hashDefine("HAT_GIZ", _ -> paren(_->blockId(2).asterisk().localSize(2).plus().localId(2)))
                .hashDefine("HAT_LIX", _ -> keyword("threadIdx").dot().threadDimId(0))
                .hashDefine("HAT_LIY", _ -> keyword("threadIdx").dot().threadDimId(1))
                .hashDefine("HAT_LIZ", _ -> keyword("threadIdx").dot().threadDimId(2))
                .hashDefine("HAT_GSX", _ -> keyword("gridDim").dot().threadDimId(0).asterisk().localSize(0))
                .hashDefine("HAT_GSY", _ -> keyword("gridDim").dot().threadDimId(1).asterisk().localSize(1))
                .hashDefine("HAT_GSZ", _ -> keyword("gridDim").dot().threadDimId(2).asterisk().localSize(2))
                .hashDefine("HAT_LSX", _ -> keyword("blockDim").dot().threadDimId(0))
                .hashDefine("HAT_LSY", _ -> keyword("blockDim").dot().threadDimId(1))
                .hashDefine("HAT_LSZ", _ -> keyword("blockDim").dot().threadDimId(2))
                .hashDefine("HAT_BIX", _ -> keyword("blockIdx").dot().threadDimId(0))
                .hashDefine("HAT_BIY", _ -> keyword("blockIdx").dot().threadDimId(1))
                .hashDefine("HAT_BIZ", _ -> keyword("blockIdx").dot().threadDimId(2))
                .hashDefine("HAT_BARRIER", _->keyword("__syncthreads").ocparen());
                    //    )
        //);
    }

    @Override
    public CudaHATKernelBuilder atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name){
        return identifier("atomicAdd").paren(_ -> ampersand().recurse(buildContext, instanceResult.op()).rarrow().identifier(name).comma().literal(1));
    }

    @Override
    public CudaHATKernelBuilder generateVectorStore(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        Value dest = hatVectorStoreView.operands().get(0);
        Value index = hatVectorStoreView.operands().get(2);

        keyword("reinterpret_cast")
                .lt()
                .typeName("float4")  // fixme
                .space()
                .asterisk()
                .gt()
                .oparen()
                .ampersand();

        if (dest instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen().osbrace().intConstZero().csbrace()
                .space().equals().space()
                .varName(hatVectorStoreView);

        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorBinary(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {

        Value op1 = hatVectorBinaryOp.operands().get(0);
        Value op2 = hatVectorBinaryOp.operands().get(1);

        if (op1 instanceof Op.Result r && r.op() instanceof HatVectorBinaryOp hatVectorBinaryOp1) {
            typeName("float" + hatVectorBinaryOp.vectorN()).space()
                            .identifier(hatVectorBinaryOp.varName() + "_1")
                                    .semicolon().nl();
            hatVectorBinaryOp1.varName(hatVectorBinaryOp.varName() + "_1");
            recurse(buildContext, hatVectorBinaryOp1);
        }

        if (op2 instanceof Op.Result r && r.op() instanceof HatVectorBinaryOp hatVectorBinaryOp2) {
            typeName("float" + hatVectorBinaryOp.vectorN()).space()
                    .identifier(hatVectorBinaryOp.varName() + "_2")
                    .semicolon().nl();
            hatVectorBinaryOp2.varName(hatVectorBinaryOp.varName() + "_2");
            recurse(buildContext, hatVectorBinaryOp2);
        }

        for (int i = 0; i < hatVectorBinaryOp.vectorN(); i++) {

           identifier(hatVectorBinaryOp.varName())
                   .dot()
                   .identifier(hatVectorBinaryOp.mapLane(i))
                   .space().equals().space();

            if (op1 instanceof Op.Result r) {
                if (!(r.op() instanceof HatVectorBinaryOp hatVectorBinaryOp1)) {
                    recurse(buildContext, r.op());
                } else {
                    identifier(hatVectorBinaryOp1.varName());
                }
            }
            dot().identifier(hatVectorBinaryOp.mapLane(i)).space();
            identifier(hatVectorBinaryOp.operationType().symbol()).space();

            if (op2 instanceof Op.Result r) {
                if (!(r.op() instanceof HatVectorBinaryOp hatVectorBinaryOp2)) {
                    recurse(buildContext, r.op());
                } else {
                    identifier(hatVectorBinaryOp2.varName());
                }
            }
            dot().identifier(hatVectorBinaryOp.mapLane(i)).semicolon().nl();
        }

        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorLoad(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        Value source = hatVectorLoadOp.operands().get(0);
        Value index = hatVectorLoadOp.operands().get(1);

        keyword("reinterpret_cast")
                .lt()
                .typeName(hatVectorLoadOp.buildType())
                .space()
                .asterisk()
                .gt()
                .oparen()
                .ampersand();

        if (source instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        rarrow().identifier("array").osbrace();

        if (index instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        csbrace().cparen().osbrace().intConstZero().csbrace();

        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        identifier(hatVSelectLoadOp.varName())
                .dot()
                .identifier(hatVSelectLoadOp.mapLane());
        return self();
    }

    @Override
    public CudaHATKernelBuilder generateVectorSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        identifier(hatVSelectStoreOp.varName())
                .dot()
                .identifier(hatVSelectStoreOp.mapLane())
                .space().equals().space();
        if (hatVSelectStoreOp.resultValue() != null) {
            // We have detected a direct resolved result (resolved name)
            varName(hatVSelectStoreOp.resultValue());
        } else {
            // otherwise, we traverse to resolve the expression
            Value storeValue = hatVSelectStoreOp.operands().get(1);
            if (storeValue instanceof Op.Result r) {
                recurse(buildContext, r.op());
            }
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorVarOp(ScopedCodeBuilderContext buildContext, HatVectorVarOp hatVectorVarOp) {
        Value operand = hatVectorVarOp.operands().getFirst();
        typeName(hatVectorVarOp.buildType())
                .space()
                .varName(hatVectorVarOp);

        if (operand instanceof Op.Result r && r.op() instanceof HatVectorBinaryOp) {
            semicolon().nl();
        } else {
            space().equals().space();
        }

        if (operand instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    @Override
    public CudaHATKernelBuilder hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HatVectorVarLoadOp hatVectorVarLoadOp) {
        varName(hatVectorVarLoadOp);
        return self();
    }

}
