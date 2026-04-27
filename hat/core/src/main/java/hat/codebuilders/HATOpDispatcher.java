/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.codebuilders;

import hat.dialect.HATBarrierOp;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATOp;
import hat.dialect.HATThreadOp;
import jdk.incubator.code.Op;
import optkl.codebuilders.BabylonOpDispatcher;
import optkl.codebuilders.ScopeAwareJavaOrC99StyleCodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

import static hat.dialect.HATF16Op.HATF16BinaryOp;
import static hat.dialect.HATF16Op.HATF16ConvOp;
import static hat.dialect.HATF16Op.HATF16ToFloatConvOp;
import static hat.dialect.HATF16Op.HATF16VarLoadOp;
import static hat.dialect.HATMemoryDefOp.HATMemoryLoadOp;
import static hat.dialect.HATPtrOp.HATPtrLengthOp;
import static hat.dialect.HATPtrOp.HATPtrLoadOp;
import static hat.dialect.HATPtrOp.HATPtrStoreOp;
import static hat.dialect.HATTensorOp.TensorCreateOp;
import static hat.dialect.HATTensorOp.TensorFillOp;
import static hat.dialect.HATTensorOp.TensorLoadOp;
import static hat.dialect.HATTensorOp.TensorMMAOp;
import static hat.dialect.HATTensorOp.TensorStoreLoadOp;
import static hat.dialect.HATTensorOp.TensorStoreOp;
import static hat.dialect.HATTensorOp.TensorVarLoadOp;
import static hat.dialect.HATVectorOp.HATVectorBinaryOp;
import static hat.dialect.HATVectorOp.HATVectorLoadOp;
import static hat.dialect.HATVectorOp.HATVectorMakeOfOp;
import static hat.dialect.HATVectorOp.HATVectorOfOp;
import static hat.dialect.HATVectorOp.HATVectorSelectLoadOp;
import static hat.dialect.HATVectorOp.HATVectorSelectStoreOp;
import static hat.dialect.HATVectorOp.HATVectorStoreView;
import static hat.dialect.HATVectorOp.HATVectorVarLoadOp;

/* this should not be too C99 specific but can reference HAT ops.  */
public interface HATOpDispatcher<T extends ScopeAwareJavaOrC99StyleCodeBuilder<T>> extends BabylonOpDispatcher<T, ScopedCodeBuilderContext> {

    T hatBarrierOp( HATBarrierOp barrierOp);

    //T hatLocalVarOp( HATMemoryVarOp.HATLocalVarOp barrierOp);

    T hatPrivateVarOp( HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp);

    T hatThreadIdOp( HATThreadOp hatThreadOp);

    //T hatVectorVarOp( HATVectorVarOp hatVectorVarOp);

    T hatVectorStoreOp( HATVectorStoreView hatFloat4StoreOp);

    T hatBinaryVectorOp( HATVectorBinaryOp hatVectorBinaryOp);

    T hatVectorLoadOp( HATVectorLoadOp hatVectorLoadOp);

    T hatSelectLoadOp( HATVectorSelectLoadOp hatVSelectLoadOp);

    T hatSelectStoreOp( HATVectorSelectStoreOp hatVSelectStoreOp);

    T hatVectorVarLoadOp( HATVectorVarLoadOp hatVectorVarLoadOp);

    //T hatF16VarOp( HATF16Op.HATF16VarOp hatF16VarOp);

    T hatF16BinaryOp( HATF16BinaryOp hatF16BinaryOp);

    T hatF16VarLoadOp( HATF16VarLoadOp hatF16VarLoadOp);

    T hatF16ConvOp( HATF16ConvOp hatF16ConvOp);

    T hatVectorOfOps( HATVectorOfOp hatVectorOp);

    T hatVectorMakeOf( HATVectorMakeOfOp hatVectorMakeOfOp);

    T hatF16ToFloatConvOp( HATF16ToFloatConvOp hatF16ToFloatConvOp);

    T hatPrivateVarInitOp( HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp);

    T hatMemoryLoadOp( HATMemoryLoadOp hatMemoryLoadOp);

    T hatPtrLoadOp(HATPtrLoadOp hatPtrLoadOp);

    T hatPtrStoreOp( HATPtrStoreOp hatPtrStoreOp);

    T hatPtrLengthOp( HATPtrLengthOp hatPtrLengthOp);

    T hatVarOp(HATMemoryVarOp.HATVarOp HATVarOp);

    T hatTensorCreateOp(TensorCreateOp tensorCreateOp);

    T hatTensorFillOp(TensorFillOp tensorFillOp);

    T hatTensorVarLoadOp(TensorVarLoadOp hatTensorVarLoadOp);

    T hatTensorMMAOp(TensorMMAOp tensorMMAOp);

    T hatTensorStoreLoadOp(TensorStoreLoadOp tensorStoreLoadOp);

    T hatTensorLoadOp(TensorLoadOp tensorLoadOp);

    T hatTensorStoreOp(TensorStoreOp tensorStoreOp);


    @Override
    default T recurse(Op op) {
        if (op instanceof HATOp hatOp) {
            switch (hatOp) {
                case HATBarrierOp $ -> hatBarrierOp($);
                //case HATMemoryVarOp.HATLocalVarOp $ -> hatLocalVarOp($);
                case HATMemoryVarOp.HATPrivateVarOp $ -> hatPrivateVarOp($);
                case HATMemoryVarOp.HATPrivateInitVarOp $ -> hatPrivateVarInitOp($);
                case HATThreadOp $ -> hatThreadIdOp($);
                //case HATVectorVarOp $ -> hatVectorVarOp($);
                case HATVectorStoreView $ -> hatVectorStoreOp($);
                case HATVectorBinaryOp $ -> hatBinaryVectorOp($);
                case HATVectorLoadOp $ -> hatVectorLoadOp($);
                case HATVectorSelectLoadOp $ -> hatSelectLoadOp($);
                case HATVectorSelectStoreOp $ -> hatSelectStoreOp($);
                case HATVectorVarLoadOp $ -> hatVectorVarLoadOp($);
                case HATVectorOfOp $ -> hatVectorOfOps($);
                //case HATF16Op.HATF16VarOp $ -> hatF16VarOp($);
                case HATF16BinaryOp $ -> hatF16BinaryOp($);
                case HATF16VarLoadOp $ -> hatF16VarLoadOp($);
                case HATF16ConvOp $ -> hatF16ConvOp($);
                case HATVectorMakeOfOp $ -> hatVectorMakeOf($);
                case HATPtrLoadOp $ -> hatPtrLoadOp($);
                case HATPtrStoreOp $ -> hatPtrStoreOp($);
                case HATPtrLengthOp $ -> hatPtrLengthOp($);
                case HATF16ToFloatConvOp $ -> hatF16ToFloatConvOp($);
                case HATMemoryLoadOp $ -> hatMemoryLoadOp($);
                case HATMemoryVarOp.HATVarOp $ -> hatVarOp($);
                case TensorCreateOp $ -> hatTensorCreateOp($);
                case TensorVarLoadOp $ -> hatTensorVarLoadOp($);
                case TensorFillOp $ -> hatTensorFillOp($);
                case TensorMMAOp $ -> hatTensorMMAOp($);
                case TensorStoreLoadOp $ -> hatTensorStoreLoadOp($);
                case TensorLoadOp $ -> hatTensorLoadOp($);
                case TensorStoreOp $ -> hatTensorStoreOp($);
                default -> throw new IllegalStateException("handle nesting of hat op " + op);
            }
        } else {
            BabylonOpDispatcher.super.recurse(op);
        }

        return (T) this;
    }
}
