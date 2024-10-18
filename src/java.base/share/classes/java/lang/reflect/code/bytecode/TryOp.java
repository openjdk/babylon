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

package java.lang.reflect.code.bytecode;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.*;
import java.util.*;

import static java.lang.reflect.code.op.CoreOp.*;

/**
 * The top-level operation class for the enclosed set of try block operations.
 */
sealed abstract class TryOp extends ExternalizableOp {



    /**
     * Creates an exception region enter operation
     *
     * @param start    the exception region block
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static TryStartOp tryStart(Block.Reference start, Block.Reference... catchers) {
        return tryStart(start, List.of(catchers));
    }

    /**
     * Creates an exception region enter operation
     *
     * @param start    the exception region block
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static TryStartOp tryStart(Block.Reference start, List<Block.Reference> catchers) {
        List<Block.Reference> s = new ArrayList<>();
        s.add(start);
        s.addAll(catchers);
        return new TryStartOp(s);
    }

    /**
     * Creates an exception region exit operation
     *
     * @param next    block following the exception region
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static TryEndOp tryEnd(Block.Reference next, Block.Reference... catchers) {
        return tryEnd(next, List.of(catchers));
    }

    /**
     * Creates an exception region enter operation
     *
     * @param next    block following the exception region
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static TryEndOp tryEnd(Block.Reference next, List<Block.Reference> catchers) {
        List<Block.Reference> s = new ArrayList<>();
        s.add(next);
        s.addAll(catchers);
        return new TryEndOp(s);
    }

    // First successor is the non-exceptional successor whose target indicates
    // the first block in the exception region or block following the exception region exit.
    // One or more subsequent successors target the exception catching blocks
    // each of which have one block argument whose type is an exception type.
    final List<Block.Reference> s;

    protected TryOp(TryOp that, CopyContext cc) {
        super(that, cc);

        this.s = that.s.stream().map(cc::getSuccessorOrCreate).toList();
    }

    protected TryOp(String name, List<Block.Reference> s) {
        super(name, List.of());

        if (s.size() < 2) {
            throw new IllegalArgumentException("Operation must have two or more successors" + opName());
        }

        this.s = List.copyOf(s);
    }

    protected TryOp(ExternalizableOp.ExternalizedOp def) {
        super(def);

        if (def.successors().size() < 2) {
            throw new IllegalArgumentException("Operation must have two or more successors" + def.name());
        }

        this.s = List.copyOf(def.successors());
    }

    @Override
    public List<Block.Reference> successors() {
        return s;
    }

    public List<Block.Reference> catchBlocks() {
        return s.subList(1, s.size());
    }

    @Override
    public TypeElement resultType() {
        return JavaType.VOID;
    }

    /**
     * The exception region start operation.
     */
    @OpFactory.OpDeclaration(TryStartOp.NAME)
    public static final class TryStartOp extends TryOp
            implements Op.BlockTerminating {
        public static final String NAME = "try.start";

        public TryStartOp(ExternalizedOp def) {
            super(def);
        }

        TryStartOp(TryStartOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public TryStartOp transform(CopyContext cc, OpTransformer ot) {
            return new TryStartOp(this, cc);
        }

        TryStartOp(List<Block.Reference> s) {
            super(NAME, s);
        }

        public Block.Reference start() {
            return s.get(0);
        }
    }


    /**
     * The exception region exit operation.
     */
    @OpFactory.OpDeclaration(TryEndOp.NAME)
    public static final class TryEndOp extends TryOp
            implements Op.BlockTerminating {
        public static final String NAME = "try.end";

        public TryEndOp(ExternalizedOp def) {
            super(def);
        }

        TryEndOp(TryEndOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public TryEndOp transform(CopyContext cc, OpTransformer ot) {
            return new TryEndOp(this, cc);
        }

        TryEndOp(List<Block.Reference> s) {
            super(NAME, s);
        }

        public Block.Reference next() {
            return s.get(0);
        }
    }
}
