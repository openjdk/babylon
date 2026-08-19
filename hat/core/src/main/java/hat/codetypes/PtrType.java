package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.Objects;

/**
 * Type to represent input/output tensors (mutable data) that will be copied
 * between host <-> device to/from the accelerator's global memory.
 */
public final class PtrType implements TileType {

    final CodeType rType;

    public PtrType(CodeType rType) {
        this.rType = rType;
    }

    public CodeType rType() {
        return rType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rType);
    }

    @Override
    public ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of("tilePtrType", List.of(rType.externalize()));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {return true;}
        if (obj == null || getClass() != obj.getClass()) {return false;}
        final PtrType other = (PtrType) obj;
        return Objects.equals(this.rType, other.rType);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
