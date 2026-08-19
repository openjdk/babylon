package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.Objects;

public final class ConstantType implements TileType {

    final CodeType codeType;
    final Object value;

    public ConstantType(CodeType type, Object value) {
        this.codeType = type;
        this.value = value;
    }

    public CodeType codeType() {
        return codeType;
    }

    public Object value() {
        return value;
    }

    @Override
    public ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of("constant", List.of(codeType.externalize(), ExternalizedCodeType.of("c:" + value)));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        ConstantType that = (ConstantType) obj;
        return Objects.equals(codeType, that.codeType) && Objects.equals(value, that.value);
    }

    @Override
    public int hashCode() {
        return Objects.hash(codeType, value);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
