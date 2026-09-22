package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Type to be used within the TypeInterpreter to pass build/pass
 * tensors with a specific shape.
 */
public final class TensorType implements TileType {

    private final CodeType dType;   // dType
    private final List<Integer> shape;  // Shape

    public TensorType(CodeType codeType, List<Integer> shape) {
        this.dType = codeType;
        this.shape = shape;
    }

    public CodeType elementType() {
        return dType;
    }

    public List<Integer> shape() {
        return shape;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) { return true; }
        if (o == null || getClass() != o.getClass()) { return false; }
        TensorType tensorType = (TensorType) o;
        return Objects.equals(dType, tensorType.dType) && Objects.equals(shape, tensorType.shape);
    }

    @Override
    public ExternalizedCodeType externalize() {
        List<ExternalizedCodeType> externalizedTypes = shape.stream().map(s -> new ExternalizedCodeType("x" + s, List.of())).collect(Collectors.toList());
        externalizedTypes.add(dType.externalize());
        return ExternalizedCodeType.of("tensor", externalizedTypes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(dType, shape);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
