package hat.codetypes;

import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Internal Type to the TypeInterpreter for providing/passing through the
 * tile shapes. This facilites shapes checking at runtime, before specializing
 * the code model.
 */
public final class ShapeType implements TileType {

    private final List<Integer> shapes;

    public ShapeType(ConstantType ...shapes) {
        this.shapes = list(Arrays.stream(shapes).toList());
    }

    public int dimensions() {
        return shapes.size();
    }

    public List<Integer> list(List<ConstantType> shapes) {
        return shapes.stream()
                .map(shape -> (Integer) shape.value)
                .collect(Collectors.toList());
    }

    public List<Integer> list() {
        return shapes;
    }

    @Override
    public ExternalizedCodeType externalize() {
        List<ExternalizedCodeType> externalizedTypes = shapes.stream().map(s -> new ExternalizedCodeType("s" + s, List.of())).collect(Collectors.toList());
        return ExternalizedCodeType.of("shape", externalizedTypes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.shapes);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) { return true; }
        if (obj == null || obj.getClass() != ShapeType.class) { return false; }
        ShapeType other = (ShapeType) obj;
        return Objects.equals(this.shapes, other.shapes);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
