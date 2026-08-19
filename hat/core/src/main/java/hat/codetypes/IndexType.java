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
