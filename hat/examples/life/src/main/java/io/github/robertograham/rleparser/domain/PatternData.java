package io.github.robertograham.rleparser.domain;

import java.util.Objects;

public class PatternData {

    private final MetaData metaData;
    private final LiveCells liveCells;

    public PatternData(MetaData metaData, LiveCells liveCells) {
        this.metaData = metaData;
        this.liveCells = liveCells;
    }

    public MetaData getMetaData() {
        return metaData;
    }

    public LiveCells getLiveCells() {
        return liveCells;
    }

    @Override
    public String toString() {
        return "PatternData{" +
                "metaData=" + metaData +
                ", liveCells=" + liveCells +
                '}';
    }

    @Override
    public boolean equals(Object object) {
        if (this == object)
            return true;

        if (!(object instanceof PatternData))
            return false;

        PatternData patternData = (PatternData) object;

        return Objects.equals(metaData, patternData.metaData) &&
                Objects.equals(liveCells, patternData.liveCells);
    }

    @Override
    public int hashCode() {
        return Objects.hash(metaData, liveCells);
    }
}
