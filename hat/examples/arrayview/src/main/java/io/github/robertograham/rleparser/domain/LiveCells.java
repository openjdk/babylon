package io.github.robertograham.rleparser.domain;

import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class LiveCells {

    private final Set<Coordinate> coordinates;

    public LiveCells(Set<Coordinate> coordinates) {
        this.coordinates = coordinates;
    }

    public Set<Coordinate> getCoordinates() {
        return coordinates;
    }

    public LiveCells filteredByX(int x) {
        return new LiveCells(getCoordinatesWithPropertyEqualToValue(x, Coordinate::getX));
    }

    public LiveCells filteredByY(int y) {
        return new LiveCells(getCoordinatesWithPropertyEqualToValue(y, Coordinate::getY));
    }

    @Override
    public String toString() {
        return "LiveCells{" +
                "coordinates=" + coordinates +
                '}';
    }

    @Override
    public boolean equals(Object object) {
        if (this == object)
            return true;

        if (!(object instanceof LiveCells))
            return false;

        LiveCells liveCells = (LiveCells) object;

        return Objects.equals(coordinates, liveCells.coordinates);
    }

    @Override
    public int hashCode() {
        return Objects.hash(coordinates);
    }

    private Set<Coordinate> getCoordinatesWithPropertyEqualToValue(int value, Function<Coordinate, Integer> coordinatePropertyAccessor) {
        return coordinates.stream()
                .filter(coordinate -> value == coordinatePropertyAccessor.apply(coordinate))
                .collect(Collectors.toSet());
    }
}
