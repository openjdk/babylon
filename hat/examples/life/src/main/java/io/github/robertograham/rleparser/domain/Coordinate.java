package io.github.robertograham.rleparser.domain;

import java.util.Objects;

public class Coordinate {

    private final int x;
    private final int y;

    public Coordinate(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public Coordinate withX(int x) {
        return this.x == x ? this : new Coordinate(x, y);
    }

    public Coordinate withY(int y) {
        return this.y == y ? this : new Coordinate(x, y);
    }

    public Coordinate plusToX(int amount) {
        return amount == 0 ? this : withX(x + amount);
    }

    public Coordinate plusToY(int amount) {
        return amount == 0 ? this : withY(y + amount);
    }

    @Override
    public String toString() {
        return "Coordinate{" +
                "x=" + x +
                ", y=" + y +
                '}';
    }

    @Override
    public boolean equals(Object object) {
        if (this == object)
            return true;

        if (!(object instanceof Coordinate))
            return false;

        Coordinate coordinate = (Coordinate) object;

        return x == coordinate.x &&
                y == coordinate.y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}
