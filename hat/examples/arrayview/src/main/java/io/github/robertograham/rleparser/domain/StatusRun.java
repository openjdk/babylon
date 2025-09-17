package io.github.robertograham.rleparser.domain;

import io.github.robertograham.rleparser.domain.enumeration.Status;

import java.util.Objects;

public class StatusRun {

    private final int length;
    private final Status status;
    private final Coordinate origin;

    public StatusRun(int length, Status status, Coordinate origin) {
        this.length = length;
        this.status = status;
        this.origin = origin;
    }

    public int getLength() {
        return length;
    }

    public Status getStatus() {
        return status;
    }

    public Coordinate getOrigin() {
        return origin;
    }

    @Override
    public String toString() {
        return "StatusRun{" +
                "length=" + length +
                ", status=" + status +
                ", origin=" + origin +
                '}';
    }

    @Override
    public boolean equals(Object object) {
        if (this == object)
            return true;

        if (!(object instanceof StatusRun))
            return false;

        StatusRun statusRun = (StatusRun) object;

        return length == statusRun.length &&
                status == statusRun.status &&
                Objects.equals(origin, statusRun.origin);
    }

    @Override
    public int hashCode() {
        return Objects.hash(length, status, origin);
    }
}
