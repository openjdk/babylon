package io.github.robertograham.rleparser.helper;

import io.github.robertograham.rleparser.domain.Coordinate;
import io.github.robertograham.rleparser.domain.StatusRun;
import io.github.robertograham.rleparser.domain.enumeration.Status;

import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class StatusRunHelper {

    private static final Pattern STATUS_RUN_PATTERN = Pattern.compile("(\\d*)([a-z$])");

    public static StatusRun readStatusRun(String encodedStatusRun, Coordinate origin) {
        Matcher matcher = STATUS_RUN_PATTERN.matcher(encodedStatusRun);

        if (matcher.find())
            return new StatusRun(
                    matcher.group(1).isEmpty() ?
                            1
                            : Integer.parseInt(matcher.group(1)),
                    Status.fromCode(matcher.group(2)),
                    origin
            );

        throw new IllegalArgumentException("Encoded run length status did not match (\\d*)([a-z$])");
    }

    public static Set<Coordinate> readCoordinates(StatusRun statusRun) {
        return IntStream.range(statusRun.getOrigin().getX(), statusRun.getOrigin().getX() + statusRun.getLength())
                .mapToObj(statusRun.getOrigin()::withX)
                .collect(Collectors.toSet());
    }
}
