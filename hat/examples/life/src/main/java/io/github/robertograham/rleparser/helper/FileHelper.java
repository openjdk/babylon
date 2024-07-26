package io.github.robertograham.rleparser.helper;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FileHelper {

    public static List<String> getFileAsStringList(URI uri) {
        try {
            return Files.readAllLines(Paths.get(uri));
        } catch (IOException e) {
            return new ArrayList<>();
        }
    }
}
