package io.github.robertograham.rleparser.helper;

public class RleFileHelper {

    private static final String COMMENT_STARTER = "#";

    public static boolean isCommentString(String rleLine) {
        return rleLine.startsWith(COMMENT_STARTER);
    }
}
