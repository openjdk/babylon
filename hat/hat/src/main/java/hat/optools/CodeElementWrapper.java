package hat.optools;

import java.lang.reflect.code.CodeElement;

public abstract class CodeElementWrapper<T extends CodeElement> {
    protected T codeElement;

    CodeElementWrapper(T codeElement) {
        this.codeElement = codeElement;
    }

}
