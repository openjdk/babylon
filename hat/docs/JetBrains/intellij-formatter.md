# HAT Code Formatting in IntelliJ
[Back to Index ../](../index.md)

## How to configure it?

In IntelliJ:

1. Import XML for auto-formatter rules: Settings -> Editor -> Code Style ->
   Java -> Scheme (gear) -> Import Scheme -> IntelliJ IDEA code style XML
2. Select file from `hat/scripts/IDEA-xml-formatter/HAT-Formatter.xml`.
3. Enable Actions on Save: Settings -> Tools -> Actions on Save -> Enable
   `Reformat Code`
4. Enable Optimize Imports: Settings -> Tools -> Actions on Save -> Enable
   `Optimize Imports`.

## From the command line.

You can execute Intellij's formatter from bash
```bash
 ~/Applications/IntelliJ\ IDEA\ Ultimate.app/Contents/bin/format.sh -sscripts/IDEA-xml-formatter/HAT_Formatter.xml YourJavaSource.java
```