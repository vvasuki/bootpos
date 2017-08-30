### Runtime parameters:

-   The object Main is the point of entry for the program. It reads various parameters for the tagging experiment, such as the language, corpus, tagging algorithm to be used etc.. from a properties file whose path is passed as an argument to the program.
-   An <span style="font-weight: bold;">example properties file</span> called partofspeechtagger/runtimeSettings.properties is included. This is the file used by default, and you may use this as a template.

### Directory structure for data files:

-   Please set DATA\_DIR = “/home/vvasuki/posTagging/data/” as appropriate in your runtimeSettings.properties file.
-   This is the directory wherein the training, test and wiktionary files are stored. To prepare it appropriately, do the following. Further details about the expected structure of this directory is found in the runtimeSettings.properties file.
    1. Please get all data files from <http://ilk.uvt.nl/conll/free_data.html> and extract them under DATA\_DIR.
    1a. Commands for doing this are given in <http://sourceforge.net/apps/mediawiki/opennlp/index.php?title=Conll06#Extract_data>.
    1b. Prepare or get raw text files for each language, give them a name containing “raw” and place them in the language/corpus/train directory. Ensure i\] One token per line, ii\] new line as separator.
    3. Download and extract the wiktionary file it: [http://toolserver.org/~enwikt/definitions/enwikt-defs-latest-all.tsv.gz]
    4. Download and extract the universal tag sets: <http://code.google.com/p/universal-pos-tags/>.

### Compilation, running, etc..:

Use the sbt script located under “bin”.

  [http://toolserver.org/~enwikt/definitions/enwikt-defs-latest-all.tsv.gz]: http://toolserver.org/%7Eenwikt/definitions/enwikt-defs-latest-all.tsv.gz
