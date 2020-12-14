NaturalLanguageProcessor Instructions:

>This program reads training data from .tsv files, like the ones provided for assignment 3.

>Firstly, the user is prompted to select which model to use. (Either NB_BOW or NB_FBOW(filtered))

>Given the model, it will be fitted to the training data. This training data is retrieved from a file,
which is specified by the author before running the code.

>The model is then used to classify the test data. This test data is also retrieved from a file, which
is specified by the author before running the code.

>The classifications are written to a trace file. The overall performance metrics of the model across
the collection of test data, are written to a separate evaluation file.