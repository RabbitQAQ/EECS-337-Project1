INSTRUCTION

1. Dependencies:
    nltk
    spacy
    sklearn

2. Some packages needs to be download(in case you miss some of them):
    python -m spacy download en
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

3. Put gg2013.json/gg2015.json/gg2018.json/gg2019.json into /data folder. (They are huge so we don't upload it)

4. run /autograder/gg_api.py and /autograder/autograder.py