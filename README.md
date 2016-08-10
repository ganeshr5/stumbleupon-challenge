# stumbleupon-challenge
https://www.kaggle.com/c/stumbleupon


StumbleUpon Evergreen Classification Challenge

To build a classifier to categorize webpages as evergreen or non-evergreen


Running the code

main.py is the main python code that runs the evaluation. Use following arguments to run the code:

    Raw text file
    Train data file
    Test data file
    Classifier to use:
        0: Logistic Regression
        1: Naive Bayes
        2: Random Forest
    Feature Selection
        0: Use TFIDF on boiler plate text
        1: Use non-boiler plate text attributes
        2: Use non-boiler plate text attributes AND extract one top LDA topic per boiler plate attribute
        3: Use LDA topic vectos for boiler plate code
    Debug (optional): to print debug statments
    Usage: python main.py \<rawData\> \<trainFile\> \<testFile\> \<classifier - 0-lr, 1-nb, 2-rf, 3-ab\> \<feature selection 0-useOtherFeatures, 1-only Boilerplate, 2-userOtherFeatures + LDA, 3-LDA\>
