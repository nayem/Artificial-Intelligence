Report for question 2 Topic Classification

** Process:

For the training part, first we use regular expression in read_train_data function to get words that are at least 2-letter long, because alphabet letters does not mean much in topics, then we store following information:
Doc_untrained: a list of documents that are 'untrained', or we can't see their true topics
Doc_content: to store all words in every document with their relative frequencies in that document
Topic_Word_dict: to store frequencies of words under every topic for those 'trained' documents, or documents we can actually see their true labels.

To make information more significant, we use tf-idf metric to define those 'significant' words inside every document(and in every topic as well, we'll use tf-idf for topics later in writing distinctive word file and in testing), so in Doc_content, only those significant words will be stored. The threshold for tf-idf here is set to 0.0005 for full information on true topics, and 0.05 for partial information on true topics. 0.05 is also set so that the program will not run forever, without suffering too much from performance loss.

Then we use LDA algorithm in Topic Classification to train the model. The algorithm has 2 stages: 
Inital assign: randomly assign topics to every word in all untrained documents
Re assign(Collapsed Gibbs Sampling): update tagging of words using this formula: argmax(topic) P(W|T)*P(T|D), P(W|T) comes from Topic_Word_dict, P(T|D) comes from Doc_Topic_dict, which is initiated randomly in init_assign state of LDA. After getting best guess of topic for a single word, we update Topic_Word_dict and Doc_Topic_dict so that calculating next word will use the latest information. This method is also called 'Collapsed Gibbs Sampling'.

We finish the training part by outputing distinctive words using 10 words for every topic with highest tf-idf values, and outputing model file by writing in Topic_Word_dict.

For the testing part, we read in model file, do a further filtering on this dictionary with threshold = 0.0015(we need to do it twice because first time it is filtering over those trained documents, and now it's filtering over trained documents + guessing over untrained using LDA). When predicting topics for documents, we predict the most probably topic for every word, then predict based on the most frequent topics we've seen in this document as the prediction result. Further detail can be seen in predict_topic function.

** Results:

FRACTION = 1 train: 20 secs test: 18 secs
Accuracy:0.682687201275
Confusion matrix:
[[ 198.    0.    0.    0.    2.    1.    1.    0.    6.    1.    0.   12.
     1.    3.    0.    2.   85.    5.    2.    0.]
 [   4.  341.    0.    1.    2.   11.    8.    0.    2.    3.    0.    1.
     1.    2.    7.    7.    1.    3.    2.    0.]
 [   0.    2.  326.    0.    0.    1.   11.    1.    0.   45.    2.    0.
     0.    1.    1.    1.    1.    1.    3.    1.]
 [  12.    1.    3.  133.    0.    3.    1.    1.    1.    3.    0.    7.
     5.    0.    0.    6.  221.    1.    0.    0.]
 [   1.    1.    0.    0.  333.    9.    7.    4.   19.    2.    5.    3.
     0.    0.    0.    6.    1.    1.    2.    2.]
 [   7.   11.    2.    0.   13.  221.   65.   25.    2.    5.   10.    6.
     0.    2.    5.    1.    2.    5.    8.    3.]
 [   1.   15.    0.    0.    0.    8.  331.    2.    2.    0.    9.    7.
     2.    1.    3.    2.    1.    1.    4.    1.]
 [   3.    3.    3.    0.    3.    8.   34.  276.    3.    2.    7.    0.
     1.    5.   11.    1.    0.    4.    8.   17.]
 [   1.    1.    2.    0.    2.    0.    4.    0.  341.    2.    0.    0.
     2.    0.    0.    5.    3.    0.    1.    0.]
 [   0.    0.    2.    0.    0.    0.    2.    0.    2.  390.    0.    0.
     0.    0.    1.    0.    0.    2.    0.    0.]
 [   2.    7.    2.    0.    0.   20.   85.   15.    1.    2.  224.    2.
     2.    0.    7.    0.    2.    3.   10.    1.]
 [  19.    3.    4.    3.    2.   18.    5.    1.    2.    4.    2.  296.
     3.    2.    2.   14.    5.    0.   11.    0.]
 [   0.    1.    1.    0.    1.    1.    1.    0.   14.    7.    2.    0.
   334.    2.    0.   11.    0.    0.    1.    0.]
 [   2.   36.    4.    0.    0.    3.    8.    0.    3.    3.    2.    0.
     0.  332.    0.    1.    0.    1.    3.    0.]
 [   1.    2.    1.    0.    2.   28.  180.   20.    0.    1.   15.    0.
     0.    2.  116.    0.    4.    5.   12.    3.]
 [   5.    1.    1.    0.    1.    0.    1.    0.  117.    8.    1.    6.
     1.    0.    0.  161.    5.    2.    0.    0.]
 [  23.    0.    3.    4.    1.    0.    0.    1.   26.    0.    0.    2.
     2.    0.    0.    6.  178.    4.    1.    0.]
 [   2.    4.    2.    0.    0.   31.    1.   14.    0.    2.    4.    5.
     0.    0.    1.    9.    1.  316.    2.    0.]
 [   2.    2.    7.    1.    5.    0.  147.   83.    0.    2.   31.    0.
     0.    1.   29.    2.    3.    8.   55.   16.]
 [   2.    0.    2.    0.    0.    7.   22.   85.    0.    1.    7.    2.
     0.    3.    3.    1.    1.    9.   10.  240.]]

FRACTION = .1 train: 262 secs test: 16 secs
Accuracy:0.191317047265
Confusion matrix:
[[  56.   10.   41.    2.    3.   17.   19.   42.   27.   12.    1.    0.
     1.   26.    5.   19.    0.   23.    4.   11.]
 [   7.   16.   28.    6.    7.   11.   49.   60.   30.   30.   24.    0.
     5.   40.   14.   21.    0.   18.   18.   12.]
 [   4.    3.  148.    1.    8.   25.   24.   25.   12.   51.   10.    0.
    11.   16.   13.   18.    0.    9.   16.    3.]
 [  21.   11.   17.    3.    4.   30.   25.   28.   13.   36.   24.    0.
     2.   51.    8.   50.    0.   29.   25.   21.]
 [   3.    2.   53.    0.   70.   27.   24.   52.   35.   13.   15.    0.
     3.   21.    5.   17.    0.   10.   20.   26.]
 [   3.   10.   35.    0.    8.   29.   48.   56.   21.   40.   23.    0.
     3.   22.   17.   18.    0.   19.   28.   13.]
 [   4.    6.   39.    1.    6.   10.  134.   18.   45.   37.   10.    0.
     4.   14.    6.   14.    0.   14.   16.   12.]
 [   3.    3.   36.    1.   10.    5.   31.  140.   19.   29.   12.    0.
     2.    7.    8.    8.    1.   17.   34.   23.]
 [   4.   14.   35.    1.    6.    7.   13.   31.   87.   22.    6.    0.
    13.   35.    7.   31.    0.   15.   23.   14.]
 [   3.    2.   53.    0.    1.    7.   18.   12.   17.  220.    2.    0.
     1.   23.    4.   17.    0.    7.   10.    2.]
 [   1.    8.   39.    0.    5.   14.   78.   37.   23.   40.   59.    0.
     1.   15.   11.   10.    0.    8.   20.   16.]
 [   5.   18.   33.    1.    8.   23.   20.   72.   27.   50.    8.   13.
    10.   17.   14.   28.    0.   29.   16.    4.]
 [   8.   19.   39.    2.    4.   10.    4.   52.   13.   30.   16.    0.
    75.   16.    6.   39.    0.    8.   20.   15.]
 [   4.   13.   23.    0.   10.   28.   13.   34.   21.   42.   17.    0.
     4.   81.    8.   61.    0.   20.   15.    4.]
 [   2.    5.   42.    0.    0.   18.   64.   77.   50.   20.   12.    0.
     5.    9.   31.    4.    0.   20.   23.   10.]
 [  15.    5.   16.    0.    2.    5.   12.   38.   36.   32.    3.    0.
     4.   20.    8.   76.    0.   13.   17.    8.]
 [  20.    6.   16.    2.    3.    4.   35.   36.   11.   12.   11.    0.
     4.   21.    7.   33.    0.   18.    6.    6.]
 [   4.    7.   31.    0.    2.   32.   50.   78.   27.   27.    5.    1.
     5.   29.    3.   18.    1.   56.   10.    8.]
 [   2.    5.   40.    0.    5.    9.   32.   57.   50.   19.    6.    0.
     2.   14.   27.    9.    0.   20.   71.   26.]
 [   2.    2.   41.    2.    6.   15.   20.   84.   44.   19.    9.    0.
     6.   16.    1.   11.    0.   11.   30.   76.]]

FRACTION = .0 train: 298 secs test: 17 secs
Accuracy:0.0459373340414
confusion matrix:
[[   5.    7.    8.    2.    0.    8.   26.    5.   26.   35.    1.    0.
    56.    2.    3.   84.   15.    9.    4.   23.]
 [  24.   14.    7.    4.    0.   12.   32.    0.   87.   37.    5.    0.
    54.    0.    6.   48.   42.   13.    1.   10.]
 [  30.   20.   28.    7.    0.   19.   13.    1.   33.   45.    3.    0.
    57.    0.    3.   82.   39.    8.    0.    9.]
 [   7.   26.   10.    3.    1.   22.   11.    1.   72.   36.    1.    0.
    49.    0.    6.   89.   25.   13.    2.   24.]
 [  17.    8.    7.    3.    0.    7.   39.    0.   74.   20.    2.    0.
    92.    0.    4.   73.   39.    5.    1.    5.]
 [  11.   17.   12.   12.    0.   21.   11.    1.   50.   44.    2.    0.
    82.    4.    7.   52.   53.    7.    3.    4.]
 [  22.   13.    7.    8.    0.   14.   42.    1.   55.   40.    4.    0.
    23.    1.   12.  103.   15.    9.    4.   17.]
 [  17.    5.    2.    6.    1.   26.   17.    5.   52.   43.    4.    2.
    58.    0.    2.   53.   69.   18.    3.    6.]
 [  13.   10.   29.   17.    0.   18.   38.    0.   49.   24.    5.    0.
    58.    1.    5.   65.   18.    6.    1.    7.]
 [  14.   13.   17.    2.    0.   27.   12.    5.   37.   61.    0.    0.
    41.    0.    3.  118.   34.   14.    0.    1.]
 [  17.    1.    3.    9.    0.   27.   38.    1.   53.   42.    2.    0.
    75.    0.    3.   36.   56.    9.    1.   12.]
 [  18.   23.   14.    8.    4.   24.   13.    4.   31.   58.    2.    0.
    81.    7.    5.   47.   44.    5.    5.    3.]
 [  12.   15.   18.    3.    0.   15.   36.    7.   49.   35.    4.    0.
    53.    1.    2.   66.   41.    6.    3.   10.]
 [  18.   26.   11.    7.    1.   38.   12.    4.   41.   26.   16.    0.
    97.    1.    8.   37.   32.   13.    1.    9.]
 [   8.    7.    0.    2.    0.   12.   49.    1.   95.   31.    1.    0.
    90.    0.    5.   35.   40.   10.    0.    6.]
 [  19.   11.   11.   17.    0.   13.   27.   19.   30.   22.    0.    0.
    54.    0.    2.   32.   42.    4.    0.    7.]
 [  26.    6.   13.    7.    0.   21.   20.    3.   25.   18.    1.    0.
    33.    3.    3.   51.   12.    5.    2.    2.]
 [  21.    6.   13.    3.    1.   16.   17.    6.  110.   56.    4.    0.
    47.    0.    1.   46.   28.    3.    3.   13.]
 [   7.   12.    9.    5.    3.   19.   53.    1.   69.   22.    0.    0.
    53.    3.    3.   59.   46.   10.    5.   15.]
 [  14.   10.    5.    1.    0.   21.   16.    4.   46.   37.    3.    0.
    72.    0.    3.   61.   76.   15.    6.    5.]]
