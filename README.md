# K-Nearest-Neighbor

### Designing the Classifier
After dividing the data in two classes of train and test and normalizing them, we run the algorithm for different values of k. First we create a function for calculating Euclidean distance which is calculated with the following equation:

![image](https://user-images.githubusercontent.com/125180530/218412785-02c739a1-d074-4def-8809-8743d538d113.png)

Now consider a point or data that you want to classify. Calculate the distance between this point and all the other points and sort the calculated values. Then, choose first k points that have the least values for distances. They are the neighbors of this point. Among these neighbors, we choose the predicted label with the majority rule. It means considering a label of neighbors that repeats more than others. This is how KNN algorithm works.

For evaluating the created model we can calculate the model's accuracy. Also for better observation and comparison we report the confusion matrix. 

### Probability Distribution of classes
We have two classes: True and False. True is for the data in which the predicted labels and actual labels are the same, and False is for the data that these two labels are different. For both of these two classes we have to calculate the probabilities. For example, if true label of a data is 1 and the predicted label is also 1, then among the nearest neighbors we have to calculate the probability of the neighbors that have the same label. On the contrary, if true label of data is 1 and the predicted label is for example 2, we have to calculate the probability of the neighbors that doesn't have label 1. IN this case we use dictionary in the code. The labels in the dictionary are just the values of probabilities and the value of each bar will show the abundance of the corresponding probability. You can see the plots for different values of k. 
