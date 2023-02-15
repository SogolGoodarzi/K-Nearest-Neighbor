# K-Nearest-Neighbor

### Designing the Classifier
After dividing the data into two classes of train and test and normalizing them, we run the algorithm for different values of k. First, we create a function for calculating Euclidean distance which is calculated with the following equation:

![image](https://user-images.githubusercontent.com/125180530/218412785-02c739a1-d074-4def-8809-8743d538d113.png)

Now consider a point or data that you want to classify. Calculate the distance between this point and all the other points and sort the calculated values. Then, choose the first k points that have the least values for distances. They are the neighbors of this point. Among these neighbors, we choose the predicted label with the majority rule. It means considering a label of neighbors that repeats more than others. This is how the KNN algorithm works.

For evaluating the created model we can calculate the model's accuracy. Also for better observation and comparison, we report the confusion matrix. 

### Probability Distribution of classes
We have two classes: True and False. True is for the data in which the predicted labels and actual labels are the same, and False is for the data in which these two labels are different. For both of these two classes, we have to calculate the probabilities. For example, if the true label of a data is 1 and the predicted label is also 1, then among the nearest neighbors we have to calculate the probability of the neighbors that have the same label. On the contrary, if the true label of data is 1 and the predicted label is for example 2, we have to calculate the probability of the neighbors that don't have label 1. In this case, we use a dictionary in the code. The labels in the dictionary are just the values of probabilities and the value of each bar will show the abundance of the corresponding probability. You can see the plots for different values of k. 

### Improving the model using metric learning
In this part, we use two metrics LMNN and LFDA. 

#### **LMNN**
Large margin nearest neighbor (LMNN) classification is a statistical machine learning algorithm for metric learning. It learns a pseudometric designed for k-nearest neighbor classification. The goal of supervised learning (more specifically classification) is to learn a decision rule that can categorize data instances into pre-defined classes. The k-nearest neighbor rule assumes a training data set of labeled instances (i.e. the classes are known). It classifies a new data instance with the class obtained from the majority vote of the k closest (labeled) training instances. Closeness is measured with a pre-defined metric. Large margin nearest neighbors is an algorithm that learns this global in a supervised fashion to improve the classification accuracy of the k-nearest neighbor rule.

The main intuition behind LMNN is to learn a pseudometric under which all data instances in the training set are surrounded by at least k instances that share the same class label. If this is achieved, the leave-one-out error (a special case of cross-validation) is minimized. Let the training data consist of a data set D, where the set of possible class categories is C.

![image](https://user-images.githubusercontent.com/125180530/218716264-bbfc3067-e910-4e70-8bfa-cf6298a4043d.png)

![image](https://user-images.githubusercontent.com/125180530/218716348-049c1b77-1840-4373-b376-e8604b77f5fb.png)

The algorithm learns a pseudometric of the type:

![image](https://user-images.githubusercontent.com/125180530/218716546-9c40e6a4-a0fc-4487-99b4-766575534265.png)

For d(.,.) to be well defined, the matrix M needs to be positive and semi-definite. The Euclidean metric is a special case, where M is the identity matrix. This generalization is often referred to as the Mahalanobis metric. The figure below illustrates the effect of the metric under varying M. The two circles show the set of points with equal distance to the center xi. In the Euclidean case, this set is a circle, whereas under the modified (Mahalanobis) metric it becomes an ellipsoid. The algorithm distinguishes between two types of special data points: target neighbors and impostors.

![image](https://user-images.githubusercontent.com/125180530/218737289-96b9dcca-9163-4089-b3ca-f13b5fc3f620.png)

**Target neighbors:** Target neighbors are selected before learning. Each instance xi has exactly k different target neighbors within D, which all share the same class label yi. The target neighbors are the data points that should become nearest neighbors under the learned metric. The goal is that all of these k neighbors have the minimum distance from xi. The distance can be calculated from the below equation:

![image](https://user-images.githubusercontent.com/125180530/218735353-b5b661c2-3d7e-4138-87c0-e381d85e6f91.png)

**Impostors:** An impostor of a data point xi is another data point xj with a different class label which is one of the nearest neighbors of xi. During learning the algorithm tries to minimize the number of impostors for all data instances in the training set.

**Algorithm** 
Large margin nearest neighbors optimize the matrix M with the help of semidefinite programming. The objective is twofold: For every data point xi, the target neighbors should be close and the impostors should be far away. The above figure shows the effect of such optimization on an illustrative example. The learned metric causes the input vector xi to be surrounded by training instances of the same class. If it was a test point, it would be classified correctly under the k = 3 nearest neighbor rule. 

The first optimization goal is achieved by minimizing the average distance between instances and their target neighbors:

![image](https://user-images.githubusercontent.com/125180530/218737810-8708989e-859d-4005-b546-c4fab9eeffed.png)

The second goal is achieved by penalizing distances to impostors xl that is less than one unit further away than target neighbors xj (and therefore pushing them out of the local neighborhood of xi). The resulting value to be minimized can be stated as:

![image](https://user-images.githubusercontent.com/125180530/218738141-145b3559-cbef-411d-b572-d186080be0ce.png)

With a hinge loss function [.]+ = max(.,0), which ensures that impostor proximity is not penalized when outside the margin. The margin of exactly one unit fixes the scale of the matrix M. Any alternative choice c>0 would result in a rescaling of M by a factor of 1/c.

The final optimization problem becomes:

![image](https://user-images.githubusercontent.com/125180530/218738625-c0635e0b-ec3a-4295-bd0b-810864ce4494.png)

![image](https://user-images.githubusercontent.com/125180530/218739338-82f36bb5-8e33-4ca6-aa1b-5dcc83e0c044.png)

#### **LFDA**
Local Fisher Discriminant Analysis (LFDA) mainly deals with dimensionality reduction following the likes of traditional Fisher discriminant analysis (FDA). However, FDA tends to give undesired results if samples in some class form several separate clusters, i.e., multimodal. LFDA solves this problem faced by the FDA by taking the local structure of the data into account so the multimodal data can be embedded appropriately.

As the primary dataset has 13 dimensions, illustrating this data is not possible in 2 dimension space. We have to reduce the dimensions from 13 to 2 with the use of metric learning methods. After reducing dimensions, we show the scatter plot of the data for three values of k (1, 5, 15). We do this process for both of the metric learning methods: LMNN and LFDA. By comparing the plots we can choose the best value for k which results in better separability of data. 

ّFor the best value of k from the reducing dimension process, we report the value of accuracy and confusion matrix for a different number of neighbors (k = 1, 5, 10, 20). After comparing the results reached from both methods, you can make a conclusion and find out which method is better. 

### Correlation Coefficient
One parameter that can show the correlation between feature sets is the correlation coefficient. So, if we want to show the correlation between every two columns of features, we can report the correlation coefficients in a matrix called correlation matrix. 

Now, we report the correlation matrix for the data after applying metric learning methods. We can see that the best method is LFDA because its correlation matrix is more like an eye matrix than the other method. For LMNN, off-diagonal elements are larger than in the LFDA method. So, by using LFDA we could better separate data into classes. In other words, after using LFDA, the distance between the data with unsimilar features is increased and the distance between the data with similar features which are in the same class is decreased. This was the main goal of data classification. 

There is another metric learning method that is presented as a new metric. You can get familiar with this method at this link: https://arxiv.org/pdf/1607.05002.pdf
