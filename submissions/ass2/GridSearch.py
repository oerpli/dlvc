import FeatureVectorDataset as f
from KnnClassifier import KnnClassifier

# This Class implements the knn Grid Search for different 
# parameter for k and the distance norm

class GridSearch:
    norms = []      
    K = range(1,5)
    defer_init = False

    def defaultValues(self,trainSet): 
        # default if no values provided, calculates suitable ranges for k
        minK = 1 # 1-nn  
        maxK = min(50, max(5, round(trainSet.size() / 50))) # find a suitable maximal k value
        maxSteps = 20  # should not do more step than that
        step = max(round(maxK / maxSteps - 0.5),1) # skip some values if maxK too big
        self.norms = ['l1','l2']

        self.K = range(minK, maxK, step)

    def __init__(self, k_range=-1 , cmp_range=-1):
        if(k_range == -1):
            self.defer_init = True
        else:
            self.norms = cmp_range
            self.K = k_range

    def classify(self,trainSet, predictSet, classifier):
        classifier.train(trainSet) # set trainingset
        correctPredictions = 0
        for i in range(0,predictSet.size()):
            workdone = i / predictSet.size()
            print("\r  k={1:02d}; cmp={2}, finished: {0:.1f}%".format(workdone * 100,classifier.K,classifier.ns), end="", flush=True)
            sample = predictSet.sample(i)
            prediction = classifier.predict(sample[0])
            if (prediction == sample[1]):
                correctPredictions = correctPredictions + 1
        accuracy = correctPredictions / predictSet.size()
        return accuracy

    def gridSearch(self,trainSet, valSet, testSet):
        if(self.defer_init):
            self.defaultValues(trainSet)
            self.defer_init = True
        print('Performing grid search ...')
        print("  [train] {0} samples".format(trainSet.size()))
        print("  [val] {0} samples".format(valSet.size()))

        bestResult = (-1, 0, 0)  # Array to store the best result so far
        for norm in self.norms:
            for k in self.K:
                accuracy = self.classify(trainSet,valSet,KnnClassifier(k,norm))
                print("\r  k={:02d}; cmp={}, accuracy: {:02.1f}%".format(k, norm, accuracy * 100))
                if (accuracy > bestResult[0]):
                    bestResult = (accuracy, k, norm)

        # Now test the best parameters on the test set.

        bestK = bestResult[1]
        bestNorm = bestResult[2]

        print("Testing best combination ({}, {}) on test set ...".format(bestK, bestNorm))
        print("  [test] {0} samples".format(testSet.size()))
        testAccuracy = self.classify(trainSet,testSet,KnnClassifier(bestK, bestNorm))
        print("\r  Accuracy: {:02.1f}%{:30s}".format(testAccuracy * 100,' ' * 30))

