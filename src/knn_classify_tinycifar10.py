from TinyCifar10Dataset import TinyCifar10Dataset
import FeatureVectorDataset as f
from KnnClassifier import KnnClassifier
from ImageVectorizer import ImageVectorizer




def classify(trainSet, predictSet, k, norm):
    classifier = KnnClassifier(k,norm)
    classifier.train(trainSet) # set trainingset
    correctPredictions = 0;
    for i in range(0,predictSet.size()):
        sample = predictSet.sample(i)
        prediction = classifier.predict(sample[0])
        if (prediction == sample[1]):
            correctPredictions = correctPredictions + 1;
        #print('{2}: Predicted {0}, should be {1}'.format(prediction,sample[1],i))
    accuracy = correctPredictions/predictSet.size() ;
    return accuracy

def gridSearch(trainSet, valSet, testSet):
    print("[Train] {0} samples".format(trainSet.size()));
    print("[Val] {0} samples".format(valSet.size()));
    minK = 1; # 1-nn
    maxK = min(50, max(5, round(trainSet.size() / 50))); # find a suitable maximal k value
    maxSteps =  20;  # should not do more step than that
    step = max(round(maxK/maxSteps-0.5),1); # skip some values if maxK too big
    norms = ['l1','l2']    
    bestResult = [0, minK, norms[0]];
    for norm in norms:             
        for k in range(minK, maxK, step):
            accuracy = classify(trainSet,valSet,k, norm);
            print("k={:02d}; cmp={}, accuracy: {:02.1f}%".format(k, norm, accuracy * 100));
            if (accuracy > bestResult[0]):
                bestResult = [accuracy, k, norm];

    bestK = bestResult[1];
    bestNorm = bestResult[2];
    print("Testing best combination ({}, {}) on test set ...".format(bestK, bestNorm))
    print("[Test] {0} samples".format(testSet.size()));
    testAccuracy = classify(trainSet,testSet,bestK, bestNorm);
    print("Accuracy: {:02.1f}%".format(testAccuracy * 100));



# The actual script
ratio = 0.1; #Default is 0.1


trainData = TinyCifar10Dataset("../data",'train',ratio)
valData = TinyCifar10Dataset("../data",'val',ratio)
testData = TinyCifar10Dataset("../data",'test',ratio)
trainSet = ImageVectorizer(trainData)
valSet = ImageVectorizer(valData)
testSet = ImageVectorizer(testData)


print('Performing random hyperparameter search ...');
gridSearch(trainSet, valSet, testSet);
print() # only here to set breakpoint
