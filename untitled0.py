package main

import (
    "fmt"
    "github.com/e-XpertSolutions/go-cluster/cluster"
)

func main() {

    //input categorical data first must be dictionary-encoded to numbers - for example for values
    //"blue", "red", "green" it can be 1,2,3

    data := cluster.NewDenseMatrix(lineNumber, columnNumber, rawData)
    newData := cluster.NewDenseMatrix(newLineNumber, newColumnNumber, newRawData)


    //input parameters for the algorithm

    //distance and initialization functions may be chosen from the package or one may use 
    //custom functions with proper arguments
    distanceFunction := cluster.WeightedHammingDistance
    initializationFunction := cluster.InitCao

    //number of clusters and maximum number of iterations 
    clustersNumber := 5
    maxIteration := 20

    //weight vector - used to set importance of the features, bigger number means greater 
    //contribution to the cost function
    //vector must be of the same length as the number of features in dataset
    //it is not compulsory, if 'nil' then all features are treated equally (weight = 1)  
    weights := []float64{1,1,2}
    wvec := [][]float64{weights}

    //path to file where model will be saved or loaded from using LoadModel(), SaveModel()
    //if no need to load or save the model, can be set to empty string
    path = "km.txt"

    //KModes algorithm
    //initialization
    km := cluster.NewKModes(distanceFunction, initializationFunction, clustersNumber, 1, 
    maxIteration, wvec, "km.txt")


    //training
    //after training it is possible to access clusters centers vectors and computed labels
    //using km.ClusterCentroids and km.Labels
    err := km.FitModel(data)
    if err != nil {
        fmt.Println(err)
    }

    //predicting labels for new data
    newLabels, err := km.Predict(newData)
    if err != nil {
        fmt.Println(err)
    }


    //KPrototypes algorithm
    //it needs two more parameters than k-modes:
    //categorical - vector with numbers indicating columns with categorical features
    //gamma - float number, importance of cost contribution for numerical values
    categorical := []int{1} // means that only column number one contains categorical data
    gamma := 0.2 //cost from distance function for numerical data will be multiplied by 0.2

    //initialization
    kp := cluster.NewKPrototypes(distanceFunction, initializationFunction, categorical, 
    clustersNumber, 1, maxIteration, wvec, gamma, "km.txt")

    //training
    err := kp.FitModel(data)
    if err != nil {
        fmt.Println(err)
    }

    //predicting labels for new data
    newLabelsP, err := kp.Predict(newData)
    if err != nil {
        fmt.Println(err)
    }
}