package main

import "github.com/nothing2512/naivebayes/bayes"

func main() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.DecisionTree(true)
}
