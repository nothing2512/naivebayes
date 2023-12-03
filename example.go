package main

import (
	"main/bayes"
)

func train() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.SplitTrainData([]int{1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20})
	b.Train.GetRoot()
	b.Test.GetRoot()
}

func gain() {
	b := bayes.ParseObject([][]string{
		{"color", "root", "sound", "texture", "umbilicus", "surface", "ripe"},
		{"g", "c", "m", "c", "h", "h", "t"},
		{"d", "c", "d", "c", "h", "h", "t"},
		{"d", "c", "m", "c", "h", "h", "t"},
		{"g", "c", "d", "c", "h", "h", "t"},
		{"l", "c", "m", "c", "h", "h", "t"},
		{"g", "sc", "m", "c", "sh", "s", "t"},
		{"d", "sc", "m", "sb", "sh", "s", "t"},
		{"d", "sc", "m", "c", "sh", "h", "t"},
		{"d", "sc", "d", "sb", "sh", "h", "f"},
		{"g", "s", "c", "c", "f", "s", "f"},
		{"l", "s", "c", "b", "f", "h", "f"},
		{"l", "c", "m", "b", "f", "s", "f"},
		{"g", "sc", "m", "sb", "h", "h", "f"},
		{"l", "sc", "d", "sb", "h", "h", "f"},
		{"d", "sc", "m", "c", "sh", "s", "f"},
		{"l", "c", "m", "b", "f", "h", "f"},
		{"g", "c", "d", "sb", "sh", "h", "f"},
	})
	b.SetClassifier("ripe")
	b.ShowGains()
}

func DecisionTree() {
	b := bayes.ParseObject([][]string{
		{"color", "root", "sound", "texture", "umbilicus", "surface", "ripe"},
		{"g", "c", "m", "c", "h", "h", "t"},
		{"d", "c", "d", "c", "h", "h", "t"},
		{"d", "c", "m", "c", "h", "h", "t"},
		{"g", "c", "d", "c", "h", "h", "t"},
		{"l", "c", "m", "c", "h", "h", "t"},
		{"g", "sc", "m", "c", "sh", "s", "t"},
		{"d", "sc", "m", "sb", "sh", "s", "t"},
		{"d", "sc", "m", "c", "sh", "h", "t"},
		{"d", "sc", "d", "sb", "sh", "h", "f"},
		{"g", "s", "c", "c", "f", "s", "f"},
		{"l", "s", "c", "b", "f", "h", "f"},
		{"l", "c", "m", "b", "f", "s", "f"},
		{"g", "sc", "m", "sb", "h", "h", "f"},
		{"l", "sc", "d", "sb", "h", "h", "f"},
		{"d", "sc", "m", "c", "sh", "s", "f"},
		{"l", "c", "m", "b", "f", "h", "f"},
		{"g", "c", "d", "sb", "sh", "h", "f"},
	})
	b.SetClassifier("ripe")
	b.DecisionTree(true)
}

func predict() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.Predict([]string{"M", "Luxury", "Medium"})
}

func smoothing() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.Smooth(1)
	b.ShowTables()
}
