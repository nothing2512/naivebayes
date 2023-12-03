# Naive Bayes Golang

Naive Bayes library using golang

## setup
```shell
go get github.com/nothing2512/naivebayes
```

## Usage
### Smoothing
```go
func smoothing() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.Smooth(1)
	b.ShowTables()
}
```

### predict

```go
func predict() {
	b := bayes.ParseString("Gender,Car Type,Shirt Size,Class\nM,Family,Small,C0\nM,Sports,Medium,C0\nM,Sports,Medium,C0\nM,Sports,Large,C0\nM,Sports,Extra Large,C0\nM,Sports,Extra Large,C0\nF,Sports,Small,C0\nF,Sports,Small,C0\nF,Sports,Medium,C0\nF,Luxury,Large,C0\nM,Family,Large,C1\nM,Family,Extra Large,C1\nM,Family,Medium,C1\nM,Luxury,Extra Large,C1\nF,Luxury,Small,C1\nF,Luxury,Small,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Medium,C1\nF,Luxury,Large,C1")
	b.SetClassifier("Class")
	b.Predict([]string{"M", "Luxury", "Medium"})
}
```

### Show Decision Tree
```go
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
```

### Show Gain
```go
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
```

LICENSE
=====================

Copyright © `2023` `Robet Atiq Maulana Rifqi`

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.