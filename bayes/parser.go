package bayes

import (
	"strings"
)

func ParseString(str string) *NaiveBayes {
	str = strings.ReplaceAll(str, "\r", "")
	strs := strings.Split(str, "\n")
	headers := strings.Split(strs[0], ",")
	var body [][]string
	for _, x := range strs[1:] {
		body = append(body, strings.Split(x, ","))
	}

	return &NaiveBayes{
		headers: headers,
		body:    body,
	}
}

func ParseObject(obj [][]string) *NaiveBayes {
	return &NaiveBayes{
		headers: obj[0],
		body:    obj[1:],
	}
}
