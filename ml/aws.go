package ml

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/machinelearning"
)

const (
	modelID     = "ml-Dce8rRsvJGR"
	endpointURL = "https://realtime.machinelearning.us-east-1.amazonaws.com"
)

// NewSvc creates a new AWS MachineLearning service
func NewSvc() *machinelearning.MachineLearning {
	sess := session.New(aws.NewConfig().WithRegion("us-east-1"))

	if sess == nil {
		fmt.Println("failed to create session")
		return nil
	}

	svc := machinelearning.New(sess)

	return svc
}

// Predict predicts for a given complex C if it belongs to the mandelbrot set or not
func Predict(svc *machinelearning.MachineLearning, c complex128) bool {
	params := &machinelearning.PredictInput{
		MLModelId:       aws.String(modelID),
		PredictEndpoint: aws.String(endpointURL),
		Record: map[string]*string{
			"real": aws.String(fmt.Sprintf("%.6f", real(c))),
			"imag": aws.String(fmt.Sprintf("%.6f", imag(c))),
		},
	}

	resp, err := svc.Predict(params)

	if err != nil {
		fmt.Println(err.Error())
		return false
	}

	return *resp.Prediction.PredictedLabel == "1"
}
