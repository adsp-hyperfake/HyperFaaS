package fake

import "time"

// FunctionInstantModel is a debug model that returns the same values for all inputs.
type FunctionInstantModel struct {
	ImageTag string
	Runtime  time.Duration
	CPU      int64
	RAM      int64
}

// PredictFunction returns the same values for all inputs.
func (f *FunctionInstantModel) PredictFunction(inputs FunctionInputs) (FunctionPrediction, error) {
	return FunctionPrediction{
		Runtime:  f.Runtime,
		CPUUsage: f.CPU,
		RAMUsage: f.RAM,
	}, nil
}

func CreateInstantModels(imageTags []string) map[string]FunctionModel {
	models := make(map[string]FunctionModel)
	for _, imageTag := range imageTags {
		models[imageTag] = &FunctionInstantModel{
			ImageTag: imageTag,
		}
	}
	return models
}
