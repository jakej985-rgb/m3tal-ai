package nn

import (
	"fmt"
	"math/rand"

	"m3tal-ai/internal/tensor"
)

// Trainer performs gradient-based training with optional mini-batches and
// shuffling. If Optimizer is nil, it uses SGD with LearningRate.
type Trainer struct {
	Epochs       int
	LearningRate float64
	BatchSize    int
	ReportEvery  int
	Shuffle      bool
	Seed         int64
	Optimizer    Optimizer
	Loss         Loss
}

// EpochLoss is a loss observation recorded during training.
type EpochLoss struct {
	Epoch int
	Loss  float64
}

// Fit trains model on inputs and targets for the configured number of epochs.
// BatchSize 0 means full-batch training. Loss observations include epoch 0,
// each ReportEvery epochs, and the final loss after the last update.
func (trainer Trainer) Fit(model *Sequential, inputs, targets tensor.Matrix) ([]EpochLoss, error) {
	if model == nil || len(model.layers) == 0 {
		return nil, fmt.Errorf("trainer requires a model with at least one layer")
	}
	if trainer.Epochs <= 0 {
		return nil, fmt.Errorf("epoch count must be positive")
	}
	if trainer.BatchSize < 0 {
		return nil, fmt.Errorf("batch size cannot be negative")
	}
	if trainer.ReportEvery < 0 {
		return nil, fmt.Errorf("report interval cannot be negative")
	}
	if inputs.Rows() == 0 {
		return nil, fmt.Errorf("training data must contain at least one row")
	}
	if targets.Rows() != inputs.Rows() {
		return nil, fmt.Errorf("input/target batch mismatch: %d and %d rows", inputs.Rows(), targets.Rows())
	}
	if inputs.Cols() == 0 || targets.Cols() == 0 {
		return nil, fmt.Errorf("training data must contain at least one feature and target")
	}

	optimizer := trainer.Optimizer
	if optimizer == nil {
		optimizer = SGD{LearningRate: trainer.LearningRate}
	}
	if err := optimizer.Validate(); err != nil {
		return nil, err
	}
	objective := trainer.Loss
	if objective == nil {
		objective = MSE{}
	}
	batchSize := trainer.BatchSize
	if batchSize == 0 || batchSize > inputs.Rows() {
		batchSize = inputs.Rows()
	}
	rng := rand.New(rand.NewSource(trainer.Seed))
	reports := make([]EpochLoss, 0)

	for epoch := 0; epoch < trainer.Epochs; epoch++ {
		if epoch == 0 || trainer.ReportEvery > 0 && epoch%trainer.ReportEvery == 0 {
			prediction, err := model.Forward(inputs)
			if err != nil {
				return nil, err
			}
			loss, err := objective.Value(prediction, targets)
			if err != nil {
				return nil, err
			}
			reports = append(reports, EpochLoss{Epoch: epoch, Loss: loss})
		}

		order := make([]int, inputs.Rows())
		for i := range order {
			order[i] = i
		}
		if trainer.Shuffle {
			rng.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
		}
		for start := 0; start < len(order); start += batchSize {
			end := start + batchSize
			if end > len(order) {
				end = len(order)
			}
			batchIndices := order[start:end]
			batchInputs, err := inputs.SelectRows(batchIndices)
			if err != nil {
				return nil, err
			}
			batchTargets, err := targets.SelectRows(batchIndices)
			if err != nil {
				return nil, err
			}

			model.ZeroGrad()
			prediction, err := model.Forward(batchInputs)
			if err != nil {
				return nil, err
			}
			gradient, err := objective.Gradient(prediction, batchTargets)
			if err != nil {
				return nil, err
			}
			if _, err := model.Backward(gradient); err != nil {
				return nil, err
			}
			optimizer.Update(model.Parameters())
		}
	}

	finalPrediction, err := model.Forward(inputs)
	if err != nil {
		return nil, err
	}
	finalLoss, err := objective.Value(finalPrediction, targets)
	if err != nil {
		return nil, err
	}
	reports = append(reports, EpochLoss{Epoch: trainer.Epochs, Loss: finalLoss})
	return reports, nil
}
