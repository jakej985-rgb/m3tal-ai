package nn

import (
	"math/rand"
	"testing"

	"m3tal-llm/internal/tensor"
)

func TestTrainerReducesLoss(t *testing.T) {
	dense, err := NewDense(1, 1, rand.New(rand.NewSource(19)))
	if err != nil {
		t.Fatal(err)
	}
	model := NewSequential(dense, &TanhLayer{})
	inputs, _ := tensor.NewMatrix(2, 1, []float64{-1, 1})
	targets, _ := tensor.NewMatrix(2, 1, []float64{-0.8, 0.8})

	reports, err := (Trainer{Epochs: 300, LearningRate: 0.1, ReportEvery: 50}).Fit(model, inputs, targets)
	if err != nil {
		t.Fatal(err)
	}
	if len(reports) < 2 {
		t.Fatalf("got %d loss reports, want at least 2", len(reports))
	}
	if reports[len(reports)-1].Loss >= reports[0].Loss {
		t.Fatalf("loss did not fall: start %.6f, end %.6f", reports[0].Loss, reports[len(reports)-1].Loss)
	}
	if reports[len(reports)-1].Epoch != 300 {
		t.Fatalf("final report epoch = %d, want 300", reports[len(reports)-1].Epoch)
	}
}

func TestTrainerRejectsInvalidConfiguration(t *testing.T) {
	dense, err := NewDense(1, 1, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	model := NewSequential(dense)
	inputs, _ := tensor.NewMatrix(1, 1, []float64{1})
	targets, _ := tensor.NewMatrix(1, 1, []float64{1})

	if _, err := (Trainer{Epochs: 0, LearningRate: 0.1}).Fit(model, inputs, targets); err == nil {
		t.Fatal("expected invalid epoch count error")
	}
	if _, err := (Trainer{Epochs: 1, LearningRate: 0}).Fit(model, inputs, targets); err == nil {
		t.Fatal("expected invalid learning rate error")
	}
}

func TestTrainerSupportsShuffledMiniBatchesAndAdam(t *testing.T) {
	dense, err := NewDense(1, 1, rand.New(rand.NewSource(23)))
	if err != nil {
		t.Fatal(err)
	}
	model := NewSequential(dense, &TanhLayer{})
	inputs, _ := tensor.NewMatrix(4, 1, []float64{-1, -0.5, 0.5, 1})
	targets, _ := tensor.NewMatrix(4, 1, []float64{-0.8, -0.4, 0.4, 0.8})

	reports, err := (Trainer{
		Epochs:      200,
		BatchSize:   2,
		Shuffle:     true,
		Seed:        5,
		ReportEvery: 50,
		Optimizer:   NewAdam(0.03),
	}).Fit(model, inputs, targets)
	if err != nil {
		t.Fatal(err)
	}
	if reports[len(reports)-1].Loss >= reports[0].Loss {
		t.Fatalf("mini-batch Adam loss did not fall: start %.6f, end %.6f", reports[0].Loss, reports[len(reports)-1].Loss)
	}
}
