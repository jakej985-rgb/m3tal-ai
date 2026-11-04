package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"m3tal-ai/internal/lm"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "m3tal-lm:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: go run ./cmd/m3tal-lm train|eval|generate|instructions [flags]")
	}
	switch args[0] {
	case "train":
		flags := flag.NewFlagSet("train", flag.ContinueOnError)
		corpusArg := flags.String("corpus", "", "text file or path-list of directories containing training text")
		evalArg := flags.String("eval-corpus", "", "optional separate held-out text path-list")
		checkpoint := flags.String("checkpoint", "models/m3tal-small.json", "output checkpoint path")
		baseModel := flags.String("base-model", "", "path to existing checkpoint to fine-tune")
		pretrainCorpus := flags.String("pretrain-corpus", "", "optional corpus path to pre-train on before instruction tuning")
		pretrainEpochs := flags.Int("pretrain-epochs", 5, "pretraining epochs if --pretrain-corpus is provided")
		includeInstructions := flags.Bool("include-instructions", false, "include curated self-owned instruction examples in training corpus")
		instructMode := flags.Bool("instruct", false, "train using prompt-loss masking on instruction examples")
		epochs := flags.Int("epochs", 2, "training epochs")
		steps := flags.Int("steps", 40, "optimizer steps per epoch")
		vocab := flags.Int("vocab", 384, "BPE tokenizer vocabulary size")
		contextLength := flags.Int("context", 512, "maximum context tokens (up to 2048)")
		width := flags.Int("width", 64, "model width (up to 512)")
		layers := flags.Int("layers", 4, "number of transformer layers (1-16)")
		heads := flags.Int("heads", 4, "number of attention heads (must divide width)")
		seed := flags.Int64("seed", 17, "deterministic initialization seed")
		learningRate := flags.Float64("learning-rate", 0.001, "Adam learning rate")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}

		var model *lm.Model
		var tokenizer *lm.Tokenizer
		var texts []string

		if strings.TrimSpace(*baseModel) != "" {
			var err error
			model, err = lm.LoadCheckpoint(*baseModel)
			if err != nil {
				return fmt.Errorf("load base model: %w", err)
			}
			tokenizer = model.Tokenizer
			if strings.TrimSpace(*corpusArg) != "" {
				texts, err = lm.LoadCorpus(strings.Split(*corpusArg, string(os.PathListSeparator)))
				if err != nil {
					return err
				}
			}
		} else {
			if strings.TrimSpace(*corpusArg) == "" && strings.TrimSpace(*pretrainCorpus) == "" {
				return fmt.Errorf("--corpus or --pretrain-corpus is required; only use text you are licensed to train on")
			}
			var pretrainTexts []string
			if strings.TrimSpace(*pretrainCorpus) != "" {
				var err error
				pretrainTexts, err = lm.LoadCorpus(strings.Split(*pretrainCorpus, string(os.PathListSeparator)))
				if err != nil {
					return fmt.Errorf("load pretrain corpus: %w", err)
				}
			}
			if strings.TrimSpace(*corpusArg) != "" {
				var err error
				texts, err = lm.LoadCorpus(strings.Split(*corpusArg, string(os.PathListSeparator)))
				if err != nil {
					return err
				}
			} else if len(pretrainTexts) > 0 {
				texts = pretrainTexts
			}

			tokenizerTexts := append([]string(nil), texts...)
			if len(pretrainTexts) > 0 {
				tokenizerTexts = append(tokenizerTexts, pretrainTexts...)
			}
			if *includeInstructions || *instructMode {
				instTexts := lm.FormatInstructionCorpus(lm.DefaultInstructionSet())
				for rep := 0; rep < 5; rep++ {
					tokenizerTexts = append(tokenizerTexts, instTexts...)
				}
			}
			var err error
			tokenizer, err = lm.TrainTokenizer(tokenizerTexts, *vocab)
			if err != nil {
				return err
			}
			model, err = lm.NewModel(lm.Config{
				Context: *contextLength,
				Width:   *width,
				Layers:  *layers,
				Heads:   *heads,
			}, tokenizer, *seed)
			if err != nil {
				return err
			}

			if len(pretrainTexts) > 0 && *pretrainEpochs > 0 {
				fmt.Printf("pre-training on %d corpus documents (%d epochs, %d steps/epoch)...\n", len(pretrainTexts), *pretrainEpochs, *steps)
				pretrainLosses, err := model.Train(pretrainTexts, *pretrainEpochs, *steps, *learningRate)
				if err != nil {
					return fmt.Errorf("pretraining: %w", err)
				}
				for i, loss := range pretrainLosses {
					fmt.Printf("pretrain epoch %d loss %.4f\n", i+1, loss)
				}
			}
		}

		if *includeInstructions && !*instructMode {
			texts = append(texts, lm.FormatInstructionCorpus(lm.DefaultInstructionSet())...)
		}

		var losses []float64
		var err error
		if *instructMode {
			losses, err = model.TrainInstructions(lm.DefaultInstructionSet(), *epochs, *steps, *learningRate)
		} else {
			if len(texts) == 0 {
				return fmt.Errorf("no training texts available; provide --corpus")
			}
			losses, err = model.Train(texts, *epochs, *steps, *learningRate)
		}
		if err != nil {
			return err
		}
		if err := model.SaveCheckpoint(*checkpoint); err != nil {
			return err
		}
		for i, loss := range losses {
			fmt.Printf("epoch %d loss %.4f\n", i+1, loss)
		}
		if strings.TrimSpace(*evalArg) != "" {
			evalTexts, err := lm.LoadCorpus(strings.Split(*evalArg, string(os.PathListSeparator)))
			if err != nil {
				return err
			}
			evalLoss, err := model.Evaluate(evalTexts)
			if err != nil {
				return err
			}
			fmt.Printf("held-out loss %.4f perplexity %.2f\n", evalLoss, math.Exp(evalLoss))
		}
		fmt.Printf("saved checkpoint: %s (vocabulary %d, parameters %d)\n", *checkpoint, len(tokenizer.Tokens), model.ParameterCount())
		return nil
	case "eval":
		flags := flag.NewFlagSet("eval", flag.ContinueOnError)
		checkpoint := flags.String("checkpoint", "models/m3tal-small.json", "model checkpoint path")
		corpusArg := flags.String("corpus", "", "text file or path-list of directories containing evaluation text")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if strings.TrimSpace(*corpusArg) == "" {
			return fmt.Errorf("--corpus is required")
		}
		model, err := lm.LoadCheckpoint(*checkpoint)
		if err != nil {
			return err
		}
		texts, err := lm.LoadCorpus(strings.Split(*corpusArg, string(os.PathListSeparator)))
		if err != nil {
			return err
		}
		evalLoss, err := model.Evaluate(texts)
		if err != nil {
			return err
		}
		fmt.Printf("evaluation loss %.4f perplexity %.2f\n", evalLoss, math.Exp(evalLoss))
		return nil
	case "generate":
		flags := flag.NewFlagSet("generate", flag.ContinueOnError)
		checkpoint := flags.String("checkpoint", "models/m3tal-small.json", "model checkpoint path")
		prompt := flags.String("prompt", "", "generation prompt")
		count := flags.Int("tokens", 80, "number of new tokens")
		temperature := flags.Float64("temperature", 0.8, "sampling temperature")
		seed := flags.Int64("seed", 19, "sampling seed")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		model, err := lm.LoadCheckpoint(*checkpoint)
		if err != nil {
			return err
		}
		generated, err := model.Generate(*prompt, *count, *temperature, *seed)
		if err != nil {
			return err
		}
		fmt.Println(generated)
		return nil
	case "instructions":
		flags := flag.NewFlagSet("instructions", flag.ContinueOnError)
		output := flags.String("output", "data/instructions.txt", "output instruction corpus path")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		examples := lm.DefaultInstructionSet()
		if err := lm.WriteInstructionCorpusFile(*output, examples); err != nil {
			return err
		}
		fmt.Printf("wrote %d instruction examples to %s\n", len(examples), *output)
		return nil
	default:
		return fmt.Errorf("unknown command %q; use train, eval, generate, or instructions", args[0])
	}
}
