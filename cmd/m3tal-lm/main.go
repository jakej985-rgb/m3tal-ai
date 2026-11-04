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
		includeInstructions := flags.Bool("include-instructions", false, "include curated self-owned instruction examples in training corpus")
		instructMode := flags.Bool("instruct", false, "train using prompt-loss masking on instruction examples")
		epochs := flags.Int("epochs", 2, "training epochs")
		steps := flags.Int("steps", 40, "optimizer steps per epoch")
		vocab := flags.Int("vocab", 320, "BPE tokenizer vocabulary size")
		contextLength := flags.Int("context", 32, "maximum context tokens")
		width := flags.Int("width", 16, "model width")
		seed := flags.Int64("seed", 17, "deterministic initialization seed")
		learningRate := flags.Float64("learning-rate", 0.001, "Adam learning rate")
		if err := flags.Parse(args[1:]); err != nil {
			return err
		}
		if strings.TrimSpace(*corpusArg) == "" {
			return fmt.Errorf("--corpus is required; only use text you are licensed to train on")
		}
		texts, err := lm.LoadCorpus(strings.Split(*corpusArg, string(os.PathListSeparator)))
		if err != nil {
			return err
		}
		if *includeInstructions {
			texts = append(texts, lm.FormatInstructionCorpus(lm.DefaultInstructionSet())...)
		}
		tokenizer, err := lm.TrainTokenizer(texts, *vocab)
		if err != nil {
			return err
		}
		model, err := lm.NewModel(lm.Config{Context: *contextLength, Width: *width}, tokenizer, *seed)
		if err != nil {
			return err
		}
		var losses []float64
		if *instructMode {
			losses, err = model.TrainInstructions(lm.DefaultInstructionSet(), *epochs, *steps, *learningRate)
		} else {
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
