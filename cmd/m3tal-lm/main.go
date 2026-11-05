package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"

	"m3tal-llm/internal/lm"
	"m3tal-llm/internal/tensor/kernel"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "m3tal-lm:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: go run ./cmd/m3tal-lm train|eval|generate|instructions|dataset|backends|model-info [flags]")
	}
	switch args[0] {
	case "backends":
		fmt.Println(kernel.Report())
		fmt.Printf("selected backend: %s\n", kernel.Selected())
		return nil
	case "dataset":
		return runDataset(args[1:])
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
		width := flags.Int("width", 64, "model width (up to 2048)")
		layers := flags.Int("layers", 4, "number of transformer layers (1-16)")
		heads := flags.Int("heads", 4, "number of attention heads (must divide width)")
		seed := flags.Int64("seed", 17, "deterministic initialization seed")
		learningRate := flags.Float64("learning-rate", 0.001, "Adam learning rate")
		datasetArg := flags.String("dataset", "", "sharded token dataset directory (built via 'dataset build')")
		orderArg := flags.String("order", "sequential", "dataset shard order: sequential or shuffled")
		maxSteps := flags.Int64("max-steps", 0, "optimizer steps to run (0 = until data exhausted)")
		logEvery := flags.Int64("log-every", 20, "log metrics every N steps (0 = silent)")
		warmup := flags.Int64("warmup-steps", 0, "linear LR warmup steps")
		minLRRatio := flags.Float64("min-lr-ratio", 0.1, "cosine LR decay floor as fraction of peak")
		gradClip := flags.Float64("grad-clip", 1.0, "global gradient-norm clip threshold")
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
		} else if strings.TrimSpace(*datasetArg) != "" {
			ds, err := lm.OpenDataset(*datasetArg)
			if err != nil {
				return fmt.Errorf("open dataset: %w", err)
			}
			if fp := lm.TokenizerFingerprint(model.Tokenizer); fp != ds.Meta.Tokenizer {
				return fmt.Errorf("dataset tokenizer fingerprint %s does not match model tokenizer %s", ds.Meta.Tokenizer, fp)
			}
			var order lm.TokenOrder
			switch *orderArg {
			case "sequential":
				order = lm.Sequential
			case "shuffled":
				order = lm.ShuffledSeeded
			default:
				return fmt.Errorf("--order must be sequential or shuffled")
			}
			maxSteps := *maxSteps
			if maxSteps == 0 {
				maxSteps = int64(*epochs) * int64(*steps)
			}
			stream := lm.NewTokenStream(ds, order, *seed)
			cfg := lm.TrainConfig{
				MaxSteps:     maxSteps,
				LearningRate: *learningRate,
				WarmupSteps:  *warmup,
				MinLRRatio:   *minLRRatio,
				GradClip:     *gradClip,
				LogEvery:     *logEvery,
				Seed:         *seed,
			}
			loss, err := model.TrainStream(stream, cfg, func(m lm.TrainMetrics) {
				fmt.Printf("step %d loss %.4f lr %.2e tok/s %.0f grad %.3f\n",
					m.Step, m.Loss, m.LearningRate, m.TokensPerSec, m.GradNorm)
			})
			if err != nil {
				return fmt.Errorf("train: %w", err)
			}
			losses = []float64{loss}
		} else {
			if len(texts) == 0 {
				return fmt.Errorf("no training texts available; provide --corpus or --dataset")
			}
			var tokens []uint32
			for _, text := range texts {
				for _, id := range model.Tokenizer.Encode(text) {
					tokens = append(tokens, uint32(id))
				}
			}
			source := lm.NewSliceSource(tokens)
			cfg := lm.TrainConfig{
				MaxSteps:     int64(*epochs) * int64(*steps),
				LearningRate: *learningRate,
				WarmupSteps:  *warmup,
				MinLRRatio:   *minLRRatio,
				GradClip:     *gradClip,
				LogEvery:     *logEvery,
				Seed:         *seed,
			}
			loss, err := model.TrainStream(source, cfg, func(m lm.TrainMetrics) {
				fmt.Printf("step %d loss %.4f lr %.2e tok/s %.0f grad %.3f\n",
					m.Step, m.Loss, m.LearningRate, m.TokensPerSec, m.GradNorm)
			})
			if err != nil {
				return fmt.Errorf("train: %w", err)
			}
			losses = []float64{loss}
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
	case "model-info":
		return modelInfo(args[1:])
	case "eval-bench":
		return evalBench(args[1:])
	default:
		return fmt.Errorf("unknown command %q; use train, eval, generate, instructions, dataset, backends, or model-info", args[0])
	}
}

// runDataset implements the "dataset" subcommand for building sharded token
// datasets from licensed corpus sources with provenance, optional
// document-level train/validation split, dedup, and resumable shards.
func runDataset(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: m3tal-lm dataset build|report|verify|split [flags]")
	}
	switch args[0] {
	case "build":
		return datasetBuild(args[1:])
	case "report":
		return datasetReport(args[1:])
	case "verify":
		return datasetVerify(args[1:])
	case "split":
		return datasetSplit(args[1:])
	case "annotate":
		return datasetAnnotate(args[1:])
	case "export":
		return datasetExport(args[1:])
	default:
		return fmt.Errorf("unknown dataset subcommand %q; use build, report, verify, or split", args[0])
	}
}

// chainFeed concatenates TokenFeeds so skip/consume operate across
// source boundaries during resumed builds.
type chainFeed struct {
	feeds []lm.TokenFeed
	cur   int
	err   error
}

func (c *chainFeed) Next(n int) ([]uint32, bool) {
	for c.cur < len(c.feeds) {
		chunk, ok := c.feeds[c.cur].Next(n)
		if ok {
			return chunk, true
		}
		if err := c.feeds[c.cur].Err(); err != nil {
			c.err = err
			return nil, false
		}
		c.cur++
	}
	return nil, false
}

func (c *chainFeed) Err() error {
	if c.err != nil {
		return c.err
	}
	if c.cur < len(c.feeds) {
		return c.feeds[c.cur].Err()
	}
	return nil
}

// docSplit assigns a document deterministically to the validation split
// when its hash falls below frac of the uint64 range.
func docSplit(text string, frac float64) bool {
	if frac <= 0 {
		return false
	}
	h := sha256.Sum256([]byte(text))
	u := binary.BigEndian.Uint64(h[:8])
	return float64(u)/float64(^uint64(0)) < frac
}

func datasetBuild(args []string) error {
	flags := flag.NewFlagSet("dataset build", flag.ContinueOnError)
	corpusArg := flags.String("corpus", "", "comma-separated corpus file paths (legacy; prefer --sources)")
	sourcesFile := flags.String("sources", "", "JSON file listing licensed corpus sources")
	checkpoint := flags.String("checkpoint", "models/m3tal-small.json", "checkpoint providing the tokenizer")
	outDir := flags.String("out", "", "output dataset directory")
	shardTokensFlag := flags.Int64("shard-tokens", 0, "tokens per shard (0 = single shard)")
	valFrac := flags.Float64("val-frac", 0, "fraction of documents held out for validation (document-level split)")
	dedup := flags.String("dedup", "none", "dedup strategy: none or exact (sha256 per document)")
	resume := flags.Bool("resume", false, "resume an existing partial dataset build")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*outDir) == "" {
		return fmt.Errorf("--out is required")
	}
	if *valFrac < 0 || *valFrac > 0.5 {
		return fmt.Errorf("--val-frac must be in [0, 0.5]")
	}
	if *dedup != "none" && *dedup != "exact" {
		return fmt.Errorf("--dedup must be none or exact")
	}

	var sources []lm.CorpusSource
	if *sourcesFile != "" {
		var err error
		sources, err = lm.LoadSources(*sourcesFile)
		if err != nil {
			return err
		}
	} else {
		if strings.TrimSpace(*corpusArg) == "" {
			return fmt.Errorf("--sources or --corpus is required")
		}
		var paths []string
		for _, p := range strings.Split(*corpusArg, ",") {
			if p = strings.TrimSpace(p); p != "" {
				paths = append(paths, p)
			}
		}
		sources = []lm.CorpusSource{{
			Name:          "local-corpus",
			Kind:          "text",
			Paths:         paths,
			License:       "unreviewed-local",
			LicenseReview: "paths supplied via --corpus; no external license review performed — do not redistribute the resulting model without reviewing source terms",
			Dedup:         *dedup,
			Filters:       []string{"utf8-valid", "size-caps", "sensitive-file-skip"},
		}}
	}
	// Record the dedup strategy actually used on every source.
	for i := range sources {
		sources[i].Dedup = *dedup
	}

	model, err := lm.LoadCheckpoint(*checkpoint)
	if err != nil {
		return fmt.Errorf("load checkpoint tokenizer: %w", err)
	}

	shardTokens := int(*shardTokensFlag)
	if shardTokens <= 0 {
		shardTokens = 1 << 20 // 1M tokens per shard default
	}
	valDir := *outDir + "-val"
	trainBuilder, err := lm.NewDatasetBuilder(*outDir, model.Tokenizer, shardTokens, *resume)
	if err != nil {
		return fmt.Errorf("open train dataset: %w", err)
	}
	var valBuilder *lm.DatasetBuilder
	if *valFrac > 0 {
		valBuilder, err = lm.NewDatasetBuilder(valDir, model.Tokenizer, shardTokens, *resume)
		if err != nil {
			return fmt.Errorf("open val dataset: %w", err)
		}
	}

	var stats lm.IngestStats
	seen := make(map[[32]byte]struct{})
	var trainFeeds, valFeeds []lm.TokenFeed

	for si := range sources {
		src := &sources[si]
		if err := src.Validate(); err != nil {
			return fmt.Errorf("source %q: %w", src.Name, err)
		}
		if err := lm.HashSource(src); err != nil {
			return fmt.Errorf("hash source %q: %w", src.Name, err)
		}
		if src.Kind == "tokendataset" {
			if *valFrac > 0 {
				return fmt.Errorf("source %q is a tokendataset; --val-frac requires text sources (use 'dataset split' for shard-level splits)", src.Name)
			}
			ds, err := lm.OpenDataset(src.Paths[0])
			if err != nil {
				return fmt.Errorf("source %q: %w", src.Name, err)
			}
			if ds.Meta.Tokenizer != lm.TokenizerFingerprint(model.Tokenizer) {
				return fmt.Errorf("source %q tokenizer %q does not match checkpoint tokenizer %q", src.Name, ds.Meta.Tokenizer, lm.TokenizerFingerprint(model.Tokenizer))
			}
			trainFeeds = append(trainFeeds, lm.NewTokenStream(ds, lm.Sequential, 0))
			stats.TokensTrain += ds.Meta.TotalTokens
			stats.DocumentsIn += ds.Meta.TotalTokens // token-stream source: docs counted as tokens
			stats.DocumentsKept += ds.Meta.TotalTokens
			continue
		}
		texts, err := lm.LoadCorpus(src.Paths)
		if err != nil {
			return fmt.Errorf("source %q: %w", src.Name, err)
		}
		stats.DocumentsIn += int64(len(texts))
		var trainToks, valToks []uint32
		for _, text := range texts {
			if *dedup == "exact" {
				sum := sha256.Sum256([]byte(text))
				if _, dup := seen[sum]; dup {
					stats.DocumentsDup++
					continue
				}
				seen[sum] = struct{}{}
			}
			stats.DocumentsKept++
			ids := model.Tokenizer.Encode(text)
			ids = append(ids, model.Tokenizer.Encode("\n")...)
			if valBuilder != nil && docSplit(text, *valFrac) {
				for _, id := range ids {
					valToks = append(valToks, uint32(id))
				}
			} else {
				for _, id := range ids {
					trainToks = append(trainToks, uint32(id))
				}
			}
		}
		if len(trainToks) > 0 {
			trainFeeds = append(trainFeeds, lm.NewSliceSource(trainToks))
		}
		if len(valToks) > 0 {
			valFeeds = append(valFeeds, lm.NewSliceSource(valToks))
		}
	}

	// Resume realignment: skip tokens already present in each builder.
	trainChain := &chainFeed{feeds: trainFeeds}
	if skip := trainBuilder.PendingSkip(); skip > 0 {
		if err := lm.SkipFeed(trainChain, skip); err != nil {
			return fmt.Errorf("resume skip (train): %w", err)
		}
	}
	if err := trainBuilder.Consume(trainChain); err != nil {
		return fmt.Errorf("build train dataset: %w", err)
	}
	if err := trainChain.Err(); err != nil {
		return fmt.Errorf("train token feed: %w", err)
	}
	if valBuilder != nil {
		valChain := &chainFeed{feeds: valFeeds}
		if skip := valBuilder.PendingSkip(); skip > 0 {
			if err := lm.SkipFeed(valChain, skip); err != nil {
				return fmt.Errorf("resume skip (val): %w", err)
			}
		}
		if err := valBuilder.Consume(valChain); err != nil {
			return fmt.Errorf("build val dataset: %w", err)
		}
		if err := valChain.Err(); err != nil {
			return fmt.Errorf("val token feed: %w", err)
		}
	}

	trainMeta, err := trainBuilder.Close()
	if err != nil {
		return fmt.Errorf("finalize train dataset: %w", err)
	}
	stats.TokensTrain = trainMeta.TotalTokens
	stats.TokensVal = 0
	valProduced := false
	if valBuilder != nil {
		valMeta, err := valBuilder.Close()
		if err != nil {
			return fmt.Errorf("finalize val dataset: %w", err)
		}
		if valMeta.ShardCount == 0 {
			// No documents landed in validation (e.g. single-document
			// corpus); drop the empty output rather than leave an
			// unopenable manifest behind.
			_ = os.RemoveAll(valDir)
			valBuilder = nil
		} else {
			stats.TokensVal = valMeta.TotalTokens
			valProduced = true
		}
	}

	splitDesc := "none"
	if valProduced {
		splitDesc = fmt.Sprintf("document-hash val_frac=%.4f -> %s", *valFrac, valDir)
	}
	prov := lm.NewProvenance(sources, pipelineHash(), splitDesc, stats)
	if err := lm.WriteProvenance(*outDir, prov); err != nil {
		return fmt.Errorf("write provenance: %w", err)
	}
	if valBuilder != nil {
		if err := lm.WriteProvenance(valDir, prov); err != nil {
			return fmt.Errorf("write val provenance: %w", err)
		}
	}
	fmt.Printf("wrote dataset to %s: %d shards, %d tokens, tokenizer %s\n",
		*outDir, trainMeta.ShardCount, trainMeta.TotalTokens, trainMeta.Tokenizer)
	if valBuilder != nil {
		fmt.Printf("wrote validation dataset to %s: %d tokens\n", valDir, stats.TokensVal)
	}
	fmt.Printf("provenance: %d sources, %s\n", len(sources), lm.DedupReport(stats.DocumentsKept, stats.DocumentsDup))
	return nil
}

func pipelineHash() string {
	if out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output(); err == nil {
		return strings.TrimSpace(string(out))
	}
	return "unknown"
}

func datasetReport(args []string) error {
	flags := flag.NewFlagSet("dataset report", flag.ContinueOnError)
	jsonOut := flags.Bool("json", false, "emit full JSON report")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: m3tal-lm dataset report [--json] DIR")
	}
	rep, err := lm.ReportDataset(flags.Arg(0))
	if err != nil {
		return err
	}
	if *jsonOut {
		data, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
		return nil
	}
	fmt.Printf("dataset: %s\n", rep.Dir)
	fmt.Printf("  id %s  tokenizer %s  dtype %s\n", rep.Meta.DatasetID, rep.Meta.Tokenizer, rep.Meta.DType)
	fmt.Printf("  shards %d  tokens %d  bytes %d  finalized %v\n",
		rep.Meta.ShardCount, rep.Meta.TotalTokens, rep.Bytes, rep.Meta.Finalized)
	if rep.HasProv {
		fmt.Printf("  provenance %s: %d sources\n", rep.Provenance.ManifestID, len(rep.Provenance.Sources))
		for _, s := range rep.Provenance.Sources {
			fmt.Printf("    - %s (%s) license=%s sha256=%s… bytes=%d\n",
				s.Name, s.Kind, s.License, s.SHA256[:12], s.Bytes)
		}
		st := rep.Provenance.IngestStats
		fmt.Printf("    docs in=%d kept=%d dup=%d tokens train=%d val=%d\n",
			st.DocumentsIn, st.DocumentsKept, st.DocumentsDup, st.TokensTrain, st.TokensVal)
	} else {
		fmt.Println("  provenance: none")
	}
	return nil
}

func datasetVerify(args []string) error {
	flags := flag.NewFlagSet("dataset verify", flag.ContinueOnError)
	checkpoint := flags.String("checkpoint", "", "checkpoint providing the tokenizer for vocab/spot checks")
	samples := flags.Int("samples", 100, "decode spot-check windows per shard")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: m3tal-lm dataset verify [--checkpoint CKPT] [--samples N] DIR")
	}
	var tok *lm.Tokenizer
	if *checkpoint != "" {
		model, err := lm.LoadCheckpoint(*checkpoint)
		if err != nil {
			return fmt.Errorf("load checkpoint tokenizer: %w", err)
		}
		tok = model.Tokenizer
	}
	rep, err := lm.VerifyDataset(flags.Arg(0), tok, *samples)
	if err != nil {
		return err
	}
	fmt.Printf("verify %s: %d shards OK, %d tokens", rep.Dir, rep.ShardsOK, rep.TokensOK)
	if rep.VocabChecked {
		fmt.Printf(", vocab %d", rep.VocabSize)
		if rep.BadTokens > 0 {
			fmt.Printf(", %d OUT-OF-VOCAB TOKENS", rep.BadTokens)
		} else {
			fmt.Print(", all tokens in vocab")
		}
	}
	if rep.Samples > 0 {
		fmt.Printf(", %d decode samples printable=%.4f", rep.Samples, rep.Printable)
	}
	fmt.Println()
	if rep.BadTokens > 0 {
		return fmt.Errorf("%d tokens outside vocabulary", rep.BadTokens)
	}
	return nil
}

func datasetSplit(args []string) error {
	flags := flag.NewFlagSet("dataset split", flag.ContinueOnError)
	valShards := flags.Int("val-shards", 0, "number of trailing shards to hold out")
	valOut := flags.String("val-out", "", "output validation dataset directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || *valShards <= 0 || strings.TrimSpace(*valOut) == "" {
		return fmt.Errorf("usage: m3tal-lm dataset split --val-shards K --val-out DIR SRCDIR")
	}
	res, err := lm.SplitDataset(flags.Arg(0), *valOut, *valShards)
	if err != nil {
		return err
	}
	fmt.Printf("split %s: train %d shards, val %d shards (%d tokens) -> %s\n",
		res.TrainDir, res.TrainShard, res.ValShard, res.ValTokens, res.ValDir)
	return nil
}

// datasetAnnotate writes a provenance manifest into an existing dataset
// directory without rebuilding it — used for datasets produced before the
// provenance pipeline existed.
func datasetAnnotate(args []string) error {
	flags := flag.NewFlagSet("dataset annotate", flag.ContinueOnError)
	sourcesFile := flags.String("sources", "", "JSON file listing licensed corpus sources")
	splitDesc := flags.String("split", "", "train/validation split description")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || strings.TrimSpace(*sourcesFile) == "" {
		return fmt.Errorf("usage: m3tal-lm dataset annotate --sources FILE DIR [--split DESC]")
	}
	sources, err := lm.LoadSources(*sourcesFile)
	if err != nil {
		return err
	}
	for i := range sources {
		if err := lm.HashSource(&sources[i]); err != nil {
			return fmt.Errorf("hash source %q: %w", sources[i].Name, err)
		}
	}
	d, err := lm.OpenDataset(flags.Arg(0))
	if err != nil {
		return err
	}
	stats := lm.IngestStats{TokensTrain: d.Meta.TotalTokens}
	prov := lm.NewProvenance(sources, pipelineHash(), *splitDesc, stats)
	if err := lm.WriteProvenance(flags.Arg(0), prov); err != nil {
		return err
	}
	fmt.Printf("wrote provenance %s to %s (%d sources)\n", prov.ManifestID, flags.Arg(0), len(sources))
	return nil
}

// modelInfo prints the analytic parameter count, component breakdown, and
// memory estimates for a model configuration (Stage 2 config benchmark).
func modelInfo(args []string) error {
	flags := flag.NewFlagSet("model-info", flag.ContinueOnError)
	context := flags.Int("context", 1024, "context length")
	width := flags.Int("width", 1024, "model width")
	layers := flags.Int("layers", 12, "transformer layers")
	heads := flags.Int("heads", 16, "attention heads")
	vocab := flags.Int("vocab", 512, "vocabulary size")
	if err := flags.Parse(args); err != nil {
		return err
	}
	cfg := lm.Config{Context: *context, Width: *width, Layers: *layers, Heads: *heads}
	total := cfg.ParameterCount(*vocab)
	emb, blocks, output := cfg.ParameterBreakdown(*vocab)
	fmt.Printf("config: context=%d width=%d layers=%d heads=%d vocab=%d\n",
		*context, *width, *layers, *heads, *vocab)
	fmt.Printf("parameters: %d (%.2fM)\n", total, float64(total)/1e6)
	fmt.Printf("  token embedding:  %12d (%.2fM)\n", emb, float64(emb)/1e6)
	fmt.Printf("  transformer blocks:%12d (%.2fM)\n", blocks, float64(blocks)/1e6)
	fmt.Printf("  output projection:%12d (%.2fM)\n", output, float64(output)/1e6)
	weightBytes := int64(total) * 4
	trainBytes := int64(total) * 16 // values + grad + Adam first/second (f32)
	fmt.Printf("memory (fp32):\n")
	fmt.Printf("  inference weights: %.2f GiB\n", float64(weightBytes)/(1<<30))
	fmt.Printf("  training state (weights+grad+2 Adam moments): %.2f GiB\n", float64(trainBytes)/(1<<30))
	fmt.Printf("approx FLOPs/token (fwd+bwd ~ 6N): %.2f GFLOP\n", float64(total)*6/1e9)
	return nil
}

// datasetExport decodes a dataset's token stream back to text — used to
// materialize held-out evaluation corpora from validation shards.
func datasetExport(args []string) error {
	flags := flag.NewFlagSet("dataset export", flag.ContinueOnError)
	out := flags.String("out", "", "output text file")
	checkpoint := flags.String("checkpoint", "", "checkpoint providing the tokenizer (required)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || strings.TrimSpace(*out) == "" || strings.TrimSpace(*checkpoint) == "" {
		return fmt.Errorf("usage: m3tal-lm dataset export --out FILE --checkpoint CKPT DIR")
	}
	model, err := lm.LoadCheckpoint(*checkpoint)
	if err != nil {
		return fmt.Errorf("load checkpoint tokenizer: %w", err)
	}
	ds, err := lm.OpenDataset(flags.Arg(0))
	if err != nil {
		return err
	}
	if fp := lm.TokenizerFingerprint(model.Tokenizer); fp != ds.Meta.Tokenizer {
		return fmt.Errorf("dataset tokenizer %s does not match checkpoint tokenizer %s", ds.Meta.Tokenizer, fp)
	}
	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	var total int
	for i := 0; i < ds.Meta.ShardCount; i++ {
		shard, err := ds.Shard(i)
		if err != nil {
			return fmt.Errorf("shard %d: %w", i, err)
		}
		ids := make([]int, len(shard.Tokens))
		for j, tok := range shard.Tokens {
			ids[j] = int(tok)
		}
		if _, err := w.WriteString(model.Tokenizer.Decode(ids)); err != nil {
			return err
		}
		if _, err := w.WriteString("\n"); err != nil {
			return err
		}
		total += len(shard.Tokens)
	}
	fmt.Printf("exported %d tokens from %s (%d shards) to %s\n", total, flags.Arg(0), ds.Meta.ShardCount, *out)
	return nil
}

// evalBenchReport is the machine-readable output of `eval-bench`.
type evalBenchReport struct {
	Checkpoint  string          `json:"checkpoint"`
	Backend     string          `json:"backend"`
	Corpus      string          `json:"corpus"`
	EvalLoss    float64         `json:"eval_loss"`
	Perplexity  float64         `json:"perplexity"`
	EvalTokens  int             `json:"eval_tokens"`
	Generations []evalGenResult `json:"generations"`
	CreatedUnix int64           `json:"created_unix"`
}

type evalGenResult struct {
	Prompt string `json:"prompt"`
	Output string `json:"output"`
}

var defaultBenchPrompts = []string{
	"Once upon a time",
	"A little girl named Lily",
	"The capital of France is",
	"M3TAL is",
	"def quicksort",
}

// evalBench runs a repeatable evaluation: held-out loss, fixed-seed
// generations on a fixed prompt set, and optional baseline regression
// detection (Stage 2 eval benchmark).
func evalBench(args []string) error {
	flags := flag.NewFlagSet("eval-bench", flag.ContinueOnError)
	checkpoint := flags.String("checkpoint", "", "model checkpoint path (required)")
	corpusArg := flags.String("corpus", "", "held-out evaluation corpus path(s), os.PathListSeparator-separated (required)")
	promptsFile := flags.String("prompts", "", "file with one prompt per line (# comments ok); defaults to built-in set")
	tokens := flags.Int("tokens", 60, "tokens per generation")
	temperature := flags.Float64("temperature", 0.8, "sampling temperature")
	seed := flags.Int64("seed", 19, "sampling seed")
	reportOut := flags.String("report", "", "write JSON report to this path (stdout summary always printed)")
	baseline := flags.String("baseline", "", "previous eval-bench JSON report to compare against")
	maxIncrease := flags.Float64("max-loss-increase", 0.05, "fail if eval loss exceeds baseline by more than this")
	maxEvalTokens := flags.Int("max-eval-tokens", 65536, "cap on corpus tokens evaluated (0 = all)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*checkpoint) == "" || strings.TrimSpace(*corpusArg) == "" {
		return fmt.Errorf("--checkpoint and --corpus are required")
	}
	model, err := lm.LoadCheckpoint(*checkpoint)
	if err != nil {
		return err
	}
	texts, err := lm.LoadCorpus(strings.Split(*corpusArg, string(os.PathListSeparator)))
	if err != nil {
		return err
	}
	// Bound evaluation cost: Evaluate runs one forward pass per token, so
	// cap the token count and truncate texts deterministically.
	if *maxEvalTokens > 0 {
		var kept []string
		total := 0
		for _, text := range texts {
			ids := model.Tokenizer.Encode(text)
			if total >= *maxEvalTokens {
				break
			}
			if total+len(ids) > *maxEvalTokens {
				ids = ids[:*maxEvalTokens-total]
				text = model.Tokenizer.Decode(ids)
			}
			kept = append(kept, text)
			total += len(ids)
		}
		texts = kept
	}
	evalLoss, err := model.Evaluate(texts)
	if err != nil {
		return err
	}
	evalTokens := 0
	for _, text := range texts {
		evalTokens += len(model.Tokenizer.Encode(text)) + len(model.Tokenizer.Encode("\n"))
	}

	prompts := append([]string(nil), defaultBenchPrompts...)
	if *promptsFile != "" {
		data, err := os.ReadFile(*promptsFile)
		if err != nil {
			return err
		}
		prompts = prompts[:0]
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if line != "" && !strings.HasPrefix(line, "#") {
				prompts = append(prompts, line)
			}
		}
		if len(prompts) == 0 {
			return fmt.Errorf("prompts file %q contains no prompts", *promptsFile)
		}
	}

	rep := evalBenchReport{
		Checkpoint:  *checkpoint,
		Backend:     kernel.Selected(),
		Corpus:      *corpusArg,
		EvalLoss:    evalLoss,
		Perplexity:  math.Exp(evalLoss),
		EvalTokens:  evalTokens,
		CreatedUnix: time.Now().Unix(),
	}
	for i, prompt := range prompts {
		out, err := model.GenerateWithOptions(prompt, lm.GenerateOptions{
			MaxTokens:   *tokens,
			Temperature: *temperature,
			Seed:        *seed + int64(i),
		})
		if err != nil {
			return fmt.Errorf("generate %q: %w", prompt, err)
		}
		rep.Generations = append(rep.Generations, evalGenResult{Prompt: prompt, Output: out})
	}

	fmt.Printf("eval-bench %s\n", *checkpoint)
	fmt.Printf("  backend %s  corpus %s  eval_tokens %d\n", rep.Backend, rep.Corpus, rep.EvalTokens)
	fmt.Printf("  eval loss %.4f  perplexity %.2f\n", rep.EvalLoss, rep.Perplexity)
	for _, g := range rep.Generations {
		fmt.Printf("  prompt %q -> %.80s%s\n", g.Prompt, g.Output, map[bool]string{true: "…", false: ""}[len(g.Output) > 80])
	}

	regressed := false
	if *baseline != "" {
		data, err := os.ReadFile(*baseline)
		if err != nil {
			return fmt.Errorf("read baseline: %w", err)
		}
		var base evalBenchReport
		if err := json.Unmarshal(data, &base); err != nil {
			return fmt.Errorf("decode baseline: %w", err)
		}
		diff := rep.EvalLoss - base.EvalLoss
		fmt.Printf("  baseline loss %.4f (delta %+.4f, threshold %.4f)\n", base.EvalLoss, diff, *maxIncrease)
		if diff > *maxIncrease {
			regressed = true
			fmt.Printf("  REGRESSION: eval loss increased by %.4f > %.4f\n", diff, *maxIncrease)
		} else {
			fmt.Println("  no regression vs baseline")
		}
	}

	if *reportOut != "" {
		data, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(*reportOut, append(data, '\n'), 0o644); err != nil {
			return err
		}
		fmt.Printf("  report written to %s\n", *reportOut)
	}
	if regressed {
		return fmt.Errorf("eval loss regressed vs baseline")
	}
	return nil
}
