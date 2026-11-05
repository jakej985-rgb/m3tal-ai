package lm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unicode"
)

// This file implements dataset inspection, splitting, and verification
// operations used by `m3tal-lm dataset report|verify|split` (Stage 2).

// ShardInfo describes one shard without loading its tokens.
type ShardInfo struct {
	Index      int    `json:"index"`
	Path       string `json:"path"`
	Bytes      int64  `json:"bytes"`
	TokenCount int    `json:"token_count"`
	DatasetID  string `json:"dataset_id"`
	Tokenizer  string `json:"tokenizer"`
}

// shardInfo reads only the shard header and file size.
func shardInfo(path string, wantIndex int) (*ShardInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var fixed [12]byte
	if _, err := io.ReadFull(f, fixed[:]); err != nil {
		return nil, fmt.Errorf("shard %q truncated header: %w", path, err)
	}
	if string(fixed[:8]) != shardMagic {
		return nil, fmt.Errorf("shard %q bad magic", path)
	}
	hlen := int(binary.LittleEndian.Uint32(fixed[8:12]))
	hbuf := make([]byte, hlen)
	if _, err := io.ReadFull(f, hbuf); err != nil {
		return nil, fmt.Errorf("shard %q truncated header JSON: %w", path, err)
	}
	var h shardHeader
	if err := json.Unmarshal(hbuf, &h); err != nil {
		return nil, fmt.Errorf("shard %q bad header: %w", path, err)
	}
	if h.ShardIndex != wantIndex {
		return nil, fmt.Errorf("shard %q index %d, expected %d", path, h.ShardIndex, wantIndex)
	}
	return &ShardInfo{
		Index:      h.ShardIndex,
		Path:       path,
		Bytes:      st.Size(),
		TokenCount: h.TokenCount,
		DatasetID:  h.DatasetID,
		Tokenizer:  h.Tokenizer,
	}, nil
}

// DatasetReport is the machine-readable output of `dataset report`.
type DatasetReport struct {
	Dir        string              `json:"dir"`
	Meta       DatasetMeta         `json:"meta"`
	Shards     []ShardInfo         `json:"shards"`
	Bytes      int64               `json:"bytes"`
	HasProv    bool                `json:"has_provenance"`
	Provenance *ProvenanceManifest `json:"provenance,omitempty"`
}

// ReportDataset validates the manifest and reads every shard header.
func ReportDataset(dir string) (*DatasetReport, error) {
	d, err := OpenDataset(dir)
	if err != nil {
		return nil, err
	}
	rep := &DatasetReport{Dir: dir, Meta: d.Meta}
	for i := 0; i < d.Meta.ShardCount; i++ {
		info, err := shardInfo(filepath.Join(dir, shardName(i)), i)
		if err != nil {
			return nil, err
		}
		rep.Shards = append(rep.Shards, *info)
		rep.Bytes += info.Bytes
	}
	if pm, err := ReadProvenance(dir); err == nil {
		rep.HasProv = true
		rep.Provenance = pm
	}
	return rep, nil
}

// SplitResult describes a train/validation shard split.
type SplitResult struct {
	TrainDir   string `json:"train_dir"`
	ValDir     string `json:"val_dir"`
	TrainShard int    `json:"train_shards"`
	ValShard   int    `json:"val_shards"`
	ValTokens  int64  `json:"val_tokens"`
}

// SplitDataset moves the last valShards shards of srcDir into valDir,
// producing a held-out validation dataset. Shard headers embed their
// index, so each moved shard is rewritten under its new index; the dataset
// ID and tokenizer fingerprint are preserved so both halves validate
// against their manifests. The train manifest is rewritten with reduced
// counts and the val manifest is written finalized.
//
// The split is at shard granularity: neighbouring shards may contain
// fragments of the same source document. For document-level guarantees on
// text corpora, split before tokenization instead.
func SplitDataset(srcDir, valDir string, valShards int) (*SplitResult, error) {
	d, err := OpenDataset(srcDir)
	if err != nil {
		return nil, err
	}
	n := d.Meta.ShardCount
	if valShards <= 0 || valShards >= n {
		return nil, fmt.Errorf("val_shards %d out of range; dataset has %d shards and must keep at least one for training", valShards, n)
	}
	if err := os.MkdirAll(valDir, 0o755); err != nil {
		return nil, err
	}
	firstVal := n - valShards
	valMeta := d.Meta
	valMeta.ShardCount = valShards
	valMeta.TotalTokens = 0
	valMeta.Finalized = true
	valMeta.DatasetID = d.Meta.DatasetID

	var valTokens int64
	for i := firstVal; i < n; i++ {
		shard, err := d.Shard(i)
		if err != nil {
			return nil, err
		}
		newIndex := i - firstVal
		if err := writeShard(filepath.Join(valDir, shardName(newIndex)), shard.Tokens, &valMeta, newIndex); err != nil {
			return nil, fmt.Errorf("write val shard %d: %w", newIndex, err)
		}
		valTokens += int64(len(shard.Tokens))
		if err := os.Remove(filepath.Join(srcDir, shardName(i))); err != nil {
			return nil, fmt.Errorf("remove moved shard %d: %w", i, err)
		}
	}
	valMeta.TotalTokens = valTokens
	valData, err := json.MarshalIndent(valMeta, "", "  ")
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(valDir, "dataset.json"), valData, 0o644); err != nil {
		return nil, err
	}
	// Carry provenance over so the val half stays attributable.
	if _, err := os.Stat(ProvenancePath(srcDir)); err == nil {
		data, err := os.ReadFile(ProvenancePath(srcDir))
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(ProvenancePath(valDir), data, 0o644); err != nil {
			return nil, err
		}
	}
	// Rewrite the train manifest with reduced counts.
	trainMeta := d.Meta
	trainMeta.ShardCount = firstVal
	trainMeta.TotalTokens -= valTokens
	trainData, err := json.MarshalIndent(trainMeta, "", "  ")
	if err != nil {
		return nil, err
	}
	tmp := filepath.Join(srcDir, "dataset.json.tmp")
	if err := os.WriteFile(tmp, trainData, 0o644); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, filepath.Join(srcDir, "dataset.json")); err != nil {
		return nil, err
	}
	return &SplitResult{
		TrainDir:   srcDir,
		ValDir:     valDir,
		TrainShard: firstVal,
		ValShard:   valShards,
		ValTokens:  valTokens,
	}, nil
}

// VerifyReport is the machine-readable output of `dataset verify`.
type VerifyReport struct {
	Dir          string  `json:"dir"`
	ShardsOK     int     `json:"shards_ok"`
	TokensOK     int64   `json:"tokens_ok"`
	VocabChecked bool    `json:"vocab_checked"`
	VocabSize    int     `json:"vocab_size"`
	BadTokens    int64   `json:"out_of_vocab_tokens"`
	Samples      int     `json:"samples_decoded"`
	Printable    float64 `json:"sample_printable_ratio"`
	ElapsedMS    int64   `json:"elapsed_ms"`
}

// VerifyDataset fully validates every shard (magic, header, CRC32,
// truncation) and, when tokenizer is non-nil, checks that every token ID
// is within vocabulary range. It then decodes up to sampleWindows
// evenly-spaced windows and reports the fraction of printable runes, a
// spot check for encoding corruption.
func VerifyDataset(dir string, tokenizer *Tokenizer, sampleWindows int) (*VerifyReport, error) {
	rep := &VerifyReport{Dir: dir}
	d, err := OpenDataset(dir)
	if err != nil {
		return nil, err
	}
	var vocab int
	if tokenizer != nil {
		vocab = len(tokenizer.Tokens)
		rep.VocabChecked = true
		rep.VocabSize = vocab
	}
	var printable, totalRunes int
	for i := 0; i < d.Meta.ShardCount; i++ {
		shard, err := d.Shard(i)
		if err != nil {
			return nil, fmt.Errorf("shard %d: %w", i, err)
		}
		rep.ShardsOK++
		rep.TokensOK += int64(len(shard.Tokens))
		if vocab > 0 {
			for _, tok := range shard.Tokens {
				if int(tok) >= vocab {
					rep.BadTokens++
				}
			}
		}
		// Spot-check decode: sample windows spread across the shard.
		if tokenizer != nil && sampleWindows > 0 && len(shard.Tokens) >= 16 {
			step := len(shard.Tokens) / (sampleWindows + 1)
			if step < 1 {
				step = 1
			}
			for start := 0; start+16 <= len(shard.Tokens) && rep.Samples < sampleWindows; start += step {
				ids := make([]int, 16)
				for j := range ids {
					ids[j] = int(shard.Tokens[start+j])
				}
				text := tokenizer.Decode(ids)
				for _, r := range text {
					totalRunes++
					if unicode.IsPrint(r) || unicode.IsSpace(r) {
						printable++
					}
				}
				rep.Samples++
			}
		}
	}
	if totalRunes > 0 {
		rep.Printable = float64(printable) / float64(totalRunes)
	}
	return rep, nil
}

// DedupReport summarizes document-level dedup for the ingest report.
func DedupReport(kept, dup int64) string {
	if kept+dup == 0 {
		return "no documents"
	}
	return fmt.Sprintf("%d kept, %d duplicates removed (%.2f%% dup rate)", kept, dup, 100*float64(dup)/float64(kept+dup))
}
