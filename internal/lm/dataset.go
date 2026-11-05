package lm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"hash/fnv"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// This file implements the sharded token dataset format (issue: Stage 0
// dataset shards). A dataset is a directory containing a manifest
// (dataset.json) and one or more binary shard files (shard-NNNNNN.bin).
//
// Shard binary layout:
//
//	offset  size  field
//	0       8     magic "M3TDS001"
//	8       4     headerLen (uint32 LE)
//	12      H     header JSON (H = headerLen bytes)
//	12+H    4*N   token data, N little-endian uint32 values
//	...     4     crc32 (IEEE) of the token data
//
// The header records the format version, token dtype, token count, shard
// index, dataset id, creation time, and a tokenizer fingerprint so a shard
// cannot be silently paired with the wrong tokenizer.
//
// Access strategies:
//   - Sequential: shards are read in index order; windows stream from the
//     concatenation of shard token streams. Memory stays bounded to one shard.
//   - Shuffled: shard visitation order is permuted with a seeded RNG. Within
//     a shard, tokens remain sequential (window-level shuffling is the
//     trainer's responsibility). This keeps I/O sequential per shard while
//     decorrelating step order across shards.

const (
	shardMagic       = "M3TDS001"
	shardFormatVer   = 1
	datasetFormatVer = 1
	tokenDType       = "u32"
	maxShardTokens   = 64 << 20 // 256 MiB of u32 tokens per shard
)

// DatasetMeta is the dataset-level manifest stored in dataset.json.
type DatasetMeta struct {
	Format      string `json:"format"` // "m3tal.tokendataset"
	Version     int    `json:"version"`
	DType       string `json:"dtype"` // "u32"
	Tokenizer   string `json:"tokenizer"`
	ShardTokens int    `json:"shard_tokens"`
	ShardCount  int    `json:"shard_count"`
	TotalTokens int64  `json:"total_tokens"`
	CreatedUnix int64  `json:"created_unix"`
	DatasetID   string `json:"dataset_id"`
}

type shardHeader struct {
	Format      string `json:"format"` // "m3tal.tokenshard"
	Version     int    `json:"version"`
	DType       string `json:"dtype"`
	TokenCount  int    `json:"token_count"`
	ShardIndex  int    `json:"shard_index"`
	DatasetID   string `json:"dataset_id"`
	CreatedUnix int64  `json:"created_unix"`
	Tokenizer   string `json:"tokenizer"`
}

// TokenizerFingerprint returns a stable hash identifying a tokenizer's
// vocabulary and merges, used to bind datasets to the tokenizer that
// produced them.
func TokenizerFingerprint(t *Tokenizer) string {
	h := fnv.New64a()
	for _, tok := range t.Tokens {
		h.Write([]byte(tok))
		h.Write([]byte{0})
	}
	for _, m := range t.Merges {
		h.Write([]byte(m.Left))
		h.Write([]byte{0})
		h.Write([]byte(m.Right))
		h.Write([]byte{0})
		h.Write([]byte(m.Token))
		h.Write([]byte{0})
	}
	return fmt.Sprintf("fnv64a:%016x", h.Sum64())
}

func shardName(i int) string { return fmt.Sprintf("shard-%06d.bin", i) }

// TokenFeed supplies token IDs incrementally so dataset construction does
// not require the full corpus in memory. TokenStream and SliceSource both
// implement it.
type TokenFeed interface {
	Next(n int) ([]uint32, bool)
	Err() error
}

// BuildDataset consumes a TokenFeed and writes shards of shardTokens tokens
// each. At most one shard's worth of tokens is held in memory. The tokenizer
// fingerprint binds the dataset to the producing tokenizer.
func BuildDataset(dir string, feed TokenFeed, tokenizer *Tokenizer, shardTokens int) (*DatasetMeta, error) {
	if shardTokens <= 0 || shardTokens > maxShardTokens {
		return nil, fmt.Errorf("shardTokens must be in (0, %d]", maxShardTokens)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	meta := &DatasetMeta{
		Format:      "m3tal.tokendataset",
		Version:     datasetFormatVer,
		DType:       tokenDType,
		Tokenizer:   TokenizerFingerprint(tokenizer),
		ShardTokens: shardTokens,
		CreatedUnix: time.Now().Unix(),
		DatasetID:   fmt.Sprintf("ds-%d", time.Now().UnixNano()),
	}
	buf := make([]uint32, 0, shardTokens)
	flush := func() error {
		if len(buf) == 0 {
			return nil
		}
		if err := writeShard(filepath.Join(dir, shardName(meta.ShardCount)), buf, meta, meta.ShardCount); err != nil {
			return fmt.Errorf("write shard %d: %w", meta.ShardCount, err)
		}
		meta.TotalTokens += int64(len(buf))
		meta.ShardCount++
		buf = buf[:0]
		return nil
	}
	for {
		chunk, ok := feed.Next(shardTokens - len(buf))
		if !ok {
			break
		}
		buf = append(buf, chunk...)
		if len(buf) == shardTokens {
			if err := flush(); err != nil {
				return nil, err
			}
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if err := feed.Err(); err != nil {
		return nil, fmt.Errorf("token feed: %w", err)
	}
	if meta.ShardCount == 0 {
		return nil, fmt.Errorf("cannot build dataset from zero tokens")
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return nil, err
	}
	tmp := filepath.Join(dir, "dataset.json.tmp")
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return nil, err
	}
	return meta, os.Rename(tmp, filepath.Join(dir, "dataset.json"))
}

func writeShard(path string, tokens []uint32, meta *DatasetMeta, index int) error {
	header := shardHeader{
		Format:      "m3tal.tokenshard",
		Version:     shardFormatVer,
		DType:       tokenDType,
		TokenCount:  len(tokens),
		ShardIndex:  index,
		DatasetID:   meta.DatasetID,
		CreatedUnix: meta.CreatedUnix,
		Tokenizer:   meta.Tokenizer,
	}
	hj, err := json.Marshal(header)
	if err != nil {
		return err
	}
	buf := make([]byte, 0, 12+len(hj)+4*len(tokens)+4)
	buf = append(buf, shardMagic...)
	var lb [4]byte
	binary.LittleEndian.PutUint32(lb[:], uint32(len(hj)))
	buf = append(buf, lb[:]...)
	buf = append(buf, hj...)
	for _, tok := range tokens {
		binary.LittleEndian.PutUint32(lb[:], tok)
		buf = append(buf, lb[:]...)
	}
	binary.LittleEndian.PutUint32(lb[:], crc32.ChecksumIEEE(buf[12+len(hj):]))
	buf = append(buf, lb[:]...)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Dataset is an opened sharded token dataset.
type Dataset struct {
	Dir  string
	Meta DatasetMeta
}

// OpenDataset reads and validates the manifest; shard files are validated
// lazily on first read so opening stays cheap for large datasets.
func OpenDataset(dir string) (*Dataset, error) {
	data, err := os.ReadFile(filepath.Join(dir, "dataset.json"))
	if err != nil {
		return nil, fmt.Errorf("read dataset manifest: %w", err)
	}
	var meta DatasetMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("decode dataset manifest: %w", err)
	}
	if meta.Format != "m3tal.tokendataset" || meta.Version != datasetFormatVer {
		return nil, fmt.Errorf("unsupported dataset format %q version %d", meta.Format, meta.Version)
	}
	if meta.DType != tokenDType {
		return nil, fmt.Errorf("unsupported token dtype %q", meta.DType)
	}
	if meta.ShardCount <= 0 || meta.ShardCount > 1<<20 {
		return nil, fmt.Errorf("implausible shard count %d", meta.ShardCount)
	}
	return &Dataset{Dir: dir, Meta: meta}, nil
}

// Shard opens and fully validates one shard: magic, header fields,
// truncation, token count, and CRC32 integrity.
func (d *Dataset) Shard(i int) (*Shard, error) {
	if i < 0 || i >= d.Meta.ShardCount {
		return nil, fmt.Errorf("shard index %d out of range [0,%d)", i, d.Meta.ShardCount)
	}
	return readShard(filepath.Join(d.Dir, shardName(i)), d.Meta, i)
}

// ShardCount returns the number of shards.
func (d *Dataset) ShardCount() int { return d.Meta.ShardCount }

// TotalTokens returns the total token count across all shards.
func (d *Dataset) TotalTokens() int64 { return d.Meta.TotalTokens }

// Shard is a validated, memory-resident shard of token IDs.
type Shard struct {
	Index  int
	Tokens []uint32
}

func readShard(path string, meta DatasetMeta, wantIndex int) (*Shard, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read shard %q: %w", path, err)
	}
	if len(data) < 12+4 {
		return nil, fmt.Errorf("shard %q truncated: %d bytes", path, len(data))
	}
	if string(data[:8]) != shardMagic {
		return nil, fmt.Errorf("shard %q bad magic", path)
	}
	hlen := int(binary.LittleEndian.Uint32(data[8:12]))
	if hlen <= 0 || 12+hlen+4 > len(data) {
		return nil, fmt.Errorf("shard %q has invalid header length %d", path, hlen)
	}
	var h shardHeader
	if err := json.Unmarshal(data[12:12+hlen], &h); err != nil {
		return nil, fmt.Errorf("shard %q bad header: %w", path, err)
	}
	if h.Format != "m3tal.tokenshard" || h.Version != shardFormatVer {
		return nil, fmt.Errorf("shard %q unsupported format %q v%d", path, h.Format, h.Version)
	}
	if h.DType != tokenDType {
		return nil, fmt.Errorf("shard %q unsupported dtype %q", path, h.DType)
	}
	if h.ShardIndex != wantIndex {
		return nil, fmt.Errorf("shard %q index %d, expected %d", path, h.ShardIndex, wantIndex)
	}
	if h.DatasetID != meta.DatasetID {
		return nil, fmt.Errorf("shard %q belongs to dataset %q, expected %q", path, h.DatasetID, meta.DatasetID)
	}
	if h.Tokenizer != meta.Tokenizer {
		return nil, fmt.Errorf("shard %q tokenizer fingerprint %q does not match dataset %q", path, h.Tokenizer, meta.Tokenizer)
	}
	dataStart := 12 + hlen
	wantBytes := h.TokenCount * 4
	if len(data) != dataStart+wantBytes+4 {
		return nil, fmt.Errorf("shard %q size mismatch: have %d, want %d", path, len(data)-dataStart-4, wantBytes)
	}
	payload := data[dataStart : dataStart+wantBytes]
	stored := binary.LittleEndian.Uint32(data[dataStart+wantBytes:])
	if crc32.ChecksumIEEE(payload) != stored {
		return nil, fmt.Errorf("shard %q CRC32 mismatch (corrupt or truncated)", path)
	}
	tokens := make([]uint32, h.TokenCount)
	for i := range tokens {
		tokens[i] = binary.LittleEndian.Uint32(payload[i*4:])
	}
	return &Shard{Index: wantIndex, Tokens: tokens}, nil
}

// TokenOrder describes how shards are visited by a TokenStream.
type TokenOrder int

const (
	// Sequential visits shards in index order.
	Sequential TokenOrder = iota
	// ShuffledSeeded visits shards in a seeded random permutation.
	ShuffledSeeded
)

// TokenStream yields bounded windows of token IDs from a dataset without
// loading the whole dataset into memory. Only one shard is resident at a
// time.
type TokenStream struct {
	dataset *Dataset
	order   []int
	pos     int      // index into order
	buf     []uint32 // current shard tokens
	bufPos  int
	done    bool
	err     error
}

// NewTokenStream creates a stream over the dataset. order=Sequential walks
// shards in order; order=ShuffledSeeded permutes shard order with seed.
func NewTokenStream(d *Dataset, order TokenOrder, seed int64) *TokenStream {
	ts := &TokenStream{dataset: d, order: make([]int, d.Meta.ShardCount)}
	for i := range ts.order {
		ts.order[i] = i
	}
	if order == ShuffledSeeded {
		rng := rand.New(rand.NewSource(seed))
		rng.Shuffle(len(ts.order), func(i, j int) { ts.order[i], ts.order[j] = ts.order[j], ts.order[i] })
	}
	return ts
}

// Next returns the next batch of up to n token IDs, or io.EOF-style nil,done.
// It is allocation-light: returned slices are only valid until the next call.
func (ts *TokenStream) Next(n int) ([]uint32, bool) {
	if ts.done {
		return nil, false
	}
	out := make([]uint32, 0, n)
	for len(out) < n {
		if ts.buf == nil || ts.bufPos >= len(ts.buf) {
			if ts.pos >= len(ts.order) {
				ts.done = true
				break
			}
			shard, err := ts.dataset.Shard(ts.order[ts.pos])
			ts.pos++
			if err != nil {
				// Surface corruption as end-of-stream with an error flag.
				ts.err = err
				ts.done = true
				break
			}
			ts.buf = shard.Tokens
			ts.bufPos = 0
		}
		take := n - len(out)
		if rem := len(ts.buf) - ts.bufPos; take > rem {
			take = rem
		}
		out = append(out, ts.buf[ts.bufPos:ts.bufPos+take]...)
		ts.bufPos += take
	}
	return out, len(out) > 0
}

// Err returns the first shard read error encountered, if any.
func (ts *TokenStream) Err() error { return ts.err }

// Window returns up to context+1 consecutive tokens for one training window,
// or nil,false at end of stream.
func (ts *TokenStream) Window(context int) ([]uint32, bool) {
	need := context + 1
	buf := make([]uint32, 0, need)
	for len(buf) < need {
		chunk, ok := ts.Next(need - len(buf))
		if !ok {
			break
		}
		buf = append(buf, chunk...)
	}
	if len(buf) < 2 {
		return nil, false
	}
	return buf, true
}

// EncodeCorpusTokenizes is a helper: encode texts with the tokenizer and
// flatten to a uint32 stream for BuildDataset.
func EncodeCorpusTokenizes(tokenizer *Tokenizer, texts []string) []uint32 {
	var out []uint32
	for _, text := range texts {
		for _, id := range tokenizer.Encode(text) {
			out = append(out, uint32(id))
		}
		for _, id := range tokenizer.Encode("\n") {
			out = append(out, uint32(id))
		}
	}
	return out
}

// ListShardFiles returns sorted shard file names for diagnostics.
func ListShardFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "shard-") && strings.HasSuffix(e.Name(), ".bin") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names, nil
}
