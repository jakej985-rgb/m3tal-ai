package lm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// This file implements corpus provenance manifests (issue: Stage 2 corpus
// ingestion). Every dataset built by `m3tal-lm dataset build` records where
// its text came from, under which license, which filters and dedup were
// applied, and a hash of the raw inputs so the ingestion is reproducible.
//
// A source is never treated as reusable merely because it is publicly
// accessible: CorpusSource.License and CorpusSource.LicenseReview are
// required, and ingestion refuses sources that omit them.

const provenanceFormat = "m3tal.provenance"
const provenanceVersion = 1

// CorpusSource describes one licensed input to the corpus pipeline.
type CorpusSource struct {
	// Name is a short identifier used in reports (e.g. "tinystories").
	Name string `json:"name"`
	// Kind is "text" (raw corpus files tokenized at build time) or
	// "tokendataset" (an existing m3tal.tokendataset directory consumed
	// as-is, e.g. a previously built shard set).
	Kind string `json:"kind"`
	// Paths are the local files or directories that make up the source.
	// For Kind "tokendataset" this is the dataset directory.
	Paths []string `json:"paths"`
	// License is the SPDX identifier or an explicit terms description
	// (e.g. "CDLA-Sharing-1.0", "self-owned"). Required.
	License string `json:"license"`
	// LicenseReview records the terms review: who checked, when, and what
	// was concluded. Required — ingestion fails without it.
	LicenseReview string `json:"license_review"`
	// URL optionally records where the data was obtained, for reference.
	URL string `json:"url,omitempty"`
	// RetrievedUnix optionally records when the data was obtained.
	RetrievedUnix int64 `json:"retrieved_unix,omitempty"`
	// Version optionally records the source revision/date.
	Version string `json:"version,omitempty"`
	// Filters lists corpus-level filters applied before tokenization
	// (e.g. "utf8-valid", "min-chars:64", "sensitive-file-skip").
	Filters []string `json:"filters,omitempty"`
	// Dedup records the deduplication strategy ("none", "exact-doc",
	// "url+simhash", ...).
	Dedup string `json:"dedup"`
	// SHA256 is the hex digest over the raw source bytes (text sources:
	// concatenated file bytes in sorted-path order; tokendataset sources:
	// concatenated shard file bytes). Filled in by HashSource.
	SHA256 string `json:"sha256,omitempty"`
	// Bytes is the total raw byte size hashed.
	Bytes int64 `json:"bytes,omitempty"`
}

// ProvenanceManifest is written to provenance.json next to dataset.json.
type ProvenanceManifest struct {
	Format        string         `json:"format"`
	Version       int            `json:"version"`
	Pipeline      string         `json:"pipeline"`      // e.g. "m3tal-lm dataset build"
	PipelineHash  string         `json:"pipeline_hash"` // git revision or "unknown"
	CreatedUnix   int64          `json:"created_unix"`
	ManifestID    string         `json:"manifest_id"`
	Sources       []CorpusSource `json:"sources"`
	IngestStats   IngestStats    `json:"ingest_stats"`
	TrainValSplit string         `json:"train_val_split,omitempty"`
}

// IngestStats records what the pipeline actually did, for the dataset
// report required by the Stage 2 acceptance criteria.
type IngestStats struct {
	DocumentsIn   int64 `json:"documents_in"`
	DocumentsKept int64 `json:"documents_kept"`
	DocumentsDup  int64 `json:"documents_duplicate"`
	TokensTrain   int64 `json:"tokens_train"`
	TokensVal     int64 `json:"tokens_val"`
}

// Validate enforces the provenance rules: every source must name a
// license and record a license review before ingestion.
func (s *CorpusSource) Validate() error {
	if strings.TrimSpace(s.Name) == "" {
		return fmt.Errorf("corpus source is missing a name")
	}
	if len(s.Paths) == 0 {
		return fmt.Errorf("corpus source %q has no paths", s.Name)
	}
	if strings.TrimSpace(s.License) == "" {
		return fmt.Errorf("corpus source %q has no license; a source is not reusable merely because it is publicly accessible", s.Name)
	}
	if strings.TrimSpace(s.LicenseReview) == "" {
		return fmt.Errorf("corpus source %q has no license_review; record the terms review before ingestion", s.Name)
	}
	switch s.Kind {
	case "", "text":
		s.Kind = "text"
	case "tokendataset":
	default:
		return fmt.Errorf("corpus source %q has unsupported kind %q", s.Name, s.Kind)
	}
	if s.Dedup == "" {
		s.Dedup = "none"
	}
	return nil
}

// LoadSources reads a JSON file containing a list of CorpusSource specs.
func LoadSources(path string) ([]CorpusSource, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read sources file %q: %w", path, err)
	}
	var sources []CorpusSource
	if err := json.Unmarshal(data, &sources); err != nil {
		return nil, fmt.Errorf("decode sources file %q: %w", path, err)
	}
	for i := range sources {
		if err := sources[i].Validate(); err != nil {
			return nil, fmt.Errorf("sources[%d]: %w", i, err)
		}
	}
	return sources, nil
}

// HashSource computes a SHA-256 over the source's raw bytes. For text
// sources it hashes the corpus-eligible files (same rules as LoadCorpus:
// .md/.txt, size caps, sensitive-file skip) in sorted-path order; for
// tokendataset sources it hashes the shard files plus dataset.json.
func HashSource(s *CorpusSource) error {
	var files []string
	var total int64
	for _, root := range s.Paths {
		info, err := os.Lstat(root)
		if err != nil {
			return fmt.Errorf("source %q stat %q: %w", s.Name, root, err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("source %q refuses symlink path %q", s.Name, root)
		}
		if s.Kind == "tokendataset" {
			matched, err := filepath.Glob(filepath.Join(root, "*"))
			if err != nil {
				return err
			}
			for _, f := range matched {
				if fi, err := os.Stat(f); err == nil && !fi.IsDir() {
					files = append(files, f)
				}
			}
			continue
		}
		if !info.IsDir() {
			if corpusTextFile(root) && !sensitiveCorpusFile(root) {
				files = append(files, root)
			}
			continue
		}
		err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if path != root && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "node_modules" || entry.Name() == "vendor" || entry.Name() == "build" || entry.Name() == "dist") {
					return filepath.SkipDir
				}
				return nil
			}
			if entry.Type()&os.ModeSymlink != 0 || !corpusTextFile(path) || sensitiveCorpusFile(path) {
				return nil
			}
			files = append(files, path)
			return nil
		})
		if err != nil {
			return fmt.Errorf("source %q walk %q: %w", s.Name, root, err)
		}
	}
	if len(files) == 0 {
		return fmt.Errorf("source %q has no hashable files under %v", s.Name, s.Paths)
	}
	sort.Strings(files)
	h := sha256.New()
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return fmt.Errorf("source %q read %q: %w", s.Name, f, err)
		}
		io.WriteString(h, filepath.Base(f))
		h.Write([]byte{0})
		h.Write(data)
		h.Write([]byte{0})
		total += int64(len(data))
	}
	s.SHA256 = hex.EncodeToString(h.Sum(nil))
	s.Bytes = total
	return nil
}

// ProvenancePath returns the manifest path inside a dataset directory.
func ProvenancePath(dir string) string { return filepath.Join(dir, "provenance.json") }

// WriteProvenance atomically writes the manifest into dir.
func WriteProvenance(dir string, m *ProvenanceManifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := ProvenancePath(dir) + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, ProvenancePath(dir))
}

// ReadProvenance loads and validates a provenance manifest.
func ReadProvenance(dir string) (*ProvenanceManifest, error) {
	data, err := os.ReadFile(ProvenancePath(dir))
	if err != nil {
		return nil, fmt.Errorf("read provenance manifest: %w", err)
	}
	var m ProvenanceManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("decode provenance manifest: %w", err)
	}
	if m.Format != provenanceFormat || m.Version != provenanceVersion {
		return nil, fmt.Errorf("unsupported provenance format %q version %d", m.Format, m.Version)
	}
	if len(m.Sources) == 0 {
		return nil, fmt.Errorf("provenance manifest has no sources")
	}
	return &m, nil
}

// NewProvenance builds a manifest for an ingestion run. pipelineHash
// should identify the code revision (git sha) when available.
func NewProvenance(sources []CorpusSource, pipelineHash, splitDesc string, stats IngestStats) *ProvenanceManifest {
	return &ProvenanceManifest{
		Format:        provenanceFormat,
		Version:       provenanceVersion,
		Pipeline:      "m3tal-lm dataset build",
		PipelineHash:  pipelineHash,
		CreatedUnix:   time.Now().Unix(),
		ManifestID:    fmt.Sprintf("prov-%d", time.Now().UnixNano()),
		Sources:       sources,
		IngestStats:   stats,
		TrainValSplit: splitDesc,
	}
}
