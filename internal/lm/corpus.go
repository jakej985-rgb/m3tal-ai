package lm

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"
)

const maxCorpusFile = 4 << 20
const maxCorpusTotal = 32 << 20

// LoadCorpus reads UTF-8 text from explicit files or directories. It accepts
// Markdown and plain text, while skipping hidden and generated directories.
// Dataset licensing remains the caller's
// responsibility and should be recorded alongside the corpus.
func LoadCorpus(paths []string) ([]string, error) {
	var texts []string
	total := 0
	for _, path := range paths {
		info, err := os.Lstat(path)
		if err != nil {
			return nil, fmt.Errorf("stat corpus path %q: %w", path, err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("refusing symlink corpus path %q", path)
		}
		if !info.IsDir() {
			if sensitiveCorpusFile(path) {
				continue
			}
			if err := addCorpusFile(path, &texts, &total); err != nil {
				return nil, err
			}
			continue
		}
		err = filepath.WalkDir(path, func(file string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() && file != path && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "node_modules" || entry.Name() == "vendor" || entry.Name() == "build" || entry.Name() == "dist") {
				return filepath.SkipDir
			}
			if entry.Type()&os.ModeSymlink != 0 {
				return nil
			}
			if entry.IsDir() || !corpusTextFile(file) {
				return nil
			}
			if sensitiveCorpusFile(file) {
				return nil
			}
			return addCorpusFile(file, &texts, &total)
		})
		if err != nil {
			return nil, fmt.Errorf("walk corpus path %q: %w", path, err)
		}
	}
	if len(texts) == 0 {
		return nil, fmt.Errorf("no supported UTF-8 text found in corpus paths")
	}
	return texts, nil
}

func addCorpusFile(path string, texts *[]string, total *int) error {
	if !corpusTextFile(path) {
		return nil
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil
	}
	if info.Size() > maxCorpusFile || *total+int(info.Size()) > maxCorpusTotal {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read corpus file %q: %w", path, err)
	}
	if !utf8.Valid(data) || strings.IndexByte(string(data), 0) >= 0 {
		return nil
	}
	text := strings.TrimSpace(string(data))
	if text == "" {
		return nil
	}
	*texts = append(*texts, text)
	*total += len(data)
	return nil
}

func corpusTextFile(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md", ".txt":
		return true
	default:
		return false
	}
}

func sensitiveCorpusFile(path string) bool {
	name := strings.ToLower(filepath.Base(path))
	if name == ".env" || strings.HasPrefix(name, ".env.") || strings.Contains(name, "secret") || strings.Contains(name, "credential") || strings.Contains(name, "private") || name == "id_rsa" || name == "id_ed25519" {
		return true
	}
	switch strings.ToLower(filepath.Ext(name)) {
	case ".key", ".pem", ".p12", ".pfx", ".keystore", ".crt":
		return true
	default:
		return false
	}
}
