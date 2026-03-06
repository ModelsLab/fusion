package kb

import (
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	_ "modernc.org/sqlite"
)

type sqliteSearchIndex struct {
	db *sql.DB
}

func newSQLiteSearchIndexFromFS(files fs.FS) (searchEngine, error) {
	data, err := fs.ReadFile(files, "knowledge.db")
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("read knowledge.db: %w", err)
	}

	path, err := materializeKnowledgeDB(data)
	if err != nil {
		return nil, err
	}

	return newSQLiteSearchIndexFromPath(path)
}

func newSQLiteSearchIndexFromPath(path string) (*sqliteSearchIndex, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("sqlite knowledge path is required")
	}
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	if err != nil {
		return nil, fmt.Errorf("open knowledge.db: %w", err)
	}
	db.SetMaxOpenConns(1)

	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("ping knowledge.db: %w", err)
	}

	return &sqliteSearchIndex{db: db}, nil
}

func LoadFromSQLitePath(path string) (*Store, error) {
	index, err := newSQLiteSearchIndexFromPath(path)
	if err != nil {
		return nil, err
	}
	store, err := loadStoreFromSQLite(index.db)
	if err != nil {
		_ = index.db.Close()
		return nil, err
	}
	store.search = index
	return store, nil
}

func loadStoreFromSQLite(db *sql.DB) (*Store, error) {
	rows, err := db.Query(`SELECT kind, json FROM knowledge_objects ORDER BY kind, id`)
	if err != nil {
		return nil, fmt.Errorf("query knowledge objects: %w", err)
	}
	defer rows.Close()

	store := &Store{
		Sources:    []Source{},
		GPUs:       []GPUProfile{},
		Strategies: []Strategy{},
		Skills:     []Skill{},
		Examples:   []Example{},
		Documents:  []Document{},
	}

	for rows.Next() {
		var (
			kind    string
			payload string
		)
		if err := rows.Scan(&kind, &payload); err != nil {
			return nil, fmt.Errorf("scan knowledge object: %w", err)
		}
		switch normalizeKind(kind) {
		case "source":
			var source Source
			if err := json.Unmarshal([]byte(payload), &source); err != nil {
				return nil, fmt.Errorf("decode source payload: %w", err)
			}
			store.Sources = append(store.Sources, source)
		case "gpu":
			var gpu GPUProfile
			if err := json.Unmarshal([]byte(payload), &gpu); err != nil {
				return nil, fmt.Errorf("decode gpu payload: %w", err)
			}
			store.GPUs = append(store.GPUs, gpu)
		case "strategy":
			var strategy Strategy
			if err := json.Unmarshal([]byte(payload), &strategy); err != nil {
				return nil, fmt.Errorf("decode strategy payload: %w", err)
			}
			store.Strategies = append(store.Strategies, strategy)
		case "skill":
			var skill Skill
			if err := json.Unmarshal([]byte(payload), &skill); err != nil {
				return nil, fmt.Errorf("decode skill payload: %w", err)
			}
			store.Skills = append(store.Skills, skill)
		case "example":
			var example Example
			if err := json.Unmarshal([]byte(payload), &example); err != nil {
				return nil, fmt.Errorf("decode example payload: %w", err)
			}
			store.Examples = append(store.Examples, example)
		case "document":
			var document Document
			if err := json.Unmarshal([]byte(payload), &document); err != nil {
				return nil, fmt.Errorf("decode document payload: %w", err)
			}
			store.Documents = append(store.Documents, document)
		default:
			return nil, fmt.Errorf("unsupported knowledge kind %q in sqlite store", kind)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate knowledge objects: %w", err)
	}

	return store, nil
}

func (s *sqliteSearchIndex) Search(query, kind string, limit int) []SearchHit {
	kind = normalizeKind(kind)
	if kind == "" {
		kind = "all"
	}
	if limit <= 0 {
		limit = 10
	}

	match := ftsQuery(query)
	if match == "" {
		return nil
	}

	rows, err := s.db.Query(`
SELECT
	o.kind,
	o.id,
	o.title,
	o.summary,
	bm25(knowledge_fts, 10.0, 5.0, 1.0) AS rank,
	o.reliability,
	o.review_status
FROM knowledge_fts
JOIN knowledge_objects AS o
	ON o.kind = knowledge_fts.kind
	AND o.id = knowledge_fts.id
WHERE knowledge_fts MATCH ?
	AND (? = 'all' OR o.kind = ?)
ORDER BY
	rank ASC,
	CASE o.reliability
		WHEN 'official' THEN 0
		WHEN 'paper' THEN 1
		WHEN 'repo' THEN 2
		WHEN 'article' THEN 3
		WHEN 'curated' THEN 4
		ELSE 5
	END ASC,
	CASE o.review_status
		WHEN 'reviewed' THEN 0
		WHEN 'queued' THEN 1
		ELSE 2
	END ASC,
	o.title ASC
LIMIT ?`, match, kind, kind, limit)
	if err != nil {
		return nil
	}
	defer rows.Close()

	hits := []SearchHit{}
	for rows.Next() {
		var (
			hit          SearchHit
			rank         float64
			reliability  string
			reviewStatus string
		)
		if err := rows.Scan(&hit.Kind, &hit.ID, &hit.Title, &hit.Summary, &rank, &reliability, &reviewStatus); err != nil {
			return hits
		}
		hit.Score = searchScore(rank, reliability, reviewStatus)
		hits = append(hits, hit)
	}

	return hits
}

func WriteSQLiteIndex(store *Store, path string) error {
	records, err := store.IndexRecords()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create knowledge db dir: %w", err)
	}

	tempPath := path + ".tmp"
	_ = os.Remove(tempPath)
	_ = os.Remove(path)

	db, err := sql.Open("sqlite", tempPath)
	if err != nil {
		return fmt.Errorf("open sqlite index: %w", err)
	}
	defer db.Close()

	schema := []string{
		`PRAGMA journal_mode=DELETE;`,
		`CREATE TABLE knowledge_objects (
			kind TEXT NOT NULL,
			id TEXT NOT NULL,
			title TEXT NOT NULL,
			summary TEXT NOT NULL,
			category TEXT NOT NULL,
			support_level TEXT NOT NULL,
			reliability TEXT NOT NULL,
			review_status TEXT NOT NULL,
			json TEXT NOT NULL,
			PRIMARY KEY(kind, id)
		);`,
		`CREATE INDEX knowledge_objects_kind_idx ON knowledge_objects(kind);`,
		`CREATE VIRTUAL TABLE knowledge_fts USING fts5(
			kind UNINDEXED,
			id UNINDEXED,
			title,
			summary,
			body,
			tokenize = 'unicode61 remove_diacritics 2'
		);`,
	}
	for _, stmt := range schema {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("init sqlite schema: %w", err)
		}
	}

	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("begin sqlite tx: %w", err)
	}

	objectStmt, err := tx.Prepare(`INSERT INTO knowledge_objects(kind, id, title, summary, category, support_level, reliability, review_status, json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("prepare object insert: %w", err)
	}
	defer objectStmt.Close()

	ftsStmt, err := tx.Prepare(`INSERT INTO knowledge_fts(kind, id, title, summary, body) VALUES(?, ?, ?, ?, ?)`)
	if err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("prepare fts insert: %w", err)
	}
	defer ftsStmt.Close()

	for _, record := range records {
		if _, err := objectStmt.Exec(record.Kind, record.ID, record.Title, record.Summary, record.Category, record.SupportLevel, record.Reliability, record.ReviewStatus, record.JSON); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("insert object %s/%s: %w", record.Kind, record.ID, err)
		}
		if _, err := ftsStmt.Exec(record.Kind, record.ID, record.Title, record.Summary, record.Body); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("insert fts %s/%s: %w", record.Kind, record.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit sqlite index: %w", err)
	}
	if err := db.Close(); err != nil {
		return fmt.Errorf("close sqlite index: %w", err)
	}

	if err := os.Rename(tempPath, path); err != nil {
		return fmt.Errorf("finalize sqlite index: %w", err)
	}
	return nil
}

func materializeKnowledgeDB(data []byte) (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("resolve user cache dir: %w", err)
	}

	sum := sha256.Sum256(data)
	path := filepath.Join(cacheDir, "fusion", "knowledge", fmt.Sprintf("knowledge-%x.db", sum[:8]))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("create knowledge cache dir: %w", err)
	}

	if existing, err := os.ReadFile(path); err == nil {
		if len(existing) == len(data) {
			return path, nil
		}
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", fmt.Errorf("write cached knowledge db: %w", err)
	}
	return path, nil
}

func ftsQuery(query string) string {
	tokens := strings.FieldsFunc(strings.ToLower(strings.TrimSpace(query)), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	out := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if len(token) == 0 {
			continue
		}
		out = append(out, token+"*")
	}
	return strings.Join(out, " ")
}

func searchScore(rank float64, reliability, reviewStatus string) int {
	score := 0
	switch {
	case rank < 0:
		score = int(math.Round(math.Abs(rank) * 1000))
	case rank == 0:
		score = 100
	default:
		score = int(math.Round(1000.0 / (rank + 1)))
	}

	switch canonical(reliability) {
	case "official":
		score += 30
	case "paper":
		score += 24
	case "repo":
		score += 18
	case "article":
		score += 10
	case "curated":
		score += 8
	}
	if canonical(reviewStatus) == "reviewed" {
		score += 8
	}
	if score < 1 {
		return 1
	}
	return score
}
