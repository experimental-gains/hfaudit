package main

import (
	"encoding/json"
	"strings"
)

// ipynbCell is the subset of a Jupyter notebook cell's JSON this tool reads:
// its type (only "code" cells contain Python that's ever executed —
// markdown/raw cells are prose or output, never calls to audit) and its
// source. nbformat allows a cell's "source" field to be encoded as either a
// single string or a list of per-line strings; json.RawMessage defers
// decoding so decodeCellSource can try both shapes.
type ipynbCell struct {
	CellType string          `json:"cell_type"`
	Source   json.RawMessage `json:"source"`
}

type ipynbDoc struct {
	Cells []ipynbCell `json:"cells"`
}

// decodeCellSource decodes a cell's "source" field into plain text, handling
// both shapes nbformat allows: a single string, or a list of strings (one
// per source line, each normally ending in its own embedded "\n" except
// possibly the last) that real notebook-writing tools join back into one
// string by plain concatenation, not by inserting an extra separator.
func decodeCellSource(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var lines []string
	if json.Unmarshal(raw, &lines) == nil {
		return strings.Join(lines, "")
	}
	return ""
}

// decodeNotebookSource extracts a Jupyter notebook's code cells (skipping
// markdown/raw cells, which are prose or output, never executed) and joins
// their decoded source into one plain-Python text blob, in cell order,
// separated by a blank line between cells.
//
// hfaudit used to run extraction directly against a .ipynb file's own raw
// JSON bytes, on the theory that a bare "#" in that JSON text doesn't
// delimit a real Python comment and docstrings aren't a distinct syntactic
// form there either, so extractReferences skipped stripDeadPythonText for
// notebooks entirely (see its doc comment, before this fix). That reasoning
// held for comments, but missed something bigger: nbformat stores each code
// cell's source as a JSON string, so every double-quote in the real Python
// source it contains — the exact character idPattern's own matching (and
// repoIDArgPattern's, and leadingPositionalIDPattern's) requires immediately
// surrounding a Hub ID — is escaped to \" in that raw JSON text. A quote
// preceded by a backslash never satisfies a `["']` match expecting the quote
// at that exact position, so none of extractReferences' patterns could ever
// match a double-quoted ID inside a notebook. Confirmed live against real
// Hugging Face example notebooks (setfit's own
// notebooks/text-classification.ipynb, line 528:
// `"model = SetFitModel.from_pretrained(\"lewtun/my-awesome-setfit-model\")\n"`)
// — hfaudit reported "no Hugging Face model/dataset references found" for
// every .ipynb file in that notebooks/ directory, despite dozens of real
// from_pretrained/load_dataset calls, because virtually all of them use
// double-quoted string literals (the standard, black-formatted style) — the
// one case this tool could never see. A notebook written with single quotes
// wouldn't hit this, since JSON doesn't require escaping those, which is why
// the gap went unnoticed for this long: whether it strikes depends on which
// quote character the source happens to use, not on anything unusual about
// the notebook.
//
// Decoding each cell's source through encoding/json first — the same way
// the real Jupyter/nbformat tooling that actually executes this code does —
// undoes that escaping before extraction ever sees the text. It also makes
// stripDeadPythonText's comment/docstring stripping correctly applicable to
// notebook source, since extractReferences now receives real decoded Python
// text instead of raw JSON; the old blanket skip for ".ipynb" no longer
// applies (see extractReferences).
//
// Returns ok=false if data isn't parseable as notebook JSON at all, so the
// caller can fall back to treating it as plain text rather than silently
// losing every reference in a malformed or non-standard file.
func decodeNotebookSource(data []byte) (text string, ok bool) {
	var doc ipynbDoc
	if err := json.Unmarshal(data, &doc); err != nil {
		return "", false
	}
	var b strings.Builder
	for _, c := range doc.Cells {
		if c.CellType != "code" {
			continue
		}
		b.WriteString(decodeCellSource(c.Source))
		b.WriteString("\n\n")
	}
	return b.String(), true
}
