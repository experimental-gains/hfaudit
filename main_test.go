package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeHFClient struct {
	results map[string]checkResult
}

func (f *fakeHFClient) check(id string, kind repoKind) checkResult {
	if r, ok := f.results[id]; ok {
		return r
	}
	return checkResult{status: "error"}
}

func TestRunExplicitIDs(t *testing.T) {
	client := &fakeHFClient{results: map[string]checkResult{
		"openai/clip-vit-base-patch32": {status: "ok"},
		"0penai/fake-model":            {status: "not_found"},
	}}
	var stdout, stderr bytes.Buffer
	code := run([]string{"-id", "openai/clip-vit-base-patch32", "-id", "0penai/fake-model", "-json"}, strings.NewReader(""), &stdout, &stderr, client)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 (not_found + typosquat present)", code)
	}
	var findings []finding
	if err := json.Unmarshal(stdout.Bytes(), &findings); err != nil {
		t.Fatalf("invalid JSON output: %v\n%s", err, stdout.String())
	}
	if len(findings) != 2 {
		t.Fatalf("got %d findings, want 2", len(findings))
	}
	for _, f := range findings {
		if f.ID == "0penai/fake-model" {
			if f.Status != "not_found" {
				t.Errorf("0penai/fake-model: status = %q, want not_found", f.Status)
			}
			if f.Typosquat == nil || f.Typosquat.Target != "openai" {
				t.Errorf("0penai/fake-model: typosquat = %+v, want target openai", f.Typosquat)
			}
		}
	}
}

func TestRunFailOnNone(t *testing.T) {
	client := &fakeHFClient{results: map[string]checkResult{
		"0penai/fake-model": {status: "not_found"},
	}}
	var stdout, stderr bytes.Buffer
	code := run([]string{"-id", "0penai/fake-model", "-fail-on", "none"}, strings.NewReader(""), &stdout, &stderr, client)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 with -fail-on none", code)
	}
}

func TestRunGatedFailOn(t *testing.T) {
	client := &fakeHFClient{results: map[string]checkResult{
		"meta-llama/Llama-2-7b": {status: "ok", gated: "manual"},
	}}
	var stdout, stderr bytes.Buffer

	code := run([]string{"-id", "meta-llama/Llama-2-7b"}, strings.NewReader(""), &stdout, &stderr, client)
	if code != 0 {
		t.Fatalf("default fail-on: exit code = %d, want 0 (gated alone shouldn't fail by default)", code)
	}

	stdout.Reset()
	code = run([]string{"-id", "meta-llama/Llama-2-7b", "-fail-on", "gated"}, strings.NewReader(""), &stdout, &stderr, client)
	if code != 1 {
		t.Fatalf("-fail-on gated: exit code = %d, want 1", code)
	}
	if !strings.Contains(stdout.String(), "gated:manual") {
		t.Errorf("text output missing gated annotation: %q", stdout.String())
	}
}

// TestRunSameIDModelAndDataset guards the fix for a false negative where a
// model reference and a dataset reference sharing the exact same "org/name"
// string collapsed into a single finding, silently dropping whichever kind
// lost the collision. The Hub's model and dataset namespaces are
// independent, so the same ID string can be a real dataset (e.g.
// "allenai/c4", confirmed 200 from /api/datasets/) while not existing as a
// model at all (confirmed 401/not_found from /api/models/) — a hallucinated
// AutoModel.from_pretrained("allenai/c4") sitting next to a legitimate
// load_dataset("allenai/c4") in the same file used to be reported as one
// "ok dataset" finding with no sign the model call would fail at runtime.
func TestRunSameIDModelAndDataset(t *testing.T) {
	// fakeHFClient.check ignores kind (it only keys on id), so it can't
	// distinguish "allenai/c4 as a model" from "allenai/c4 as a dataset" —
	// exactly the distinction this test needs to verify. Use a kind-aware
	// client instead.
	kindAware := &kindAwareFakeClient{
		model:   checkResult{status: "not_found"},
		dataset: checkResult{status: "ok"},
	}

	var stdout, stderr bytes.Buffer
	src := "model = AutoModel.from_pretrained(\"allenai/c4\")\n" +
		"ds = load_dataset(\"allenai/c4\")\n"
	code := run([]string{"-fail-on", "not_found"}, strings.NewReader(src), &stdout, &stderr, kindAware)
	if code != 1 {
		t.Fatalf("exit code = %d, want 1 (the model reference is not_found)", code)
	}
	out := stdout.String()
	if !strings.Contains(out, "not_found model   allenai/c4") {
		t.Errorf("stdout missing the not_found model finding, got:\n%s", out)
	}
	if !strings.Contains(out, "ok        dataset allenai/c4") {
		t.Errorf("stdout missing the ok dataset finding, got:\n%s", out)
	}
	if strings.Count(out, "allenai/c4") != 2 {
		t.Errorf("want exactly 2 lines mentioning allenai/c4 (one model, one dataset), got:\n%s", out)
	}
}

// kindAwareFakeClient returns a fixed result per repoKind, regardless of ID
// — just enough to prove that main's dedup logic checks model and dataset
// references separately rather than collapsing them by ID alone.
type kindAwareFakeClient struct {
	model   checkResult
	dataset checkResult
}

func (k *kindAwareFakeClient) check(id string, kind repoKind) checkResult {
	if kind == kindDataset {
		return k.dataset
	}
	return k.model
}

func TestRunNoReferencesFound(t *testing.T) {
	client := &fakeHFClient{results: map[string]checkResult{}}
	var stdout, stderr bytes.Buffer
	code := run(nil, strings.NewReader("print('nothing to see here')\n"), &stdout, &stderr, client)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 when no HF references are found", code)
	}
	if !strings.Contains(stderr.String(), "no Hugging Face") {
		t.Errorf("stderr = %q, want a message about finding no references", stderr.String())
	}
}

func TestRunStdinExtraction(t *testing.T) {
	client := &fakeHFClient{results: map[string]checkResult{
		"openai/clip-vit-base-patch32": {status: "ok"},
	}}
	var stdout, stderr bytes.Buffer
	src := `model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")` + "\n"
	code := run([]string{"-fail-on", "none"}, strings.NewReader(src), &stdout, &stderr, client)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "openai/clip-vit-base-patch32") {
		t.Errorf("stdout = %q, want it to mention the extracted ID", stdout.String())
	}
}

// TestRunNotebookFile is an end-to-end guard (real file on disk, through
// run()'s own filepath.Walk, not just a direct decodeNotebookSource/
// extractReferences unit call) for the fix to a bug where hfaudit scanned a
// .ipynb file's raw JSON bytes directly: nbformat JSON-escapes every
// double-quote in a code cell's real Python source, so a `["']`-anchored
// match never lined up with a double-quoted Hub ID and every notebook using
// the standard double-quote style (confirmed against setfit's own shipped
// notebooks) silently produced zero findings.
func TestRunNotebookFile(t *testing.T) {
	nb := `{"cells": [` +
		`{"cell_type": "code", "source": ["model = AutoModel.from_pretrained(\"openai/clip-vit-base-patch32\")\n"]}` +
		`]}`
	dir := t.TempDir()
	path := filepath.Join(dir, "demo.ipynb")
	if err := os.WriteFile(path, []byte(nb), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	client := &fakeHFClient{results: map[string]checkResult{
		"openai/clip-vit-base-patch32": {status: "ok"},
	}}
	var stdout, stderr bytes.Buffer
	code := run([]string{"-fail-on", "none", path}, strings.NewReader(""), &stdout, &stderr, client)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "openai/clip-vit-base-patch32") {
		t.Errorf("stdout = %q, want it to mention the ID extracted from the notebook's double-quoted call", stdout.String())
	}
}

func TestParseFailOn(t *testing.T) {
	cases := []struct {
		spec string
		want []string
		skip []string
	}{
		{"not_found,typosquat", []string{"not_found", "typosquat"}, []string{"gated", "error"}},
		{"none", nil, []string{"not_found", "typosquat", "gated", "error"}},
		{"", nil, []string{"not_found", "typosquat", "gated", "error"}},
	}
	for _, c := range cases {
		got := parseFailOn(c.spec)
		for _, k := range c.want {
			if !got[k] {
				t.Errorf("parseFailOn(%q)[%q] = false, want true", c.spec, k)
			}
		}
		for _, k := range c.skip {
			if got[k] {
				t.Errorf("parseFailOn(%q)[%q] = true, want false", c.spec, k)
			}
		}
	}
}
