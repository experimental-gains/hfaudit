package main

import (
	"bytes"
	"encoding/json"
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
