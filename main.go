// Command hfaudit scans source code for Hugging Face Hub model/dataset
// references and checks each one against the public Hub API, flagging IDs
// that don't exist (typos, or names an LLM hallucinated outright) and IDs
// whose namespace is suspiciously close to a well-known org (a possible
// impersonation/typosquat), the same way slopcheck does for PyPI/npm
// package names.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type repoKind string

const (
	kindModel   repoKind = "model"
	kindDataset repoKind = "dataset"
	kindSpace   repoKind = "space"
)

type finding struct {
	ID        string          `json:"id"`
	Kind      repoKind        `json:"kind"`
	Status    string          `json:"status"`
	Gated     string          `json:"gated,omitempty"`
	Source    string          `json:"source"`
	Typosquat *typosquatMatch `json:"typosquat,omitempty"`
}

// stringList accumulates repeated -id flag values.
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, defaultHFClient()))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer, client hfClient) int {
	fs := flag.NewFlagSet("hfaudit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var ids stringList
	fs.Var(&ids, "id", "explicit Hugging Face model ID to check (org/name), repeatable")
	jsonOut := fs.Bool("json", false, "print findings as JSON instead of text")
	failOn := fs.String("fail-on", "not_found,typosquat", "comma-separated conditions that cause a nonzero exit: not_found,typosquat,gated,error (or \"none\")")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: hfaudit [flags] [paths...]")
		fmt.Fprintln(stderr, "  scans the given .py files/directories for Hugging Face model/dataset")
		fmt.Fprintln(stderr, "  references (from_pretrained, pipeline, load_dataset, hf_hub_download,")
		fmt.Fprintln(stderr, "  snapshot_download calls) and checks each one against the public Hub API.")
		fmt.Fprintln(stderr, "  With no paths and no -id flags, reads source text from stdin.")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}

	// refs is keyed by (id, kind), not just id: the Hub's model and dataset
	// namespaces are independent, so the exact same "org/name" string can
	// legitimately be a real dataset and, separately, a nonexistent (or
	// hallucinated) model — e.g. "allenai/c4" is a real dataset (200 from
	// /api/datasets/) but returns 401/not_found from /api/models/. Keying
	// by id alone used to collapse both references into one finding
	// (whichever kind was seen "preferred" — dataset over model — on
	// collision), silently dropping the other kind's check entirely: a
	// hallucinated model reference sharing a real dataset's name was
	// reported as a single "ok dataset" finding with no sign the model
	// call would actually fail at runtime.
	type refKey struct {
		id   string
		kind repoKind
	}
	refs := map[refKey]repoRef{}
	addRef := func(r repoRef) {
		refs[refKey{id: r.id, kind: r.kind}] = r
	}

	for _, id := range ids {
		addRef(repoRef{id: id, kind: kindModel, source: "-id flag"})
	}

	paths := fs.Args()
	if len(paths) == 0 && len(ids) == 0 {
		data, err := io.ReadAll(stdin)
		if err != nil {
			fmt.Fprintln(stderr, "hfaudit:", err)
			return 2
		}
		for _, r := range extractReferences(string(data), "stdin") {
			addRef(r)
		}
	}
	for _, p := range paths {
		err := filepath.Walk(p, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() || !isSourceFile(path) {
				return nil
			}
			data, rerr := os.ReadFile(path)
			if rerr != nil {
				fmt.Fprintln(stderr, "hfaudit:", rerr)
				return nil
			}
			text := string(data)
			if strings.HasSuffix(path, ".ipynb") {
				if decoded, ok := decodeNotebookSource(data); ok {
					text = decoded
				}
				// On decode failure (malformed/non-standard notebook JSON),
				// fall through and scan the raw bytes as plain text rather
				// than silently finding nothing.
			}
			for _, r := range extractReferences(text, path) {
				addRef(r)
			}
			return nil
		})
		if err != nil {
			fmt.Fprintln(stderr, "hfaudit:", err)
			return 2
		}
	}

	if len(refs) == 0 {
		fmt.Fprintln(stderr, "hfaudit: no Hugging Face model/dataset references found")
		return 0
	}

	fail := parseFailOn(*failOn)
	findings := make([]finding, 0, len(refs))
	exitCode := 0
	for key, r := range refs {
		id := key.id
		res := client.check(id, r.kind)
		f := finding{ID: id, Kind: r.kind, Status: res.status, Gated: res.gated, Source: r.source}
		f.Typosquat = closestPopularOrg(id)
		findings = append(findings, f)
		if fail[f.Status] || (f.Typosquat != nil && fail["typosquat"]) || (f.Gated != "" && fail["gated"]) {
			exitCode = 1
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].ID != findings[j].ID {
			return findings[i].ID < findings[j].ID
		}
		return findings[i].Kind < findings[j].Kind
	})

	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(findings)
	} else {
		printText(stdout, findings)
	}
	return exitCode
}

func parseFailOn(spec string) map[string]bool {
	out := map[string]bool{}
	if spec == "none" || spec == "" {
		return out
	}
	for _, s := range strings.Split(spec, ",") {
		out[strings.TrimSpace(s)] = true
	}
	return out
}

func printText(w io.Writer, findings []finding) {
	for _, f := range findings {
		line := fmt.Sprintf("%-9s %-7s %s", f.Status, f.Kind, f.ID)
		if f.Gated != "" {
			line += fmt.Sprintf("  [gated:%s]", f.Gated)
		}
		if f.Typosquat != nil {
			line += fmt.Sprintf("  [looks like %q, edit distance %d]", f.Typosquat.Target, f.Typosquat.Distance)
		}
		line += fmt.Sprintf("  (%s)", f.Source)
		fmt.Fprintln(w, line)
	}
}
