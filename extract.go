package main

import (
	"fmt"
	"regexp"
	"strings"
)

// repoRef is a Hugging Face Hub ID found in source text, with the kind the
// call site implies and where it came from (for the human-readable report).
type repoRef struct {
	id     string
	kind   repoKind
	source string
}

// idPattern matches a Hugging Face repo ID: exactly one "/" separating an
// org/user segment from a name segment, both drawn from the character set
// the Hub actually allows (letters, digits, '-', '_', '.'). This deliberately
// excludes local filesystem paths with more than one slash and plain
// filenames with no slash at all, which from_pretrained also accepts (e.g.
// "./my-model" or "bert-base-uncased" without a namespace) — those aren't
// Hub IDs this tool can look up.
const idPattern = `[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*`

var extractPatterns = []struct {
	re   *regexp.Regexp
	kind repoKind
}{
	// Any `<something>.from_pretrained("org/name")` — covers AutoModel,
	// AutoTokenizer, every task-specific Auto* class, and third-party
	// wrappers (SentenceTransformer.from_pretrained, CLIPModel, ...).
	{regexp.MustCompile(`\.from_pretrained\(\s*["'](` + idPattern + `)["']`), kindModel},
	// pipeline("task", "org/name") or pipeline("task", model="org/name") —
	// the model ID can be positional (2nd arg) or the model= keyword.
	{regexp.MustCompile(`\bpipeline\(\s*["'][^"']*["']\s*,\s*["'](` + idPattern + `)["']`), kindModel},
	{regexp.MustCompile(`\bpipeline\([^)]*?\bmodel\s*=\s*["'](` + idPattern + `)["']`), kindModel},
	// hf_hub_download(repo_id="org/name") / snapshot_download(repo_id=...)
	{regexp.MustCompile(`\b(?:hf_hub_download|snapshot_download)\([^)]*?\brepo_id\s*=\s*["'](` + idPattern + `)["']`), kindModel},
	// load_dataset("org/name")
	{regexp.MustCompile(`\bload_dataset\(\s*["'](` + idPattern + `)["']`), kindDataset},
}

// extractReferences scans source text for Hugging Face Hub references and
// reports each match's 1-based line number via "label:line" in source.
//
// Matching runs against the whole text rather than line-by-line: real-world
// calls are routinely wrapped across multiple lines (black/ruff formatting,
// or just extra kwargs like trust_remote_code=True), putting the opening
// "from_pretrained(" and the quoted ID on different lines. Per-line matching
// would silently miss every one of those — a false negative in a tool whose
// entire job is catching bad IDs.
func extractReferences(text, label string) []repoRef {
	var refs []repoRef
	for _, p := range extractPatterns {
		for _, m := range p.re.FindAllStringSubmatchIndex(text, -1) {
			id := text[m[2]:m[3]]
			line := 1 + strings.Count(text[:m[0]], "\n")
			refs = append(refs, repoRef{
				id:     id,
				kind:   p.kind,
				source: fmt.Sprintf("%s:%d", label, line),
			})
		}
	}
	return refs
}

func isSourceFile(path string) bool {
	for _, ext := range []string{".py", ".ipynb"} {
		if strings.HasSuffix(path, ext) {
			return true
		}
	}
	return false
}
