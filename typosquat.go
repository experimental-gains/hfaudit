package main

import "strings"

// popularOrgs is a curated list of Hugging Face Hub namespaces popular
// enough that an attacker impersonating one, or an LLM hallucinating a
// near-miss of one, is worth flagging. Lowercase; Hub namespaces are
// case-sensitive on the Hub itself but attackers (and LLMs) routinely vary
// case, so comparison below is case-insensitive.
var popularOrgs = []string{
	"openai", "google", "meta-llama", "facebook", "microsoft",
	"stabilityai", "mistralai", "qwen", "deepseek-ai", "nvidia",
	"huggingface", "huggingfaceh4", "bigscience", "eleutherai",
	"togethercomputer", "thebloke", "sentence-transformers",
	"laion", "runwayml", "black-forest-labs", "anthropic", "xai-org",
	"ai21labs", "cohere", "databricks", "salesforce", "ibm",
	"allenai", "bigcode", "codellama", "unsloth", "unitary",
	"intfloat", "baai", "openbmb", "internlm", "01-ai",
}

// typosquatMatch names the popular org a given namespace is suspiciously
// close to, and how close (edit distance between the two namespaces).
type typosquatMatch struct {
	Target   string `json:"target"`
	Distance int    `json:"distance"`
}

// knownLegitimateOrgs lists real, actively-used Hub namespaces that happen
// to land within closestPopularOrg's edit-distance threshold of a
// popularOrgs entry, but aren't impersonations — each is itself a
// substantial, independently-real organization, confirmed live against
// /api/organizations/{name}/overview:
//
//   - "facebookai" (distance 2 from "facebook") — the transformers team's
//     own org "maintained by the transformers team at Hugging Face" for
//     Facebook's historical pre-Hub checkpoints (RoBERTa, XLM, XLM-RoBERTa);
//     RoBERTa-base alone has 7.8M+ downloads.
//   - "huggingfacem4" and "huggingfacetb" (distance 1 and 2 from
//     "huggingfaceh4" and "huggingface") — Hugging Face's own internal
//     teams (multimodal, and the "Smol Models Research" group), team-plan
//     orgs with dozens of models/datasets each.
//   - "zai-org" (distance 1 from "xai-org") — Z.ai/Zhipu AI's real org
//     (154 models, 2179 followers, the GLM model family's publisher), an
//     entirely different company from xAI that happens to be one edit
//     away from "xai-org" by coincidence, not imitation.
//   - "huggingfacefw" (distance 2 from "huggingface") — Hugging Face's own
//     "FineData"/science team org (105 models, 35 datasets, 1864
//     followers), publisher of the FineWeb/FineWeb-Edu/finewiki datasets.
//
// Found by running hfaudit against the real transformers, diffusers, peft,
// accelerate, and sentence-transformers library source (not synthetic
// examples): every one of these namespaces triggered a false "possible
// typosquat" flag on an extremely popular, legitimate repo — e.g.
// FacebookAI/roberta-base, HuggingFaceTB/SmolLM-360M, zai-org/GLM-Image,
// HuggingFaceFW/finewiki — across six unrelated real codebases, not a
// one-off (the last found separately, in peft's own
// method_comparison/MetaMathQA/data.py). Since hfaudit's default -fail-on
// includes "typosquat", any project merely referencing RoBERTa,
// XLM-RoBERTa, IDEFICS, SmolLM/SmolVLM, GLM, or FineWeb-family data would
// fail its build on these false positives. Lowercase; compared case-
// insensitively like popularOrgs itself.
var knownLegitimateOrgs = map[string]bool{
	"facebookai":    true,
	"huggingfacem4": true,
	"huggingfacetb": true,
	"huggingfacefw": true,
	"zai-org":       true,
}

// closestPopularOrg returns the nearest popularOrgs entry to id's namespace
// segment, if it's close enough to be worth flagging: distance 0 would be
// an exact match (not a typosquat, just that org's own repo) so this only
// ever returns a match for distance 1 or 2 — one or two edits away from a
// name an attacker or a hallucinating LLM could easily have landed on
// instead of the real thing, but not so loose it flags unrelated short
// names. Returns nil if no popular org is within that range.
func closestPopularOrg(id string) *typosquatMatch {
	namespace, _, ok := strings.Cut(id, "/")
	if !ok {
		return nil
	}
	ns := strings.ToLower(namespace)
	if knownLegitimateOrgs[ns] {
		return nil
	}

	best := -1
	bestOrg := ""
	for _, org := range popularOrgs {
		if ns == org {
			return nil
		}
		// Cap how many edits still count as "suspiciously close" relative
		// to the org name's own length: for a 4-letter org like "qwen",
		// distance 2 is half the string and matches too much to be a
		// meaningful signal, so short names get a tighter cap than long
		// ones like "sentence-transformers".
		maxDist := 2
		if len(org) <= 5 {
			maxDist = 1
		}
		d := levenshtein(ns, org)
		if d >= 1 && d <= maxDist && (best == -1 || d < best) {
			best = d
			bestOrg = org
		}
	}
	if best >= 1 {
		return &typosquatMatch{Target: bestOrg, Distance: best}
	}
	return nil
}

// levenshtein returns the classic edit distance between a and b.
func levenshtein(a, b string) int {
	if a == b {
		return 0
	}
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	cur := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := cur[j-1] + 1
			sub := prev[j-1] + cost
			m := del
			if ins < m {
				m = ins
			}
			if sub < m {
				m = sub
			}
			cur[j] = m
		}
		prev, cur = cur, prev
	}
	return prev[len(rb)]
}
