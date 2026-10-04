package main

import "testing"

func TestClosestPopularOrg(t *testing.T) {
	cases := []struct {
		id       string
		wantNil  bool
		wantOrg  string
		wantDist int
	}{
		{"openai/clip-vit-base-patch32", true, "", 0},                                // exact match on a popular org: that org's own repo, not a typosquat
		{"0penai/fake-model", false, "openai", 1},                                    // digit-for-letter swap
		{"Openai/fake-model", true, "", 0},                                           // case-insensitive exact match
		{"qwen/Qwen2.5-7B", true, "", 0},                                             // exact match, short org name
		{"qwenn/fake", false, "qwen", 1},                                             // short org (<=5 chars): only distance-1 flagged
		{"qwxx/fake", true, "", 0},                                                   // short org: distance 2 from "qwen" must NOT be flagged
		{"some-random-independent-group/my-model", true, "", 0},                      // unrelated namespace, no popular org nearby
		{"sentence-transformer/all-MiniLM-L6-v2", false, "sentence-transformers", 1}, // long org, missing one char
		// Real, independently-legitimate orgs that land within edit distance
		// of a popularOrgs entry by coincidence (or shared lineage), not
		// impersonation — confirmed live against huggingface.co and found by
		// running hfaudit against real transformers/diffusers/accelerate/
		// sentence-transformers source: without the knownLegitimateOrgs
		// exemption, each of these false-flagged a real, heavily-used repo
		// (e.g. FacebookAI/roberta-base, 7.8M+ downloads) as a possible
		// typosquat, which fails the build by default since "typosquat" is
		// in -fail-on's default set.
		{"FacebookAI/roberta-base", true, "", 0},   // HF's own org for Facebook's historical pre-Hub checkpoints
		{"HuggingFaceM4/idefics-9b", true, "", 0},  // Hugging Face's own multimodal team
		{"HuggingFaceTB/SmolLM-360M", true, "", 0}, // Hugging Face's own "Smol Models" team
		{"zai-org/GLM-Image", true, "", 0},         // Z.ai/Zhipu AI's real org, unrelated to xAI
		{"HuggingFaceFW/finewiki", true, "", 0},    // Hugging Face's own "FineData"/science team, publisher of FineWeb
	}
	for _, c := range cases {
		got := closestPopularOrg(c.id)
		if c.wantNil {
			if got != nil {
				t.Errorf("closestPopularOrg(%q) = %+v, want nil", c.id, got)
			}
			continue
		}
		if got == nil {
			t.Errorf("closestPopularOrg(%q) = nil, want target %q dist %d", c.id, c.wantOrg, c.wantDist)
			continue
		}
		if got.Target != c.wantOrg || got.Distance != c.wantDist {
			t.Errorf("closestPopularOrg(%q) = %+v, want {%q %d}", c.id, got, c.wantOrg, c.wantDist)
		}
	}
}

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"abc", "abc", 0},
		{"abc", "", 3},
		{"kitten", "sitting", 3},
		{"openai", "0penai", 1},
		{"qwen", "qwenn", 1},
	}
	for _, c := range cases {
		if got := levenshtein(c.a, c.b); got != c.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}
