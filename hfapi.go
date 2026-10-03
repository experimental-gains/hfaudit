package main

import (
	"encoding/json"
	"net/http"
	"time"
)

// hfClient checks whether a Hugging Face Hub repo ID exists, using only the
// Hub's public, unauthenticated metadata API — no account or token needed,
// so this tool has no signup/KYC surface of its own.
type hfClient interface {
	check(id string, kind repoKind) checkResult
}

type checkResult struct {
	status string // "ok", "not_found", or "error"
	gated  string // "", "auto", or "manual" — only meaningful when status is "ok"
}

type httpHFClient struct {
	http *http.Client
	base string
}

func defaultHFClient() hfClient {
	return &httpHFClient{
		http: &http.Client{Timeout: 10 * time.Second},
		base: "https://huggingface.co",
	}
}

// hfRepoMeta is the subset of the Hub API's response this tool reads.
// "gated" is "auto", "manual", or false (a bool, not a string) when the
// repo isn't gated at all — json.RawMessage defers decoding so the bool
// case doesn't fail to unmarshal into a string field.
type hfRepoMeta struct {
	Gated json.RawMessage `json:"gated"`
}

// check returns "ok" for a repo that exists (gated or not — gating only
// restricts downloading files, not reading metadata), "not_found" when it
// doesn't, and "error" when the lookup itself didn't complete.
//
// Confirmed live against the real API (no auth): a missing namespace, a
// real namespace with a nonexistent repo name, and (presumably) a private
// repo an anonymous caller can't see all answer identically with HTTP 401
// and body `{"error":"Invalid username or password."}` — an auth-shaped
// error that in practice just means "nothing here, anonymously". There is
// no separate 404 for this endpoint. Gating is reported inside the 200
// body's "gated" field instead (seen "manual" on meta-llama/Llama-2-7b,
// which still returns full metadata with no auth).
func (c *httpHFClient) check(id string, kind repoKind) checkResult {
	endpoint := "models"
	if kind == kindDataset {
		endpoint = "datasets"
	}
	// id's own "/" separates the org and name segments and must stay a
	// literal path separator, so build the URL by concatenation rather
	// than url.PathEscape(id), which would percent-encode it too.
	u := c.base + "/api/" + endpoint + "/" + id

	resp, err := c.http.Get(u)
	if err != nil {
		return checkResult{status: "error"}
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		var meta hfRepoMeta
		gated := ""
		if json.NewDecoder(resp.Body).Decode(&meta) == nil {
			var s string
			if json.Unmarshal(meta.Gated, &s) == nil {
				gated = s
			}
		}
		return checkResult{status: "ok", gated: gated}
	case http.StatusUnauthorized:
		return checkResult{status: "not_found"}
	default:
		return checkResult{status: "error"}
	}
}
