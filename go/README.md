# 🪨 `von` (Go)

**Official Go SDK for Von — The Open-Source System One Decision Model.**
*Drop-in replacement for the TypeSafe Jev client.*

[![Go Reference](https://pkg.go.dev/badge/github.com/wfzyx/von/go.svg)](https://pkg.go.dev/github.com/wfzyx/von/go)

---

## Installation

```bash
go get github.com/wfzyx/von/go
```

The SDK is pure standard library — no third-party dependencies.

> The module lives in the `go/` directory of the repository. Go resolves
> submodule tags as `go/vX.Y.Z`, so releases are tagged in lockstep with the
> Python and npm SDKs.

---

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	von "github.com/wfzyx/von/go"
)

func main() {
	// Base URL and API key fall back to VON_BASE_URL / VON_API_KEY (and the
	// TYPESAFE_* equivalents), then to http://localhost:8000 with no auth.
	client := von.NewClient(von.ClientOptions{})

	ctx := context.Background()

	// 1. Multi-question fan-out: one forward pass for every question.
	resp, err := client.SystemOne(ctx,
		"Customer asks for an immediate refund on duplicate invoice charge #401",
		map[string]von.Question{
			"department": von.Choice("Which department should handle this?", von.Descriptions(map[string]string{
				"billing": "Invoices, refunds, payments",
				"tech":    "Software bugs, outages",
			})),
			"isUrgent":   von.Noul("Does the request communicate time pressure?", nil),
			"frustration": von.Score("Rate the user frustration level", []string{
				"Calm", "Concerned", "Very angry",
			}),
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Answers["department"].Choice)      // "billing"
	fmt.Println(resp.Answers["department"].Confidence)  // 0.92
	fmt.Println(resp.Answers["isUrgent"].Noul)          // 0.88

	// 2. Discrete decision helper.
	routing, err := client.Decide(ctx,
		"Server CPU temperature reached 105 degrees Celsius",
		von.Descriptions(map[string]string{
			"hardware": "Hardware/temperature issue",
			"software": "Application bug",
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(routing.Choice) // "hardware"

	// 3. Binary verification helper (Noul). Pass criteria or leave nil.
	isOutage, err := client.Judge(ctx,
		"Database connection pool exhausted on port 5432",
		"Is the database down or failing?",
		&von.NoulCriteria{True: "Failing or unreachable", False: "Healthy"},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(isOutage) // e.g. 0.9412

	// 4. Ordinal rating helper.
	severity, err := client.Rate(ctx,
		"Memory at 98%, OOM killer firing.",
		[]string{"Low", "Medium", "High", "Critical"},
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(severity.Score) // probability-weighted position
}
```

---

## Migration from `@typesafe-ai/sdk`

`TypeSafeClient` is an alias for `VonClient`, so existing call sites only change
the import and the transport:

```go
// Before (TypeSafe Jev):
// client := typesafe.NewClient(typesafe.Options{...})

// After:
client := von.NewClient(von.ClientOptions{
	BaseURL: "http://localhost:8000", // optionally set VON_BASE_URL instead
})
```

---

## API

### Questions

| constructor | wire type | criteria |
|---|---|---|
| `Choice(instructions, map[string]*string)` | `choice` | option key → description (`nil` = key stands in) |
| `ChoiceOptions(instructions, keys ...string)` | `choice` | bare keys, all `nil` descriptions |
| `Noul(instructions, *NoulCriteria)` | `noul` | `{true, false}` descriptions, or `nil` |
| `Score(instructions, []string)` | `score` | ordered level names |
| `ScoreDescriptions(instructions, map[string]string)` | `score` | level → description |

`Descriptions(map[string]string)` and `Options(keys...)` convert the ergonomic
forms into the `map[string]*string` that `Choice` and `Decide` accept.

### Client

| method | endpoint id | returns |
|---|---|---|
| `SystemOne(ctx, state, questions)` | — (full fan-out) | `*SystemOneResponse` |
| `SystemOneModel(ctx, state, questions, model)` | — | `*SystemOneResponse` |
| `Decide(ctx, state, criteria, instructions?)` | `decision` | `*Answer` (`.Choice`) |
| `DecideOptions(ctx, state, keys, instructions?)` | `decision` | `*Answer` |
| `Judge(ctx, state, instructions, criteria)` | `verdict` | `float64` (committed P(true)) |
| `Rate(ctx, state, levels, instructions?)` | `rating` | `*Answer` (`.Score`) |
| `RateDescriptions(ctx, state, levels, instructions?)` | `rating` | `*Answer` |

`instructions...` is variadic: omit it to use the same defaults as the Python
and TypeScript SDKs.

### Errors

Every failure is a `*von.VonError` with the HTTP `Status` (0 for transport and
timeout failures) and raw `Details` for non-2xx responses:

```go
var ve *von.VonError
if errors.As(err, &ve) && ve.Status == http.StatusUnprocessableEntity {
	// oversize state with --on-overflow refuse
}
```

### Confidence gating

`Answer.Noul` is the committed decision and sits in the band. Gate on
`Answer.NoulRaw` (the calibrated pre-band P(true)) instead — read it from a
`SystemOne` response, since `Judge` returns the committed value:

```go
resp, _ := client.SystemOne(ctx, state, map[string]von.Question{
	"blocking": von.Noul("Is this blocking customers?", nil),
})
raw := resp.Answers["blocking"].NoulRaw // *float64
```

---

## A note on Noul criteria keys

The Von backend reads Noul descriptions from criteria keys `"true"` and
`"false"` (see `presets.py` and `option_marker_backend.py`). `NoulCriteria`
encodes exactly those keys. The Python SDK uses
`criteria={"true": ..., "false": ...}`; the JavaScript SDK's `pos`/`neg` keys
are silently ignored by the backend, so its criteria currently have no effect.
This SDK matches the actual wire protocol.

---

## Environment variables

| variable | effect |
|---|---|
| `VON_BASE_URL` / `TYPESAFE_BASE_URL` | server root (default `http://localhost:8000`) |
| `VON_API_KEY` / `TYPESAFE_API_KEY` | sent as `Authorization: Bearer <key>` |

---

## License

Apache-2.0
