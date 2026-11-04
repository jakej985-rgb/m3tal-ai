# Go quality baseline

Baseline run on 2026-11-03 with Go 1.23.2, against the current working tree.

| Check | Result |
| --- | --- |
| `go build ./...` | Pass |
| `go vet ./...` | Pass |
| `go test ./...` | Pass |
| `staticcheck ./...` | Not run: `staticcheck` is not installed in this environment |
| `govulncheck ./...` | Not run: `govulncheck` is not installed in this environment |

No failures were reported by the checks that ran. The CI workflow installs and
runs Staticcheck and govulncheck on pushes and pull requests.
