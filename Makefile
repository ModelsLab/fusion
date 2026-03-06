BINARY ?= fusion
CMD := ./cmd/fusion

.PHONY: build install test fmt snapshot

build:
	go build -o bin/$(BINARY) $(CMD)

install:
	go install $(CMD)

test:
	go test ./...

fmt:
	go fmt ./...

snapshot:
	goreleaser release --snapshot --clean
