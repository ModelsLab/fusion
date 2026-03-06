BINARY ?= fusion
CMD := ./cmd/fusion

.PHONY: build install test fmt snapshot kb

kb:
	go run ./scripts/build_knowledge_db.go

build:
	$(MAKE) kb
	go build -o bin/$(BINARY) $(CMD)

install:
	$(MAKE) kb
	go install $(CMD)

test:
	$(MAKE) kb
	go test ./...

fmt:
	go fmt ./...

snapshot:
	goreleaser release --snapshot --clean
