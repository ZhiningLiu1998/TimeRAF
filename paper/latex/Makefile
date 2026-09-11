TECTONIC_BIN ?= $(if $(wildcard ../tools/tectonic),../tools/tectonic,tectonic)

PAPER_ROOT := $(abspath ..)
BUILD_DIR := build
OUTPUT := main.pdf

.PHONY: pdf

pdf:
	@mkdir -p "$(BUILD_DIR)" "$(PAPER_ROOT)/.cache" \
		"$(PAPER_ROOT)/.config" "$(PAPER_ROOT)/.tmp"
	@env XDG_CACHE_HOME="$(PAPER_ROOT)/.cache" \
		XDG_CONFIG_HOME="$(PAPER_ROOT)/.config" \
		TMPDIR="$(PAPER_ROOT)/.tmp" \
		"$(TECTONIC_BIN)" -X compile main.tex \
		--outdir "$(BUILD_DIR)" --keep-intermediates --keep-logs --only-cached
	@mv "$(BUILD_DIR)/main.pdf" "$(OUTPUT)"
	@printf 'Wrote %s/%s\n' "$(CURDIR)" "$(OUTPUT)"
