# Uses the uv-managed project environment. Override values as needed, e.g.:
# make run-duet INPUT=/path/to/song.flac DEVICE=cuda:0

UV ?= uv
THIRD_PARTY ?= third_party
UNMIXX_REPO ?= $(THIRD_PARTY)/unmixx
MSS_REPO ?= $(THIRD_PARTY)/Music-Source-Separation-Training
UNMIXX_REQUIREMENTS_PATCH := $(CURDIR)/patches/unmixx-requirements.patch

INPUT ?= songs/FallingSlowly/01 - Falling Slowly.flac
OUTPUT_DIR ?= run_song
DEVICE ?= cuda
UNMIXX_CHUNK_SECONDS ?= 4
UNMIXX_OVERLAP_SECONDS ?= 1

.PHONY: help prepare repositories patch-unmixx-requirements dependencies run-duet duet

help:
	@echo "make prepare    Update both upstream repositories and install their latest requirements with uv."
	@echo "make run-duet   Prepare the environment and run the duet pipeline."

prepare: repositories patch-unmixx-requirements dependencies

repositories:
	@mkdir -p "$(THIRD_PARTY)"
	@if test -d "$(UNMIXX_REPO)/.git"; then \
		git -C "$(UNMIXX_REPO)" checkout -- requirements.txt && git -C "$(UNMIXX_REPO)" pull --ff-only; \
	else \
		git clone https://github.com/jihoojung0106/unmixx.git "$(UNMIXX_REPO)"; \
	fi
	@if test -d "$(MSS_REPO)/.git"; then \
		git -C "$(MSS_REPO)" pull --ff-only; \
	else \
		git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$(MSS_REPO)"; \
	fi

patch-unmixx-requirements: repositories $(UNMIXX_REQUIREMENTS_PATCH)
	git -C "$(UNMIXX_REPO)" apply --check "$(UNMIXX_REQUIREMENTS_PATCH)"
	git -C "$(UNMIXX_REPO)" apply "$(UNMIXX_REQUIREMENTS_PATCH)"

dependencies: patch-unmixx-requirements requirements.txt pyproject.toml
	$(UV) sync --upgrade --inexact
	$(UV) pip install --upgrade -r "$(UNMIXX_REPO)/requirements.txt"
	$(UV) pip install --upgrade -r "$(MSS_REPO)/requirements.txt"
	$(UV) pip install --upgrade -r requirements.txt

run-duet duet: prepare
	$(UV) run python pipeline.py "$(INPUT)" \
		--mode duet \
		--unmixx-repo "$(UNMIXX_REPO)" \
		--device "$(DEVICE)" \
		--unmixx-chunk-seconds "$(UNMIXX_CHUNK_SECONDS)" \
		--unmixx-overlap-seconds "$(UNMIXX_OVERLAP_SECONDS)" \
		--output-dir "$(OUTPUT_DIR)"
