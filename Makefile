# Prefers uv locally and falls back to pip when uv is unavailable (for example,
# in Colab). Override values as needed, e.g.:
# make run-duet INPUT=/path/to/song.flac DEVICE=cuda:0

UV ?= uv
PYTHON ?= python3
PACKAGE_MANAGER ?= auto
THIRD_PARTY ?= third_party
UNMIXX_REPO ?= $(THIRD_PARTY)/unmixx
MSS_REPO ?= $(THIRD_PARTY)/Music-Source-Separation-Training
UNMIXX_REQUIREMENTS_PATCH := $(CURDIR)/patches/unmixx-requirements.patch

INPUT ?= songs/FallingSlowly/01 - Falling Slowly.flac
OUTPUT_DIR ?= run_song
DEVICE ?= cuda
UNMIXX_CHUNK_SECONDS ?= 4
UNMIXX_OVERLAP_SECONDS ?= 1
CHOIR_INPUT ?=
CHOIR_OUTPUT_DIR ?= choir_parts
CHOIR_DEVICE ?= $(DEVICE)
CHOIR_SEGMENT_SECONDS ?= 5.046
CHOIR_OVERLAP_SECONDS ?= 0.5

UV_AVAILABLE := $(shell command -v $(UV) >/dev/null 2>&1 && printf uv || printf pip)
ifeq ($(PACKAGE_MANAGER),auto)
RESOLVED_PACKAGE_MANAGER := $(UV_AVAILABLE)
else
RESOLVED_PACKAGE_MANAGER := $(PACKAGE_MANAGER)
endif

ifeq ($(RESOLVED_PACKAGE_MANAGER),uv)
PREPARE_ENVIRONMENT = $(UV) sync --upgrade --inexact
INSTALL_REQUIREMENTS = $(UV) pip install --upgrade -r
RUN_PYTHON = $(UV) run python
else ifeq ($(RESOLVED_PACKAGE_MANAGER),pip)
PREPARE_ENVIRONMENT = $(PYTHON) -m pip install --upgrade pip
INSTALL_REQUIREMENTS = $(PYTHON) -m pip install --upgrade -r
RUN_PYTHON = $(PYTHON)
else
$(error PACKAGE_MANAGER must be auto, uv, or pip)
endif

.PHONY: help prepare repositories patch-unmixx-requirements dependencies run-duet duet run-full full run-choir-parts choir-parts

help:
	@echo "make prepare    Update upstream repositories and install their latest requirements ($(RESOLVED_PACKAGE_MANAGER))."
	@echo "make run-duet   Prepare the environment and run the duet pipeline."
	@echo "make run-full   Prepare the environment and run the full lead/choir/duet pipeline."
	@echo "make run-choir-parts CHOIR_INPUT=...  Split an existing 01_choir_backing.wav into vocal-ensemble parts."

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
	$(PREPARE_ENVIRONMENT)
	$(INSTALL_REQUIREMENTS) "$(UNMIXX_REPO)/requirements.txt"
	$(INSTALL_REQUIREMENTS) "$(MSS_REPO)/requirements.txt"
	$(INSTALL_REQUIREMENTS) requirements.txt

run-duet duet: prepare
	$(RUN_PYTHON) pipeline.py "$(INPUT)" \
		--mode duet \
		--unmixx-repo "$(UNMIXX_REPO)" \
		--device "$(DEVICE)" \
		--unmixx-chunk-seconds "$(UNMIXX_CHUNK_SECONDS)" \
		--unmixx-overlap-seconds "$(UNMIXX_OVERLAP_SECONDS)" \
		--output-dir "$(OUTPUT_DIR)"

run-full full: prepare
	$(RUN_PYTHON) pipeline.py "$(INPUT)" \
		--mode full \
		--mss-repo "$(MSS_REPO)" \
		--unmixx-repo "$(UNMIXX_REPO)" \
		--device "$(DEVICE)" \
		--unmixx-chunk-seconds "$(UNMIXX_CHUNK_SECONDS)" \
		--unmixx-overlap-seconds "$(UNMIXX_OVERLAP_SECONDS)" \
		--output-dir "$(OUTPUT_DIR)"

run-choir-parts choir-parts: prepare
	@test -n "$(CHOIR_INPUT)" || (echo "Set CHOIR_INPUT to an existing 01_choir_backing.wav file." >&2; exit 2)
	$(RUN_PYTHON) choir_parts_separation.py "$(CHOIR_INPUT)" \
		--output-dir "$(CHOIR_OUTPUT_DIR)" \
		--device "$(CHOIR_DEVICE)" \
		--segment-seconds "$(CHOIR_SEGMENT_SECONDS)" \
		--overlap-seconds "$(CHOIR_OVERLAP_SECONDS)"
