# Prefers uv locally and falls back to pip when uv is unavailable (for example,
# in Colab). Override values as needed, e.g.:
# make run-duet INPUT=/path/to/song.flac DEVICE=cuda:0

UV ?= uv
PYTHON ?= python3
PACKAGE_MANAGER ?= auto
THIRD_PARTY ?= third_party
UNMIXX_REPO ?= $(THIRD_PARTY)/unmixx
MSS_REPO ?= $(THIRD_PARTY)/Music-Source-Separation-Training
GAME_REPO ?= $(THIRD_PARTY)/GAME
GAME_MODEL_DIR ?= $(THIRD_PARTY)/GAME-model-large
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
MIDI_INPUT ?=
MIDI_OUTPUT_DIR ?= midi
MIDI_LANGUAGE ?=
MIDI_BATCH_SIZE ?= 4
MIDI_TEMPO ?= 120
MIDI_GLOB ?=

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

GAME_LANGUAGE_FLAG = $(if $(MIDI_LANGUAGE),--language "$(MIDI_LANGUAGE)")
GAME_GLOB_FLAG = $(if $(MIDI_GLOB),--glob "$(MIDI_GLOB)")

.PHONY: help prepare repositories patch-unmixx-requirements dependencies download-game-model run-duet duet run-full full run-choir-parts choir-parts run-midi midi run-singing-midi singing-midi run-polyphonic-choir-midi polyphonic-choir-midi

help:
	@echo "make prepare    Update upstream repositories, install their latest requirements, and update GAME Large ($(RESOLVED_PACKAGE_MANAGER))."
	@echo "make run-duet   Prepare the environment and run the duet pipeline."
	@echo "make run-full   Prepare the environment and run the full lead/choir/duet pipeline."
	@echo "make run-choir-parts CHOIR_INPUT=...  Split an existing 01_choir_backing.wav into vocal-ensemble parts."
	@echo "make run-midi MIDI_INPUT=...  Create MIDI from isolated lead or choir-part stems with GAME Large."
	@echo "make run-polyphonic-choir-midi MIDI_INPUT=...  Create one polyphonic MIDI from an unseparated choir mix."

prepare: repositories patch-unmixx-requirements dependencies download-game-model

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
	@if test -d "$(GAME_REPO)/.git"; then \
		git -C "$(GAME_REPO)" pull --ff-only; \
	else \
		git clone https://github.com/openvpi/GAME.git "$(GAME_REPO)"; \
	fi

patch-unmixx-requirements: repositories $(UNMIXX_REQUIREMENTS_PATCH)
	git -C "$(UNMIXX_REPO)" apply --check "$(UNMIXX_REQUIREMENTS_PATCH)"
	git -C "$(UNMIXX_REPO)" apply "$(UNMIXX_REQUIREMENTS_PATCH)"

dependencies: patch-unmixx-requirements requirements.txt pyproject.toml
	$(PREPARE_ENVIRONMENT)
	$(INSTALL_REQUIREMENTS) "$(UNMIXX_REPO)/requirements.txt"
	$(INSTALL_REQUIREMENTS) "$(MSS_REPO)/requirements.txt"
	$(INSTALL_REQUIREMENTS) "$(GAME_REPO)/requirements.txt"
	$(INSTALL_REQUIREMENTS) requirements.txt

download-game-model: repositories download_game_model.py
	$(RUN_PYTHON) download_game_model.py --output-dir "$(GAME_MODEL_DIR)" --upgrade

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

run-midi midi: run-singing-midi

run-singing-midi singing-midi: prepare
	@test -n "$(MIDI_INPUT)" || (echo "Set MIDI_INPUT to an isolated singing file or directory of stems." >&2; exit 2)
	$(RUN_PYTHON) game_to_midi.py "$(MIDI_INPUT)" --output-dir "$(MIDI_OUTPUT_DIR)" --game-repo "$(GAME_REPO)" --model-dir "$(GAME_MODEL_DIR)" --batch-size "$(MIDI_BATCH_SIZE)" --tempo "$(MIDI_TEMPO)" $(GAME_LANGUAGE_FLAG) $(GAME_GLOB_FLAG)

run-polyphonic-choir-midi polyphonic-choir-midi: prepare
	@test -n "$(MIDI_INPUT)" || (echo "Set MIDI_INPUT to a non-separated choir mix or directory." >&2; exit 2)
	$(RUN_PYTHON) audio_to_midi.py "$(MIDI_INPUT)" \
		--output-dir "$(MIDI_OUTPUT_DIR)"
