# Prefers uv locally and uses Colab's existing pip/Python environment when the
# COLAB_RELEASE_TAG marker is present. Override values as needed, e.g.:
# make run-duet INPUT=/path/to/song.flac DEVICE=mps

UV ?= uv
IN_COLAB := $(if $(COLAB_RELEASE_TAG),1,0)

ifeq ($(IN_COLAB),1)
DEFAULT_PYTHON := python
DEFAULT_PACKAGE_MANAGER := pip
else
DEFAULT_PYTHON := python3
DEFAULT_PACKAGE_MANAGER := auto
endif

PYTHON ?= $(DEFAULT_PYTHON)
PACKAGE_MANAGER ?= $(DEFAULT_PACKAGE_MANAGER)
THIRD_PARTY ?= third_party
UNMIXX_REPO ?= $(THIRD_PARTY)/unmixx
MSS_REPO ?= $(THIRD_PARTY)/Music-Source-Separation-Training
GAME_REPO ?= $(THIRD_PARTY)/GAME
GAME_MODEL_DIR ?= $(THIRD_PARTY)/GAME-model-large
UNMIXX_REQUIREMENTS_MODE ?= runtime
MSS_REQUIREMENTS_MODE ?= runtime

INPUT ?= songs/FallingSlowly/01 - Falling Slowly.flac
OUTPUT_DIR ?= run_song
DEVICE ?= auto
UNMIXX_CHUNK_SECONDS ?= 4
UNMIXX_OVERLAP_SECONDS ?= 1
UNMIXX_SILENCE_THRESHOLD_DB ?= -80
UNMIXX_IDENTITY_MODEL ?= byol
UNMIXX_IDENTITY_MIN_PEAK_DB ?= -45
UNMIXX_IDENTITY_MIN_MARGIN ?= 0.05
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

UV_INDEX_STRATEGY=first-index
UV_INDEX_URL=https://pypi.org/simple
UV_EXTRA_INDEX_URL=

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

ifeq ($(UNMIXX_REQUIREMENTS_MODE),runtime)
# UNMIXX's upstream requirements.txt is a frozen author workstation export.
# Runtime mode installs only the curated inference dependencies we need.
INSTALL_UNMIXX_REQUIREMENTS = $(INSTALL_REQUIREMENTS) requirements-unmixx-runtime.txt
else ifeq ($(UNMIXX_REQUIREMENTS_MODE),full)
INSTALL_UNMIXX_REQUIREMENTS = $(INSTALL_REQUIREMENTS) "$(UNMIXX_REPO)/requirements.txt"
else
$(error UNMIXX_REQUIREMENTS_MODE must be runtime or full)
endif

ifeq ($(MSS_REQUIREMENTS_MODE),runtime)
# Music-Source-Separation-Training's requirements also include training tools,
# GUI packages, and optional CUDA kernels. Runtime mode installs only the
# curated Mega53 / bs_roformer inference dependencies we actually need.
INSTALL_MSS_REQUIREMENTS = $(INSTALL_REQUIREMENTS) requirements-mss-runtime.txt
else ifeq ($(MSS_REQUIREMENTS_MODE),full)
INSTALL_MSS_REQUIREMENTS = $(INSTALL_REQUIREMENTS) "$(MSS_REPO)/requirements.txt"
else
$(error MSS_REQUIREMENTS_MODE must be runtime or full)
endif

GAME_LANGUAGE_FLAG = $(if $(MIDI_LANGUAGE),--language "$(MIDI_LANGUAGE)")
GAME_GLOB_FLAG = $(if $(MIDI_GLOB),--glob "$(MIDI_GLOB)")

.PHONY: help prepare prepare-game-midi prepare-choir-parts prepare-polyphonic-choir-midi repositories patch-unmixx-requirements dependencies game-dependencies choir-parts-dependencies polyphonic-choir-midi-dependencies download-game-model run-duet duet run-full full run-choir-parts choir-parts run-midi midi run-singing-midi singing-midi run-polyphonic-choir-midi polyphonic-choir-midi

help:
	@echo "make prepare    Update separation repositories and install only separation runtime requirements ($(RESOLVED_PACKAGE_MANAGER))."
	@echo "make prepare-game-midi  Additionally install GAME and download its current MIDI model."
	@echo "make prepare-choir-parts  Additionally install the DPTNet choir-part runtime."
	@echo "make run-duet   Prepare the environment and run the duet pipeline."
	@echo "make run-full   Prepare the environment and run the full lead/choir/duet pipeline."
	@echo "make run-choir-parts CHOIR_INPUT=...  Split an existing 01_choir_backing.wav into vocal-ensemble parts."
	@echo "make run-midi MIDI_INPUT=...  Create MIDI from isolated lead or choir-part stems with GAME Large."
	@echo "make run-polyphonic-choir-midi MIDI_INPUT=...  Create one polyphonic MIDI from an unseparated choir mix."

prepare: repositories patch-unmixx-requirements dependencies

prepare-game-midi: prepare game-dependencies download-game-model

prepare-choir-parts: prepare choir-parts-dependencies

prepare-polyphonic-choir-midi: prepare polyphonic-choir-midi-dependencies

repositories:
	@mkdir -p "$(THIRD_PARTY)"
	@if test -d "$(UNMIXX_REPO)/.git"; then \
		git -C "$(UNMIXX_REPO)" checkout -- requirements.txt look2hear/models/unmixx_model.py && git -C "$(UNMIXX_REPO)" pull --ff-only; \
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

patch-unmixx-requirements: repositories patch_unmixx_requirements.py
	$(RUN_PYTHON) patch_unmixx_requirements.py "$(UNMIXX_REPO)"

dependencies: patch-unmixx-requirements requirements.txt requirements-unmixx-runtime.txt requirements-mss-runtime.txt pyproject.toml
	$(PREPARE_ENVIRONMENT)
	$(INSTALL_UNMIXX_REQUIREMENTS)
	$(INSTALL_MSS_REQUIREMENTS)
	$(INSTALL_REQUIREMENTS) requirements.txt

game-dependencies: repositories
	$(INSTALL_REQUIREMENTS) "$(GAME_REPO)/requirements.txt"

choir-parts-dependencies: requirements-choir-parts.txt
	$(INSTALL_REQUIREMENTS) requirements-choir-parts.txt

polyphonic-choir-midi-dependencies: requirements-polyphonic-choir-midi.txt
	$(INSTALL_REQUIREMENTS) requirements-polyphonic-choir-midi.txt

download-game-model: repositories download_game_model.py
	$(RUN_PYTHON) download_game_model.py --output-dir "$(GAME_MODEL_DIR)" --upgrade

run-duet duet: prepare
	$(RUN_PYTHON) pipeline.py "$(INPUT)" \
		--mode duet \
		--unmixx-repo "$(UNMIXX_REPO)" \
		--device "$(DEVICE)" \
		--unmixx-chunk-seconds "$(UNMIXX_CHUNK_SECONDS)" \
		--unmixx-overlap-seconds "$(UNMIXX_OVERLAP_SECONDS)" \
		--unmixx-silence-threshold-db "$(UNMIXX_SILENCE_THRESHOLD_DB)" \
		--unmixx-identity-model "$(UNMIXX_IDENTITY_MODEL)" \
		--unmixx-identity-min-peak-db "$(UNMIXX_IDENTITY_MIN_PEAK_DB)" \
		--unmixx-identity-min-margin "$(UNMIXX_IDENTITY_MIN_MARGIN)" \
		--output-dir "$(OUTPUT_DIR)"

run-full full: prepare
	$(RUN_PYTHON) pipeline.py "$(INPUT)" \
		--mode full \
		--mss-repo "$(MSS_REPO)" \
		--unmixx-repo "$(UNMIXX_REPO)" \
		--device "$(DEVICE)" \
		--unmixx-chunk-seconds "$(UNMIXX_CHUNK_SECONDS)" \
		--unmixx-overlap-seconds "$(UNMIXX_OVERLAP_SECONDS)" \
		--unmixx-silence-threshold-db "$(UNMIXX_SILENCE_THRESHOLD_DB)" \
		--unmixx-identity-model "$(UNMIXX_IDENTITY_MODEL)" \
		--unmixx-identity-min-peak-db "$(UNMIXX_IDENTITY_MIN_PEAK_DB)" \
		--unmixx-identity-min-margin "$(UNMIXX_IDENTITY_MIN_MARGIN)" \
		--output-dir "$(OUTPUT_DIR)"

run-choir-parts choir-parts: prepare-choir-parts
	@test -n "$(CHOIR_INPUT)" || (echo "Set CHOIR_INPUT to an existing 01_choir_backing.wav file." >&2; exit 2)
	$(RUN_PYTHON) choir_parts_separation.py "$(CHOIR_INPUT)" \
		--output-dir "$(CHOIR_OUTPUT_DIR)" \
		--device "$(CHOIR_DEVICE)" \
		--segment-seconds "$(CHOIR_SEGMENT_SECONDS)" \
		--overlap-seconds "$(CHOIR_OVERLAP_SECONDS)"

run-midi midi: run-singing-midi

run-singing-midi singing-midi: prepare-game-midi
	@test -n "$(MIDI_INPUT)" || (echo "Set MIDI_INPUT to an isolated singing file or directory of stems." >&2; exit 2)
	$(RUN_PYTHON) game_to_midi.py "$(MIDI_INPUT)" --output-dir "$(MIDI_OUTPUT_DIR)" --game-repo "$(GAME_REPO)" --model-dir "$(GAME_MODEL_DIR)" --batch-size "$(MIDI_BATCH_SIZE)" --tempo "$(MIDI_TEMPO)" $(GAME_LANGUAGE_FLAG) $(GAME_GLOB_FLAG)

run-polyphonic-choir-midi polyphonic-choir-midi: prepare-polyphonic-choir-midi
	@test -n "$(MIDI_INPUT)" || (echo "Set MIDI_INPUT to a non-separated choir mix or directory." >&2; exit 2)
	$(RUN_PYTHON) audio_to_midi.py "$(MIDI_INPUT)" \
		--output-dir "$(MIDI_OUTPUT_DIR)"

clean:
	rm -rf *.egg-info
