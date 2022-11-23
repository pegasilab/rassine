#!/bin/bash
# shellcheck disable=SC2181
# ^ due to how we check for exit status

# treat unset variables as an error
set -u

programname=$0

# Help message
function usage {
  echo "Runs the RASSINE pipeline using standardized folder/file names"
  echo
  echo "The pipeline will be run according to https://pegasilab.github.io/rassine/pipeline.html"
  echo "with all the paths in this data graph relative to the provided RASSINE_ROOT"
  echo "parameter. The RASSINE_ROOT can be a relative path, in which case it will be resolved"
  echo "using the current working directory."
  echo
  echo "Note that the parallelization parameters (number of threads, size of chunks, nice level)"
  echo "are set according to the provided configuration file."
  echo
  echo "The parameters [STEP1 STEP2 ...] must be taken from the list:"
  echo "  import reinterpolate stacking rassine matching_anchors matching_diff"
  echo
  echo "The order can be arbitrary, the pipeline always runs in the set order."
  echo
  echo "usage: $programname [-h] [-l {ERROR|WARNING|INFO|DEBUG}] [-c CONFIG_FILE] [RASSINE_ROOT] [STEP1 STEP2 ...]"
  echo "  options:"
  echo "    -h               Display usage information"
  echo "    -l LOGGING_LEVEL Set the logging level (optional, default: WARNING)"
  echo "    -c CONFIG_FILE   Sets the configuration (required)"
  echo "                     In doubt, use harpn.ini or harps03.ini from"
  echo "                     https://github.com/pegasilab/rassine"
  echo
  echo "Examples:"
  echo
  echo "  ./run_rassine.sh -c harpn.ini HD110315/data/s1d/HARPN"
  echo "    runs the whole pipeline for the data in the given folder using the harpn.ini config file"
  echo
  echo "  ./run_rassine.sh -c harpn.ini HD110315/data/s1d/HARPN import reinterpolate stacking"
  echo "    runs the first three steps, which reinterpolates and stacks the spectra"
  echo
}

# Processing command line arguments
function process_command_line_arguments {
  unset -v RASSINE_CONFIG
  unset -v RASSINE_ROOT
  RASSINE_LOGGING_LEVEL=WARNING
  local opt
  local nprocesses
  local nchunks
  local nice

  RASSINE_CONFIG=

  while getopts hl:c: opt; do
    case $opt in
    h) usage; exit 0 ;;
    l) RASSINE_LOGGING_LEVEL=$OPTARG ;;
    c) RASSINE_CONFIG=$(python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$OPTARG") ;;
    *) usage; exit 1 ;;
    esac
  done

  if [ -z "$RASSINE_CONFIG" ]; then
    echo "Error: CONFIG_FILE not provided"
    echo
    usage
    exit 1
  fi

  shift "$((OPTIND - 1))"

  # Process remaining positional arguments

  # Rassine root folder
  if [ "$#" -lt 1 ]; then
    echo "Error: RASSINE_ROOT not provided"
    usage
    exit 1
  fi
  RASSINE_ROOT=$(python -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1")
  shift 1

  # Steps to run. If no argument provided, run all steps.
  if [ "$#" -eq 0 ]; then
    STEPS="import reinterpolate stacking rassine matching_anchors matching_diff"
  else
    STEPS="$*"
  fi

  # Setting up general configuration parameters from the configuration file

  nprocesses=$(python3 -c "import configparser; c = configparser.ConfigParser(); c.read('harpn.ini'); print(c['run_rassine']['nprocesses'])") # number of concurrent jobs
  nchunks=$(python3 -c "import configparser; c = configparser.ConfigParser(); c.read('harpn.ini'); print(c['run_rassine']['nchunks'])")       # number of items per Python script invocation
  nice=$(python3 -c "import configparser; c = configparser.ConfigParser(); c.read('harpn.ini'); print(c['run_rassine']['nice'])")             # number of items per Python script invocation
  PARALLEL_OPTIONS=(--nice "$nice" --will-cite "-N$nchunks" --jobs "$nprocesses" --keep-order --halt "soon,fail=1")
  if [ "$RASSINE_LOGGING_LEVEL" == "INFO" ] || [ "$RASSINE_LOGGING_LEVEL" == "DEBUG" ]; then
    PARRALEL_OPTIONS+=(--verbose)
  fi

  export RASSINE_LOGGING_LEVEL
  export RASSINE_CONFIG
  export RASSINE_ROOT
}

# Create a tag, used later for the master spectrum filename
function create_or_read_tag_and_set_master_filename {

  if [ ! -d "$RASSINE_ROOT" ]; then
    echo Error: root directory "$RASSINE_ROOT" does not exist
    exit 1
  fi
  if [ ! -f "$RASSINE_ROOT/tag" ]; then
    # create a new tag
    date -u +%Y-%m-%dT%H:%M:%S >"$RASSINE_ROOT/tag"
  fi

  TAG=$(<"$RASSINE_ROOT/tag")
  MASTER="Master_spectrum_$TAG.p"
}

# Import and preprocesses spectra
#
# We read the FITS files in the format of a particular instrument and output files in the format
# expected by RASSINE. The data is reformatted, and a few parameters are extracted.
function import_step {
  local individual_imported

  mkdir -p "$RASSINE_ROOT/PREPROCESSED"

  # Preprocess the DACE table to extract a few key parameters in a CSV file
  preprocess_table -I DACE_TABLE/Dace_extracted_table.csv -i RAW -O individual_basic.csv
  # exit if the previous command exited with an error (we will do this repeatedly in this script)
  if [ $? -ne 0 ]; then exit 1; fi

  # Now process individual spectra. The processing is done in parallel, and each step appends to
  # CSV file. It is thus important that the CSV file is empty before starting.

  # Delete the produced table if it exists already
  individual_imported="$RASSINE_ROOT/individual_imported.csv"
  [ -f "$individual_imported" ] && rm "$individual_imported"

  # Perform the import step in parallel
  enumerate_table_rows individual_basic.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    preprocess_import -i RAW -o 'PREPROCESSED/{name}.p' -I individual_basic.csv -O individual_imported.csv
  if [ $? -ne 0 ]; then exit 1; fi

  # Reorder the lines in the CSV file for clarity
  reorder_csv --column name --reference individual_basic.csv individual_imported.csv
  if [ $? -ne 0 ]; then exit 1; fi

}

# Reinterpolation step
function reinterpolate_step {
  local individual_reinterpolated

  # We reinterpolate in parallel. The same procedure as in "import_step" is used with respect
  # to the outputted CSV file.

  individual_reinterpolated="$RASSINE_ROOT/individual_reinterpolated.csv"
  [ -f "$individual_reinterpolated" ] && rm "$individual_reinterpolated"

  enumerate_table_rows individual_imported.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    reinterpolate -i PREPROCESSED -o PREPROCESSED -I individual_imported.csv -O individual_reinterpolated.csv
  if [ $? -ne 0 ]; then exit 1; fi

  reorder_csv --column name --reference individual_imported.csv individual_reinterpolated.csv
  if [ $? -ne 0 ]; then exit 1; fi
}

# Stacking
#
# Temporal summation to reduce the noise; sum instead of average as to keep the error estimation
#
# This also creates the MASTER file which sums all the spectra
#
# The output files are written in /STACKED
function stacking_step {

  stacking_create_groups -I individual_reinterpolated.csv -O individual_group.csv
  if [ $? -ne 0 ]; then exit 1; fi

  stacked_basic=$RASSINE_ROOT/stacked_basic.csv
  [ -f "$stacked_basic" ] && rm "$stacked_basic"

  enumerate_table_column_unique_values -c group individual_group.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    stacking_stack -I individual_reinterpolated.csv -G individual_group.csv -O stacked_basic.csv -i PREPROCESSED -o STACKED
  if [ $? -ne 0 ]; then exit 1; fi

  sort_csv --column group stacked_basic.csv
  if [ $? -ne 0 ]; then exit 1; fi

  stacking_master_spectrum -I stacked_basic.csv -O master_spectrum.csv -i STACKED -o "MASTER/$MASTER"
  if [ $? -ne 0 ]; then exit 1; fi

}

# RASSINE normalization
#
# First we process the Master file to obtain the RASSINE_Master* file
#
# This computes the parameters of the model, which are written in a "anchor INI file".
#
# This anchor INI file is provided during the (parallel) processing of the stacked spectra.
function rassine_step {
  rassine --input-spectrum "$MASTER" --input-folder MASTER --output-folder MASTER \
    --output-plot-folder MASTER --output-anchor-ini "anchor_Master_spectrum_$TAG.ini"
  if [ $? -ne 0 ]; then exit 1; fi

  enumerate_table_rows stacked_basic.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    rassine --input-table stacked_basic.csv --input-folder STACKED --output-folder STACKED \
    --config "$RASSINE_ROOT/anchor_Master_spectrum_$TAG.ini" --output-folder STACKED --output-plot-folder STACKED
  if [ $? -ne 0 ]; then exit 1; fi
}

# Matching anchors step
function matching_anchors_step {

  # We first compute the parameters and create the master tool file.
  # See Fig D7 of the paper
  matching_anchors_scan --input-table stacked_basic.csv --input-pattern 'STACKED/RASSINE_{name}.p' \
    --output-file "MASTER/Master_tool_$TAG.p" --no-master-spectrum
  if [ $? -ne 0 ]; then exit 1; fi

  # Then we apply the master tool in parallel. The files are modified in place.

  anchors_table=$RASSINE_ROOT/matching_anchors.csv
  [ -f "$anchors_table" ] && rm "$anchors_table"

  enumerate_table_rows stacked_basic.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    matching_anchors_filter --master-tool "MASTER/Master_tool_$TAG.p" --process-table stacked_basic.csv --process-pattern 'STACKED/RASSINE_{name}.p' --output-table matching_anchors.csv
  if [ $? -ne 0 ]; then exit 1; fi

  reorder_csv --column name --reference stacked_basic.csv matching_anchors.csv
  if [ $? -ne 0 ]; then exit 1; fi

  # We process the master spectrum last
  matching_anchors_filter --master-tool "MASTER/Master_tool_$TAG.p" --process-master "MASTER/RASSINE_$MASTER" --output-table matching_anchors.csv
  if [ $? -ne 0 ]; then exit 1; fi
}

# Matching diff step
function matching_diff_step {
  # step done in parallel
  enumerate_table_rows stacked_basic.csv | ./parallel "${PARALLEL_OPTIONS[@]}" \
    matching_diff --anchor-file "MASTER/RASSINE_$MASTER" --process-table stacked_basic.csv --process-pattern 'STACKED/RASSINE_{name}.p'
  touch "$RASSINE_ROOT/rassine_finished.txt"
}

process_command_line_arguments "$@"
create_or_read_tag_and_set_master_filename

if [[ "$STEPS" == *"import"* ]]; then
  echo
  echo "Step: import"
  time import_step
fi

if [[ "$STEPS" == *"reinterpolate"* ]]; then
  echo
  echo "Step: reinterpolate"
  time reinterpolate_step
fi

if [[ "$STEPS" == *"stacking"* ]]; then
  echo
  echo "Step: stacking"
  time stacking_step
fi

if [[ "$STEPS" == *"rassine"* ]]; then
  echo
  echo "Step: RASSINE (can be long, inspect the $RASSINE_ROOT/STACKED directory for progress)"
  time rassine_step
fi

if [[ "$STEPS" == *"matching_anchors"* ]]; then
  echo
  echo "Step: matching_anchors"
  time matching_anchors_step
fi

if [[ "$STEPS" == *"matching_diff"* ]]; then
  echo
  echo "Step: matching_diff"
  time matching_diff_step
fi

rm "$RASSINE_ROOT/"*.lock 2>/dev/null || true
