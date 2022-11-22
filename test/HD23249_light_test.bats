setup_file() {
    RASSINE_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" >/dev/null 2>&1 && pwd)"
    export RASSINE_DIR
    cd "$RASSINE_DIR"
    cd ..
    python -m venv .venv
    poetry install
    git submodule init test/HD23249_light
    cd test/HD23249_light
    git reset --hard HEAD
    git clean -fdx
    cd ../..
}

@test "HD23249_light import" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 import
}

@test "HD23249_light reinterpolate" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 reinterpolate
}

@test "HD23249_light stacking" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 stacking
}

@test "HD23249_light rassine" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 rassine
}

@test "HD23249_light matching_anchors" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 matching_anchors
}

@test "HD23249_light matching_diff" {
    poetry run ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 matching_diff
}
