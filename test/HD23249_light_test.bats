setup_file() {
    RASSINE_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" >/dev/null 2>&1 && pwd)"
    export RASSINE_DIR
    cd "$RASSINE_DIR"
    cd ..
    git submodule update --init test/HD23249_light
    cd test/HD23249_light
    git reset --hard HEAD
    git clean -fdx
    cd ../..
}

@test "HD23249_light import" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 import
}

@test "HD23249_light reinterpolate" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 reinterpolate
}

@test "HD23249_light stacking" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 stacking
}

@test "HD23249_light rassine" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 rassine
    compare_normalized_output --kind output test/HD23249_light_RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p test/HD23249_light/data/s1d/HARPS03/STACKED/RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p
}

@test "HD23249_light matching_anchors" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 matching_anchors
    compare_normalized_output --kind matching_anchors test/HD23249_light_RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p test/HD23249_light/data/s1d/HARPS03/STACKED/RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p
}

@test "HD23249_light matching_diff" {
    ./run_rassine.sh -l WARNING -c harps03.ini test/HD23249_light/data/s1d/HARPS03 matching_diff
    compare_normalized_output --kind matching_diff test/HD23249_light_RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p test/HD23249_light/data/s1d/HARPS03/STACKED/RASSINE_Stacked_spectrum_B1.00_D2006-03-18T23\:52\:34.403.p
}
