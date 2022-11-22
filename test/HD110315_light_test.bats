setup_file() {
    TEST_DIR="$(cd "$(dirname "$BATS_TEST_FILENAME")" >/dev/null 2>&1 && pwd)"
    export TEST_DIR
    cd "$TEST_DIR"
    cd ..
    git submodule update --init test/HD110315_light
    cd test/HD110315_light
    git reset --hard HEAD
    git clean -fdx
    cd ../..
}

@test "HD110315_light import" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN import
}

@test "HD110315_light reinterpolate" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN reinterpolate
}

@test "HD110315_light stacking" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN stacking
}

@test "HD110315_light rassine" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN rassine
    compare_matching_diff_output --kind output test/HD110315_light_RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p test/HD110315_light/data/s1d/HARPN/STACKED/RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p
}

@test "HD110315_light matching_anchors" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN matching_anchors
    compare_matching_diff_output --kind matching_anchors test/HD110315_light_RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p test/HD110315_light/data/s1d/HARPN/STACKED/RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p
}

@test "HD110315_light matching_diff" {
    ./run_rassine.sh -l WARNING -c harpn.ini test/HD110315_light/data/s1d/HARPN matching_diff
    compare_matching_diff_output --kind matching_diff test/HD110315_light_RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p test/HD110315_light/data/s1d/HARPN/STACKED/RASSINE_Stacked_spectrum_B1.00_D2013-01-16T05\:26\:33.144.p
}
