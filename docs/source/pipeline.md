# Pipeline in run_rassine.sh

TODO: All the paths below refer to RASSINE_ROOT, which 

```{graphviz}

    digraph process {
        newrank=false

        // declare nodes, needs to be done before

        subgraph cluster_input_data {
            label=<<i>input data</i>>
            peripheries=0

            "RAW/{name}.fits" [shape=none]
            "DACE_TABLE/Dace_extracted_table.csv" [
                shape=none,
                label=<DACE_TABLE/Dace_extracted_table.csv<br/><font color="#444444">rassine.imports.data.DACE</font>>,
                href="../_autosummary/rassine.imports.data.DACE.html#rassine.imports.data.DACE",
                target="_top"
            ]
        }

        subgraph cluster_data0 {
            peripheries=0

            "PREPROCESSED/{name}.p (bis)" [
                shape=none
                label=<PREPROCESSED/{name}.p<br/><font color="#444444">rassine.imports.data.ReinterpolatedSpectrumPickle</font>>,
                href="../_autosummary/rassine.imports.data.ReinterpolatedSpectrumPickle.html",
                target="_top"
            ]
            
            "individual_reinterpolated.csv" [
                shape=none,
                label=<individual_reinterpolated.csv<br/><font color="#444444">rassine.imports.data.IndividualReinterpolatedRow</font>>,
                href="../_autosummary/rassine.imports.data.IndividualReinterpolatedRow.html",
                target="_top"
            ]
        }

        subgraph cluster_data1 {
            peripheries=0

            "stacked_basic.csv" [
                shape=none,
                label=<stacked_basic.csv<br/><font color="#444444">rassine.stacking.data.StackedBasicRow</font>>,
                href="../_autosummary/rassine.stacking.data.StackedBasicRow.html",
                target="_top"
            ]
            "master_spectrum.csv" [
                shape=none,
                label=<master_spectrum.csv<br/><font color="#444444">rassine.stacking.data.MasterRow</font>>,
                href="../_autosummary/rassine.stacking.data.MasterRow.html",
                target="_top"
            ]
            "STACKED/{name}.p" [
                shape=none,
                label=<STACKED/{name}.p<br/><font color="#444444">rassine.stacking.data.StackedPickle</font>>,
                href="../_autosummary/rassine.stacking.data.StackedPickle.html",
                target="_top"
            ]
            "MASTER/Master_spectrum_{tag}.p" [
                shape=none,
                label=<MASTER/Master_spectrum_{tag}.p<br/><font color="#444444">rassine.stacking.data.MasterPickle</font>>,
                href="../_autosummary/rassine.stacking.data.MasterPickle.html",
                target="_top"
            ]
        }

        subgraph cluster_data2         {
            peripheries=0

            "MASTER/RASSINE_Master_spectrum_{tag}.p" [
                shape=none,
                label=<MASTER/RASSINE_Master_spectrum_{tag}.p<br/><font color="#444444">rassine.rassine.data.RassinePickle</font>>,
                href="../_autosummary/rassine.rassine.data.RassinePickle.html",
                target="_top"
            ]
            "STACKED/RASSINE_{name}.p" [
                shape=none,
                label=<STACKED/RASSINE_{name}.p<br/><font color="#444444">rassine.rassine.data.RassinePickle</font>>,
                href="../_autosummary/rassine.rassine.data.RassinePickle.html",
                target="_top"
            ]
        }

        subgraph cluster_output_data {
            label=<<i>output data</i>>
            peripheries=0

            "STACKED/RASSINE_{name}.p (ter)" [
                shape=none,
                label=<STACKED/RASSINE_{name}.p<br/><font color="#444444">rassine.matching.data.MatchingPickle</font>>,
                href="../_autosummary/rassine.matching.data.MatchingPickle.html",
                target="_top"
            ]
        }

        // now we go from top to bottom

        subgraph cluster_import {
            label=<<i>import</i>>

            // data

            "individual_basic.csv" [
                shape=none,
                label=<individual_basic.csv<br/><font color="#444444">rassine.imports.data.IndividualBasicRow</font>>,
                href="../_autosummary/rassine.imports.data.IndividualBasicRow.html",
                target="_top"
            ]

            "PREPROCESSED/{name}.p" [
                shape=none
                label=<PREPROCESSED/{name}.p<br/><font color="#444444">rassine.imports.data.PickledIndividualSpectrum</font>>,
                href="../_autosummary/rassine.imports.data.PickledIndividualSpectrum.html",
                target="_top"
            ]

            "individual_imported.csv" [
                shape=none,
                label=<individual_imported.csv<br/><font color="#444444">rassine.imports.data.IndividualImportedRow</font>>,
                href="../_autosummary/rassine.imports.data.IndividualImportedRow.html",
                target="_top"
            ]

            // CLI tools and edges

            "preprocess_table" [shape=box, href="../cli/preprocess_table.html", target="_top"]
                "RAW/{name}.fits" -> "preprocess_table" [style="dashed"]
                "DACE_TABLE/Dace_extracted_table.csv" -> "preprocess_table"
                "preprocess_table" -> "individual_basic.csv"

            "preprocess_import" [shape=box, href="../cli/preprocess_import.html", target="_top"]
                "individual_basic.csv" -> "preprocess_import"
                "RAW/{name}.fits" -> "preprocess_import"
                "preprocess_import" -> "PREPROCESSED/{name}.p"
                "preprocess_import" -> "individual_imported.csv"

            "reinterpolate"  [shape=box, href="../cli/reinterpolate.html", target="_top"]
                "individual_imported.csv" -> "reinterpolate"
                "PREPROCESSED/{name}.p" -> "reinterpolate"
                "reinterpolate" -> "PREPROCESSED/{name}.p (bis)"
                "reinterpolate" -> "individual_reinterpolated.csv"
        }

        subgraph cluster_stacking1 {
            label=<<i>stacking</i>>

            // data

            "individual_group.csv" [
                shape=none,
                label=<individual_group.csv<br/><font color="#444444">rassine.stacking.data.IndividualGroupRow</font>>,
                href="../_autosummary/rassine.stacking.data.IndividualGroupRow.html",
                target="_top"
            ]

            // CLI tools and edges

            "stacking_create_groups" [shape="box", href="../cli/stacking_create_groups.html", target="_top"]
                "individual_reinterpolated.csv" -> "stacking_create_groups"
                "stacking_create_groups" -> "individual_group.csv"


            "stacking_stack" [shape="box", href="../cli/stacking_stack.html", target="_top"]
                "individual_group.csv" -> "stacking_stack"
                "individual_reinterpolated.csv" -> "stacking_stack"
                "PREPROCESSED/{name}.p (bis)" -> "stacking_stack"
                "stacking_stack" -> "stacked_basic.csv"
                "stacking_stack" -> "STACKED/{name}.p"
        }

        subgraph cluster_stacking2 {
            label=<<i>stacking</i>>

            // CLI tools and edges

            "stacking_master_spectrum" [shape="box", href="../cli/stacking_master_spectrum.html", target="_top"]
                "STACKED/{name}.p" -> "stacking_master_spectrum"
                "stacked_basic.csv" -> "stacking_master_spectrum"
                "stacking_master_spectrum" -> "MASTER/Master_spectrum_{tag}.p"
                "stacking_master_spectrum" -> "master_spectrum.csv"
        }

        subgraph cluster_rassine {
            label=<<i>rassine</i>>

            // data

            "anchor_Master_spectrum_{tag}.ini" [
                shape=none,
                label=<anchor_Master_spectrum_{tag}.ini<br/><font color="#444444">Config INI for rassine CLI tool</font>>,
                href="../cli/rassine.html",
                target="_top"
            ]
    
            // CLI tools and edges

            "rassine1" [label="rassine", shape="box", href="../cli/rassine.html", target="_top"]
                "MASTER/Master_spectrum_{tag}.p" -> "rassine1"
                "rassine1" -> "anchor_Master_spectrum_{tag}.ini"
                "rassine1" -> "MASTER/RASSINE_Master_spectrum_{tag}.p"

            "rassine2" [label="rassine", shape="box", href="../cli/rassine.html", target="_top"]
                "STACKED/{name}.p" -> "rassine2"
                "anchor_Master_spectrum_{tag}.ini" -> "rassine2"
                "stacked_basic.csv" -> "rassine2"
                "rassine2" -> "STACKED/RASSINE_{name}.p"
        }

        subgraph cluster_matching {
            label=<<i>matching</i>>

            // data

            "MASTER/Master_tool_{tag}.p" [
                shape=none,
                label=<MASTER/Master_tool_{tag}.p<br/><font color="#444444">rassine.matching.data.MasterToolPickle</font>>,
                href="../_autosummary/rassine.matching.data.MasterToolPickle.html",
                target="_top"
            ]
            "MASTER/RASSINE_Master_spectrum_{tag}.p (bis)" [
                shape=none,
                label=<MASTER/RASSINE_Master_spectrum_{tag}.p<br/><font color="#444444">rassine.matching.data.AnchorPickle</font>>,
                href="../_autosummary/rassine.matching.data.AnchorPickle.html",
                target="_top"
            ]
            "STACKED/RASSINE_{name}.p (bis)" [
                shape=none,
                label=<STACKED/RASSINE_{name}.p<br/><font color="#444444">rassine.matching.data.AnchorPickle</font>>,
                href="../_autosummary/rassine.matching.data.AnchorPickle.html",
                target="_top"
            ]
            "matching_anchors.csv" [
                shape=none,
                label=<matching_anchors.csv<br/><font color="#444444">rassine.matching.data.MatchingAnchorsRow</font>>,
                href="../_autosummary/rassine.matching.data.MatchingAnchorsRow.html",
                target="_top"
            ]

            // CLI tools and edges

            "matching_anchors_scan" [shape=box, href="../cli/matching_anchors_scan.html", target="_top"]
                "stacked_basic.csv" -> "matching_anchors_scan"
                "STACKED/RASSINE_{name}.p" -> "matching_anchors_scan"
                "MASTER/RASSINE_Master_spectrum_{tag}.p" -> "matching_anchors_scan" [style="dashed"]
                "matching_anchors_scan" -> "MASTER/Master_tool_{tag}.p"

            "matching_anchors_filter1" [label="matching_anchors_filter", shape=box, href="../cli/matching_anchors_filter.html", target="_top"]
                "stacked_basic.csv" -> "matching_anchors_filter1"
                "STACKED/RASSINE_{name}.p" -> "matching_anchors_filter1"
                "MASTER/Master_tool_{tag}.p" -> "matching_anchors_filter1"
                "matching_anchors_filter1" -> "STACKED/RASSINE_{name}.p (bis)"
                "matching_anchors_filter1" -> "matching_anchors.csv"

            "matching_anchors_filter2" [label="matching_anchors_filter", shape=box, href="../cli/matching_anchors_filter.html", target="_top"]
                "MASTER/RASSINE_Master_spectrum_{tag}.p" -> "matching_anchors_filter2"
                "MASTER/Master_tool_{tag}.p" -> "matching_anchors_filter2"
                "matching_anchors_filter2" -> "MASTER/RASSINE_Master_spectrum_{tag}.p (bis)"
                "matching_anchors_filter2" -> "matching_anchors.csv"

            "matching_diff" [shape=box, href="../cli/matching_diff.html", target="_top"]
                "MASTER/RASSINE_Master_spectrum_{tag}.p (bis)" -> "matching_diff"
                "stacked_basic.csv" -> "matching_diff"
                "STACKED/RASSINE_{name}.p (bis)" -> "matching_diff"
                "matching_diff" -> "STACKED/RASSINE_{name}.p (ter)"
        }
    }
```