from argparse import ArgumentParser

def build_data_config(parser: ArgumentParser) -> None:
    data_grp = parser.add_argument_group("data", description="Data IO parameters")
    data_grp.add_argument(
        "--train-trees", "-t", required=False, help="Directory with training trees"
    )
    data_grp.add_argument(
        "--train-alignments",
        "-a",
        required=False,
        help="Directory with training alignments",
    )
    data_grp.add_argument(
        "--val-trees", "-T", required=False, help="Directory with validation trees"
    )
    data_grp.add_argument(
        "--val-alignments",
        "-A",
        required=False,
        help="Directory with validation alignments",
    )
    data_grp.add_argument(
        "--train-regex", "-r", default=None, help="Regex to filter training examples"
    )
    data_grp.add_argument(
        "--val-regex", "-R", default=None, help="Regex to filter validation examples"
    )
    data_grp.add_argument(
        "--distance-cache-dir",
        default=None,
        type=str,
        help=(
            "Directory for cached distance targets (.pt). "
            "Uses <distance-cache-dir>/train and <distance-cache-dir>/val."
        ),
    )
    data_grp.add_argument(
        "--precompute-distance-cache",
        action="store_true",
        help=(
            "Precompute and save all target distance vectors before training "
            "(requires --distance-cache-dir)."
        ),
    )
    data_grp.add_argument(
        "--overwrite-distance-cache",
        action="store_true",
        help="Overwrite existing cached target distances when precomputing.",
    )
    data_grp.add_argument(
        "--alignment-cache-dir",
        default=None,
        type=str,
        help=(
            "Directory for cached one-hot alignments (.pt). "
            "Uses <alignment-cache-dir>/train and <alignment-cache-dir>/val."
        ),
    )
    data_grp.add_argument(
        "--precompute-alignment-cache",
        action="store_true",
        help=(
            "Precompute and save all one-hot alignments before training "
            "(requires --alignment-cache-dir)."
        ),
    )
    data_grp.add_argument(
        "--overwrite-alignment-cache",
        action="store_true",
        help="Overwrite existing cached alignments when precomputing.",
    )
